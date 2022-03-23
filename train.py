import torch
import os
import deepspeed
import deepspeed.engine
import wandb
from torch.utils.data import random_split, ConcatDataset
from torch.optim import AdamW
from tqdm import tqdm
from functools import partial
from magma.datasets import (
    collate_fn,
    ImgCptDataset,
)
from magma.magma import (
    Magma,
)
from magma.config import load_config
from magma.utils import (
    is_main,
    cycle,
    parse_args,
    wandb_log,
    wandb_init,
    save_model,
    load_model,
    print_main,
    configure_param_groups,
)
from magma.train_loop import (
    eval_step,
    inference_step,
    train_step,
)
from deepspeed.utils import log_dist


# TODO: hack _save_checkpoint and _load_checkpoint
# def _save_checkpoint(self, save_dir, tag, client_state={}):
#
#     save_path = self._get_ckpt_name(save_dir, tag)
#     # A hack to save the checkpointing directory. Pipeline parallelism overrides
#     # module_state_dict() and uses this path to save the model. module_state_dict()
#     # then instead just returns None.
#     self._curr_ckpt_path = os.path.join(save_dir, tag)
#
#     state = dict(module=self.module_state_dict(),
#                  buffer_names=self._get_buffer_names(),
#                  optimizer=self.optimizer.state_dict()
#                  if self.optimizer and not self.zero_optimization() else None,
#                  param_shapes=self._get_zero_param_shapes()
#                  if self.optimizer and self.zero_optimization() else None,
#                  lr_scheduler=self.lr_scheduler.state_dict()
#                  if self.lr_scheduler is not None else None,
#                  sparse_tensor_module_names=self.sparse_tensor_module_names,
#                  skipped_steps=self.skipped_steps,
#                  global_steps=self.global_steps,
#                  global_samples=self.global_samples,
#                  dp_world_size=self.dp_world_size,
#                  mp_world_size=self.mp_world_size,
#                  ds_config=self.config,
#                  ds_version=version)
#     state.update(client_state)
#
#     log_dist(message=f'Saving model checkpoint: {save_path}', ranks=[0, 1])
#     torch.save(state, save_path)
#     self._curr_save_path = None
#
#
# deepspeed.engine.DeepSpeedEngine._save_checkpoint = _save_checkpoint


def _load_img_cpt_datasets(dataset_dir, tokenizer, transforms):
    if isinstance(dataset_dir, (list, tuple)):
        return ConcatDataset(
            [_load_img_cpt_datasets(d, tokenizer, transforms) for d in dataset_dir]
        )
    elif isinstance(dataset_dir, str):
        return ImgCptDataset(dataset_dir, tokenizer=tokenizer, transforms=transforms)
    else:
        raise TypeError("dataset dir wrong type")


def get_pretraining_datasets(config, tokenizer, transforms):
    # if config.train_dataset_dir is a list, load all datasets + join together
    train_dataset = _load_img_cpt_datasets(
        config.train_dataset_dir, tokenizer, transforms
    )
    # if no dedicated eval sets are given, use a percentage of the train dataset
    if config.eval_dataset_dir is None:
        eval_len = int(len(train_dataset) * config.eval_dataset_pct)
        train_len = len(train_dataset) - eval_len
        print(
            f"Randomly splitting train_dataset into two datasets of length {train_len} and {eval_len}"
        )
        train_dataset, eval_dataset = random_split(train_dataset, [train_len, eval_len])
    else:
        eval_dataset = _load_img_cpt_datasets(
            config.eval_dataset_dir, tokenizer, transforms
        )

    print_main(f"Loaded train dataset with {len(train_dataset)} samples")
    print_main(f"Loaded eval dataset with {len(eval_dataset)} samples")

    return train_dataset, eval_dataset


# tell tokenizers not to do parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    # parse command line arguments:
    args = parse_args()
    # fakeargs = FakeArgs(args.config)
    deepspeed.init_distributed()


    # load model + tokenizer:
    ckpt_path = load_config(args.config).get('ckpt_path')
    if ckpt_path:
        print('loading split')
        model = Magma.from_split_checkpoint(args.config, ckpt_path, os.path.join(ckpt_path, 'lm.pt'), device='cuda:0')
    else:
        model = Magma(
            args.config
        )
    tokenizer, config, transforms = model.tokenizer, model.config, model.transforms

    # filter frozen from trainable parameters:
    trainable_parameters = configure_param_groups(model, config)

    # load data:
    train_dataset, eval_dataset = get_pretraining_datasets(
        config, tokenizer, transforms
    )

    print_main(f"Loaded train dataset with {len(train_dataset)} samples")
    print_main(f"Loaded eval dataset with {len(eval_dataset)} samples")

    opt = AdamW(
        trainable_parameters,
        config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    model_engine, opt, train_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=opt,
        model_parameters=trainable_parameters,
        training_data=train_dataset,
        collate_fn=partial(collate_fn, seq_len=model.seq_len),
        config_params=config.deepspeed_config_params,
    )
    eval_loader = cycle(model_engine.deepspeed_io(eval_dataset))
    train_loader = cycle(train_loader)

    # initialize training
    global_step = 0
    if config.load:
        # loads a deepspeed checkpoint if provided. For finetuning, set load_optimizer to false
        previous_global_step = load_model(
            model_engine,
            config.load,
            load_optimizer_states=config.load_optimizer,
            load_lr_scheduler_states=config.load_optimizer,
        )

        if config.load_optimizer:
            global_step = previous_global_step

    pbar = tqdm(
        range(0, config.train_steps),
        desc="training...",
        initial=global_step,
        total=config.train_steps,
        disable=not is_main(),
    )
    wandb_init(
        project=config.wandb_project,
        name=config.name or wandb.util.generate_id(),
        config=config,
    )

    # training loop
    for i in pbar:
        if global_step >= config.train_steps:
            break

        ##### train step
        loss = train_step(config, train_loader, model_engine)

        global_step += 1

        if global_step % config.log_every == 0:
            pbar.set_description(f"training... Step: {global_step} Loss: {loss}")
            current_lr = (
                [lr for lr in lr_scheduler.get_lr()]
                if lr_scheduler is not None
                else config.lr
            )
            to_log = {"train/loss": loss, "train/lr": current_lr}
            wandb_log(to_log, step=global_step)

        ##### Evaluation phase
        if global_step % config.eval_every == 0:
            model_engine.eval()
            with torch.no_grad():

                ##### eval step:
                eval_loss = eval_step(config, eval_loader, model_engine)

                wandb_log({"eval/loss": eval_loss}, step=global_step)
                pbar.set_description(
                    f"evaluating... Step: {global_step} Eval Loss: {eval_loss}"
                )

                ##### inference:
                image_grid, caption = inference_step(config, eval_loader, model_engine)
                wandb_log(
                    {"inference/image": wandb.Image(image_grid, caption=caption)},
                    step=global_step,
                )

            model_engine.train()

        ##### Save model
        if global_step % config.save_every == 0:
            if config.save is not None:
                save_model(model_engine, config.save, global_step)
                print_main(f"saving model at step {global_step}")

    ##### Save model after training is finished
    if config.save is not None:
        save_model(model_engine, config.save, global_step)
        print_main(f"saving model at end of training (step {global_step})")
