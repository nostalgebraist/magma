import torch
from tqdm import tqdm
from .utils import reduce_losses, to_cuda_half
from torchvision.utils import make_grid


def train_step(config, train_loader, model_engine, use_torch_amp=False, grad_scaler=None):
    losses = []

    for _ in range(config.gradient_accumulation_steps):
        images, captions = next(train_loader)
        images, captions = images.half().cuda(), captions.cuda()
        if config.run_blind:
            images = torch.zeros_like(images)
        with torch.cuda.amp.autocast(enabled=use_torch_amp):
            outputs = model_engine(images, captions)
        loss = outputs.loss
        losses.append(loss)
        model_engine.backward(loss.float())
        if use_torch_amp and (grad_scaler is not None):
            model_engine.backward(grad_scaler.scale(loss))
            grad_scaler.step(model_engine)
            grad_scaler.update()
        else:
            model_engine.step()

    return reduce_losses(torch.mean(torch.stack(losses))).item(), grad_scaler


def train_step_classification(config, train_loader, model_engine, return_accuracy=True):
    losses = []
    if return_accuracy:
        accuracies = []
    for _ in range(config.gradient_accumulation_steps):
        images, captions, class_labels = next(train_loader)
        images, captions, class_labels = to_cuda_half(images, captions, class_labels)
        if config.run_blind:
            images = torch.zeros_like(images)
        loss, logits = model_engine(images, captions, class_labels)
        losses.append(loss)
        if return_accuracy:
            argmax_pred = logits.argmax(dim=-1)
            accuracies.append((argmax_pred == class_labels).float().mean())
        model_engine.backward(loss)
        model_engine.step()

    loss_reduced = reduce_losses(torch.mean(torch.stack(losses))).item()
    if return_accuracy:
        accuracy_reduced = reduce_losses(torch.mean(torch.stack(accuracies))).item()
        return loss_reduced, accuracy_reduced
    return loss_reduced


def eval_step(config, eval_loader, model_engine, use_torch_amp=False):
    losses = []

    for i in tqdm(range(config.eval_steps), "evaluating..."):
        images, captions = next(eval_loader)
        images, captions = images.half().cuda(), captions.cuda()
        if config.run_blind:
            images = torch.zeros_like(images)
        with torch.cuda.amp.autocast(enabled=use_torch_amp):
            outputs = model_engine(images, captions)
        loss = outputs.loss
        if torch.isnan(loss).any():
            print('found nan, skipping')
            continue
        losses.append(loss)

    return reduce_losses(torch.mean(torch.stack(losses))).item()


def eval_step_classification(config, train_loader, model_engine, return_accuracy=True):
    losses = []
    if return_accuracy:
        accuracies = []
    for _ in range(config.gradient_accumulation_steps):
        images, captions, class_labels = next(train_loader)
        images, captions, class_labels = to_cuda_half(images, captions, class_labels)
        if config.run_blind:
            images = torch.zeros_like(images)
        loss, logits = model_engine(images, captions, class_labels)
        losses.append(loss)
        if return_accuracy:
            argmax_pred = logits.argmax(dim=-1)
            accuracies.append((argmax_pred == class_labels).float().mean())

    loss_reduced = reduce_losses(torch.mean(torch.stack(losses))).item()
    if return_accuracy:
        accuracy_reduced = reduce_losses(torch.mean(torch.stack(accuracies))).item()
        return loss_reduced, accuracy_reduced
    return loss_reduced


def inference_step(config, eval_loader, model_engine, use_torch_amp=False):
    images, _ = next(eval_loader)
    images = images.half().cuda()
    if config.run_blind:
        images = torch.zeros_like(images)
    with torch.cuda.amp.autocast(enabled=use_torch_amp):
        captions = model_engine(
            images, captions=None, inference=True
        )  # [caption1, caption2, ... b]
    width = min(2, images.shape[0])
    image_grid = make_grid(images[:width])
    caption = ""
    for i in range(width):
        caption += f"Caption {i}: \n{captions[i]}\n"
    return image_grid, caption
