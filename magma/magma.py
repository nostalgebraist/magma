from pathlib import Path
from os.path import exists
import os
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, List
from torchtyping import TensorType
from transformers.file_utils import ModelOutput
from magma.config import MultimodalConfig

from magma.utils import get_tokenizer
from .language_model import get_gptj
from .adapters import (
    Adapter,
    ParallelAdapter,
    AdapterWrapper,
    ParallelAdapterWrapper,
)
from .image_prefix import ImagePrefix
from .sampling import generate
from .utils import build_labels, is_url, print_main, download_checkpoint
from .image_input import ImageInput
from .transforms import get_transforms

# ------------------------- Magma main class ----------------------------------


class Magma(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()

        if isinstance(config, (str, Path)):
            config = MultimodalConfig.from_yml(
                config
            )  # load config from yml file if config is a string
        else:
            assert isinstance(config, MultimodalConfig)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config
        self.lm = get_gptj().to(self.device)
        self.seq_len = self.lm.config.max_position_embeddings

        self.tokenizer = get_tokenizer("gpt2", sequence_length=self.seq_len)

        self.image_token = self.tokenizer.cls_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.lm.resize_token_embeddings(len(self.tokenizer))
        self.lm.config.pad_token_id = self.tokenizer.eos_token_id
        self.word_embedding = self.lm.transformer.wte.to(device)
        self.transformer = self.lm.transformer.h

        # adapter settings
        self.mlp_adapter_added, self.attn_adapter_added = False, False

        self.image_prefix = ImagePrefix(
            config=config,
            out_dim=self.lm.config.hidden_size,
        ).to(self.device)

        # might change based on the type of image encoder, so get from prefix instead of config
        self.image_prefix_seq_len = self.image_prefix.out_seq_len

        self.transforms = get_transforms(
            config.image_size,
            config.encoder_name,
            input_resolution=self.image_prefix.enc.input_resolution,
        )

        # add adapters
        self.adapter_map = {}
        if config.adapter_config:
            mlp_config = deepcopy(config.adapter_config.get("mlp", None))
            if mlp_config:
                assert mlp_config.get("adapter_type") is not None
                self.build_adapters(
                    location="mlp",
                    adapter_type=mlp_config.pop("adapter_type"),
                    downsample_factor=mlp_config.pop("downsample_factor", 4),
                    **mlp_config,
                )
            attn_config = deepcopy(config.adapter_config.get("attention", None))
            if attn_config:
                assert attn_config.get("adapter_type") is not None
                self.build_adapters(
                    location="attention",
                    adapter_type=attn_config.pop("adapter_type"),
                    **attn_config,
                )

        self.add_adapters()

        # freeze parameters
        if config.freeze_lm:
            for name, param in self.lm.named_parameters():  # freeze lm weights
                if config.adapter_config and "adapter" in name:
                    param.requires_grad = True

        if config.freeze_img_encoder:
            for param in self.image_prefix.enc.parameters():
                param.requires_grad = False

    def build_adapters(
        self,
        downsample_factor: int = 4,
        adapter_type = "normal",
        location = "mlp",
        ff_attr: str = "mlp",
        attn_attr: str = "attn",
        **adapter_kwargs,
    ):
        adapter_map = {}

        for l in range(len(self.transformer)):
            if location == "mlp":
                mlp = getattr(self.transformer[l], ff_attr)
                if adapter_type in ["parallel", "scaled_parallel"]:
                    adpt = ParallelAdapter(
                        module=None,
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        scaled=adapter_type == "scaled_parallel",
                        **adapter_kwargs,
                    )
                else:
                    adpt = Adapter(
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs,
                    )
                adapter_map[(l, 'mlp')] = adpt
            else:
                attn = getattr(self.transformer[l], attn_attr)
                if adapter_type in ["parallel", "scaled_parallel"]:
                    adapter_layer = ParallelAdapterWrapper(
                        module=None,
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        scaled="scaled" in adapter_type,
                        **adapter_kwargs,
                    )
                else:
                    adapter_layer = AdapterWrapper(
                        attn_block=None,
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs,
                    )
                adapter_map[(l, 'mlp')] = adapter_layer
        self.adapter_map.update(adapter_map)

    def add_adapters(
        self,
        ff_attr: str = "mlp",
        attn_attr: str = "attn",
    ):
        for l in range(len(self.transformer)):
            if (l, 'mlp') in self.adapter_map:
                adpt = self.adapter_map.pop((l, 'mlp'))
                if self.mlp_adapter_added:
                    raise ValueError("Adapter layer already added")
                mlp = getattr(self.transformer[l], ff_attr)
                if isinstance(adpt, ParallelAdapterWrapper):
                    adapter_layer = adpt
                    setattr(adapter_layer, 'module', mlp)
                else:
                    adapter_layer = nn.Sequential(
                        *[
                            mlp,
                            adpt,
                        ]
                    )
                setattr(self.transformer[l], ff_attr, adapter_layer)
            elif (l, 'attn') in self.adapter_map:
                adpt = self.adapter_map.pop((l, 'attn'))
                if self.attn_adapter_added:
                    raise ValueError("Adapter layer already added")
                attn = getattr(self.transformer[l], attn_attr)
                setattr(adapter_layer, 'module', attn)
                setattr(self.transformer[l], attn_attr, adapter_layer)

    def detach_adapters(
        self,
        ff_attr: str = "mlp",
        attn_attr: str = "attn",
    ):
        for l in range(len(self.transformer)):
            mlp = getattr(self.transformer[l], ff_attr)
            if isinstance(mlp, ParallelAdapterWrapper):
                orig = getattr(mlp, 'module')
                self.adapter_map[(l, 'mlp')] = mlp
                self.adapter_map[(l, 'mlp')].module = None
                setattr(self.transformer[l], ff_attr, orig)
            elif isinstance(mlp, nn.Sequential):
                orig = mlp[0]
                self.adapter_map[(l, 'mlp')] = mlp[1]
                setattr(self.transformer[l], ff_attr, orig)

            attn = getattr(self.transformer[l], attn_attr)
            if isinstance(attn, AdapterWrapper) or isinstance(attn, ParallelAdapterWrapper):
                orig = getattr(attn, 'module')
                self.adapter_map[(l, 'attn')] = attn
                self.adapter_map[(l, 'attn')].module = None
                setattr(self.transformer[l], attn_attr, orig)

    def preprocess_inputs(self, input_list: list, embed = True) -> List[torch.Tensor]:
        """
        Expects a list of strings and instances of ImageInput
        Converts them into a list of tensors and then optionally runs self.embed over it
        """
        for i in range(len(input_list)):
            inp = input_list[i]
            if isinstance(inp, str):
                input_list[i] = self.tokenizer.encode(inp, return_tensors="pt")
            elif isinstance(inp, ImageInput):
                input_list[i] = inp.get_transformed_image(transform_fn = self.transforms)
            else:
                raise Exception(f'Invalid input type:{type(inp)}')

        if embed == True:
            return self.embed(input_list)
        else:
            return input_list

    def embed(self, inputs: List[torch.Tensor]) -> TensorType["b", "s", "d"]:
        """
        Embeds a list of tensors In the correct format to input into the LM (b, s, d).
        For each tensor, if it's 2d assume it's text and use word embedding,
        if it's 4d, assume it's an image, and use image_prefix to embed.
        """
        emb_list = []
        for x in inputs:
            if x.ndim == 2:
                x = x.to(self.device)
                emb_list.append(self.word_embedding(x))
            elif x.ndim == 4:
                x = x.to(self.device).half()
                image_embeddings = self.image_prefix(x)
                emb_list.append(image_embeddings)
            else:
                raise ValueError(f"Expected 2d or 4d tensor, got {x.ndim}d")
        return torch.cat(emb_list, dim=1)

    @torch.no_grad()
    def generate(
        self,
        embeddings: TensorType["b", "s", "d"],
        max_steps: int = 100,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 0.9,
        decode: bool = True,
    ):
        """
        Generates captions for a batch of embeddings.
        """

        return generate(
            self,
            embeddings=embeddings,
            max_steps=max_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            decode=decode,
        )

    def forward(
        self,
        images: TensorType["b", "c", "h", "w"] = None,
        captions: Optional[TensorType["b", "seq"]] = None,
        output_hidden_states: bool = False,
        input_embeddings: TensorType["b", "s", "d"] = None,
    ) -> ModelOutput:
        assert captions is not None, "Must provide captions in training"
        assert any([i is not None for i in [images, input_embeddings]]) and not all(
            [i is not None for i in [images, input_embeddings]]
        ), "Pass in either images, or input embeddings, not both."
        assert (
            captions.shape[1] == self.seq_len
        ), f"in training, captions should be padded to sequence length ({self.seq_len}), but are length {captions.shape[1]}"

        if input_embeddings is None:
            input_embeddings = self.image_prefix(images)
        labels = build_labels(
            input_embeddings, captions, self.eos_token, self.device
        )  # build labels from input_embeddings
        word_embeddings = self.word_embedding(captions)

        # join together
        input_embeddings = torch.cat(
            (
                input_embeddings,
                word_embeddings[:, : -input_embeddings.shape[1], :],
            ),  # remove padding in the word embedding before concatenating
            dim=1,
        )

        # forward joined embeddings through lm
        lm_outputs = self.lm(
            inputs_embeds=input_embeddings,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )

        return lm_outputs

    @classmethod
    def from_checkpoint(cls, config_path, checkpoint_path, device = 'cpu'):
        """
        Loads a model checkpoint from disk / downlods from url if not present
        """

        checkpoint_url = 'https://bit.ly/aleph-alpha-magma-download'

        if exists(checkpoint_path) ==  False:
            print_main(f'checkpoint: {checkpoint_path} does not exist, downloading model')
            download_checkpoint(checkpoint_url = checkpoint_url, save_as = checkpoint_path)

        model = cls(config = config_path)

        sd = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        if "module" in sd.keys():
            sd = sd["module"]

        print_main('loading checkpoint magma')
        model.load_state_dict(sd, strict=False)
        print_main("magma model successfully loaded")

        model.half().to(device)
        return model

    def save_split_checkpoint(self, path, save_lm=False):
        os.makedirs(path, exist_ok=True)

        self.detach_adapters()

        print('saving: image prefix')
        torch.save(self.image_prefix.state_dict(), f'{path}/image_prefix.pt')

        print('saving: adapters')
        adapter_map_sd = {k: self.adapter_map[k].state_dict() for k in self.adapter_map}
        torch.save(adapter_map_sd, f'{path}/adapter_map.pt')

        if save_lm:
            print('saving: lm')
            torch.save(self.lm.state_dict(), f'{path}/lm.pt')

        self.add_adapters()

    @classmethod
    def from_split_checkpoint(cls, config_path, path, lm_path_or_state_dict, device = 'cpu'):
        model = cls(config = config_path)

        model.detach_adapters()

        model.lm.resize_token_embeddings(50400)

        if isinstance(lm_path_or_state_dict, str):
            # path
            lm_state_dict = torch.load(lm_path_or_state_dict)
        else:
            lm_state_dict = lm_path_or_state_dict

        model.lm.load_state_dict(lm_state_dict)

        model.lm.resize_token_embeddings(50258)

        torch.load(f'{path}/image_prefix.pt')
        model.image_prefix.load_state_dict()

        adapter_map_sd = torch.load(f'{path}/adapter_map.pt')
        for k in model.adapter_map:
            print(f'loading sd for {k}')  # debug
            model.adapter_map[k].load_state_dict(adapter_map_sd[k])

        model.add_adapters()

        model.half().to(device)
        return model
