import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

from .attention_processor import LoRALinearLayer

from diffusers.utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    deprecate,
    is_safetensors_available,
    is_transformers_available,
    logging,
)

from .hub_utils import _get_model_file

if is_safetensors_available():
    import safetensors




logger = logging.get_logger(__name__)

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

TEXT_INVERSION_NAME = "learned_embeds.bin"
TEXT_INVERSION_NAME_SAFE = "learned_embeds.safetensors"

CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"

if is_transformers_available():
    from transformers import CLIPTextModel, PreTrainedModel, PreTrainedTokenizer

class PatchedLoraProjection(nn.Module):
    def __init__(self, regular_linear_layer, lora_scale=1, network_alpha=None, rank=4, dtype=None):
        super().__init__()
        self.regular_linear_layer = regular_linear_layer

        device = self.regular_linear_layer.weight.device

        if dtype is None:
            dtype = self.regular_linear_layer.weight.dtype

        self.lora_linear_layer = LoRALinearLayer(
            self.regular_linear_layer.in_features,
            self.regular_linear_layer.out_features,
            network_alpha=network_alpha,
            device=device,
            dtype=dtype,
            rank=rank,
        )

        self.lora_scale = lora_scale

    def forward(self, input):
        return self.regular_linear_layer(input) + self.lora_scale * self.lora_linear_layer(input)

def text_encoder_attn_modules(text_encoder):
    attn_modules = []

    if isinstance(text_encoder, CLIPTextModel):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            name = f"text_model.encoder.layers.{i}.self_attn"
            mod = layer.self_attn
            attn_modules.append((name, mod))
    else:
        raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

    return attn_modules


class LoraLoaderMixin:
    r"""
    Load LoRA layers into [`UNet2DConditionModel`] and
    [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).
    """
    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into self.unet and self.text_encoder.

        All kwargs are forwarded to `self.lora_state_dict`.

        See [`~loaders.LoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.

        See [`~loaders.LoraLoaderMixin.load_lora_into_unet`] for more details on how the state dict is loaded into
        `self.unet`.

        See [`~loaders.LoraLoaderMixin.load_lora_into_text_encoder`] for more details on how the state dict is loaded
        into `self.text_encoder`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].

            kwargs:
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
        """
        state_dict, network_alpha = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
        self.load_lora_into_unet(state_dict, network_alpha=network_alpha, unet=self.unet)
        self.load_lora_into_text_encoder(
            state_dict, network_alpha=network_alpha, text_encoder=self.text_encoder, lora_scale=self.lora_scale
        )

    @classmethod
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        r"""
        Return state dict for lora weights

        <Tip warning={true}>

        We support loading A1111 formatted LoRA checkpoints in a limited capacity.

        This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        """
        # Load the main state dict first which has the LoRA layers for either of
        # UNet and text encoder or both.
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        if use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetensors"
            )

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except (IOError, safetensors.SafetensorError) as e:
                    if not allow_pickle:
                        raise e
                    # try loading non-safetensors weights
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # Convert kohya-ss Style LoRA attn procs to diffusers attn procs
        network_alpha = None
        if all((k.startswith("lora_te_") or k.startswith("lora_unet_")) for k in state_dict.keys()):
            state_dict, network_alpha = cls._convert_kohya_lora_to_diffusers(state_dict)

        return state_dict, network_alpha

    @classmethod
    def load_lora_into_unet(cls, state_dict, network_alpha, unet):
        """
        This will load the LoRA layers specified in `state_dict` into `unet`

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            network_alpha (`float`):
                See `LoRALinearLayer` for more details.
            unet (`UNet2DConditionModel`):
                The UNet model to load the LoRA layers into.
        """

        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())
        if all(key.startswith(cls.unet_name) or key.startswith(cls.text_encoder_name) for key in keys):
            # Load the layers corresponding to UNet.
            unet_keys = [k for k in keys if k.startswith(cls.unet_name)]
            logger.info(f"Loading {cls.unet_name}.")
            unet_lora_state_dict = {
                k.replace(f"{cls.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys
            }
            unet.load_attn_procs(unet_lora_state_dict, network_alpha=network_alpha)

        # Otherwise, we're dealing with the old format. This means the `state_dict` should only
        # contain the module names of the `unet` as its keys WITHOUT any prefix.
        elif not all(
            key.startswith(cls.unet_name) or key.startswith(cls.text_encoder_name) for key in state_dict.keys()
        ):
            unet.load_attn_procs(state_dict)
            warn_message = "You have saved the LoRA weights using the old format. To convert the old LoRA weights to the new format, you can first load them in a dictionary and then create a new dictionary like the following: `new_state_dict = {f'unet'.{module_name}: params for module_name, params in old_state_dict.items()}`."
            warnings.warn(warn_message)

    @classmethod
    def load_lora_into_text_encoder(cls, state_dict, network_alpha, text_encoder, lora_scale=1.0):
        """
        This will load the LoRA layers specified in `state_dict` into `text_encoder`

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The key shoult be prefixed with an
                additional `text_encoder` to distinguish between unet lora layers.
            network_alpha (`float`):
                See `LoRALinearLayer` for more details.
            text_encoder (`CLIPTextModel`):
                The text encoder model to load the LoRA layers into.
            lora_scale (`float`):
                How much to scale the output of the lora linear layer before it is added with the output of the regular
                lora layer.
        """

        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())
        if all(key.startswith(cls.unet_name) or key.startswith(cls.text_encoder_name) for key in keys):
            # Load the layers corresponding to text encoder and make necessary adjustments.
            text_encoder_keys = [k for k in keys if k.startswith(cls.text_encoder_name)]
            text_encoder_lora_state_dict = {
                k.replace(f"{cls.text_encoder_name}.", ""): v for k, v in state_dict.items() if k in text_encoder_keys
            }
            if len(text_encoder_lora_state_dict) > 0:
                logger.info(f"Loading {cls.text_encoder_name}.")

                if any("to_out_lora" in k for k in text_encoder_lora_state_dict.keys()):
                    # Convert from the old naming convention to the new naming convention.
                    #
                    # Previously, the old LoRA layers were stored on the state dict at the
                    # same level as the attention block i.e.
                    # `text_model.encoder.layers.11.self_attn.to_out_lora.up.weight`.
                    #
                    # This is no actual module at that point, they were monkey patched on to the
                    # existing module. We want to be able to load them via their actual state dict.
                    # They're in `PatchedLoraProjection.lora_linear_layer` now.
                    for name, _ in text_encoder_attn_modules(text_encoder):
                        text_encoder_lora_state_dict[
                            f"{name}.q_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_q_lora.up.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.k_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_k_lora.up.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.v_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_v_lora.up.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.out_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_out_lora.up.weight")

                        text_encoder_lora_state_dict[
                            f"{name}.q_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_q_lora.down.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.k_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_k_lora.down.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.v_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_v_lora.down.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.out_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_out_lora.down.weight")

                rank = text_encoder_lora_state_dict[
                    "text_model.encoder.layers.0.self_attn.out_proj.lora_linear_layer.up.weight"
                ].shape[1]

                cls._modify_text_encoder(text_encoder, lora_scale, network_alpha, rank=rank)

                # set correct dtype & device
                text_encoder_lora_state_dict = {
                    k: v.to(device=text_encoder.device, dtype=text_encoder.dtype)
                    for k, v in text_encoder_lora_state_dict.items()
                }

                load_state_dict_results = text_encoder.load_state_dict(text_encoder_lora_state_dict, strict=False)
                if len(load_state_dict_results.unexpected_keys) != 0:
                    raise ValueError(
                        f"failed to load text encoder state dict, unexpected keys: {load_state_dict_results.unexpected_keys}"
                    )

    @property
    def lora_scale(self) -> float:
        # property function that returns the lora scale which can be set at run time by the pipeline.
        # if _lora_scale has not been set, return 1
        return self._lora_scale if hasattr(self, "_lora_scale") else 1.0

    def _remove_text_encoder_monkey_patch(self):
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder)

    @classmethod
    def _remove_text_encoder_monkey_patch_classmethod(cls, text_encoder):
        for _, attn_module in text_encoder_attn_modules(text_encoder):
            if isinstance(attn_module.q_proj, PatchedLoraProjection):
                attn_module.q_proj = attn_module.q_proj.regular_linear_layer
                attn_module.k_proj = attn_module.k_proj.regular_linear_layer
                attn_module.v_proj = attn_module.v_proj.regular_linear_layer
                attn_module.out_proj = attn_module.out_proj.regular_linear_layer

    @classmethod
    def _modify_text_encoder(cls, text_encoder, lora_scale=1, network_alpha=None, rank=4, dtype=None):
        r"""
        Monkey-patches the forward passes of attention modules of the text encoder.
        """

        # First, remove any monkey-patch that might have been applied before
        cls._remove_text_encoder_monkey_patch_classmethod(text_encoder)

        lora_parameters = []

        for _, attn_module in text_encoder_attn_modules(text_encoder):
            attn_module.q_proj = PatchedLoraProjection(
                attn_module.q_proj, lora_scale, network_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.q_proj.lora_linear_layer.parameters())

            attn_module.k_proj = PatchedLoraProjection(
                attn_module.k_proj, lora_scale, network_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.k_proj.lora_linear_layer.parameters())

            attn_module.v_proj = PatchedLoraProjection(
                attn_module.v_proj, lora_scale, network_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.v_proj.lora_linear_layer.parameters())

            attn_module.out_proj = PatchedLoraProjection(
                attn_module.out_proj, lora_scale, network_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.out_proj.lora_linear_layer.parameters())

        return lora_parameters

    @classmethod
    def save_lora_weights(
        self,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, torch.nn.Module] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = False,
    ):
        r"""
        Save the LoRA parameters corresponding to the UNet and text encoder.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            unet_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the UNet.
            text_encoder_lora_layers (`Dict[str, torch.nn.Module] or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text
                encoder LoRA state dict because it comes 🤗 Transformers.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})

            else:
                save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        # Create a flat dictionary.
        state_dict = {}
        if unet_lora_layers is not None:
            weights = (
                unet_lora_layers.state_dict() if isinstance(unet_lora_layers, torch.nn.Module) else unet_lora_layers
            )

            unet_lora_state_dict = {f"{self.unet_name}.{module_name}": param for module_name, param in weights.items()}
            state_dict.update(unet_lora_state_dict)

        if text_encoder_lora_layers is not None:
            weights = (
                text_encoder_lora_layers.state_dict()
                if isinstance(text_encoder_lora_layers, torch.nn.Module)
                else text_encoder_lora_layers
            )

            text_encoder_lora_state_dict = {
                f"{self.text_encoder_name}.{module_name}": param for module_name, param in weights.items()
            }
            state_dict.update(text_encoder_lora_state_dict)

        # Save the model
        if weight_name is None:
            if safe_serialization:
                weight_name = LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = LORA_WEIGHT_NAME

        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")

    @classmethod
    def _convert_kohya_lora_to_diffusers(cls, state_dict):
        unet_state_dict = {}
        te_state_dict = {}
        network_alpha = None

        for key, value in state_dict.items():
            if "lora_down" in key:
                lora_name = key.split(".")[0]
                lora_name_up = lora_name + ".lora_up.weight"
                lora_name_alpha = lora_name + ".alpha"
                if lora_name_alpha in state_dict:
                    alpha = state_dict[lora_name_alpha].item()
                    if network_alpha is None:
                        network_alpha = alpha
                    elif network_alpha != alpha:
                        raise ValueError("Network alpha is not consistent")

                if lora_name.startswith("lora_unet_"):
                    diffusers_name = key.replace("lora_unet_", "").replace("_", ".")
                    diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")
                    diffusers_name = diffusers_name.replace("mid.block", "mid_block")
                    diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")
                    diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
                    diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
                    diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
                    diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
                    diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
                    if "transformer_blocks" in diffusers_name:
                        if "attn1" in diffusers_name or "attn2" in diffusers_name:
                            diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
                            diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
                            unet_state_dict[diffusers_name] = value
                            unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]
                elif lora_name.startswith("lora_te_"):
                    diffusers_name = key.replace("lora_te_", "").replace("_", ".")
                    diffusers_name = diffusers_name.replace("text.model", "text_model")
                    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                    diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                    diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                    diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                    diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                    if "self_attn" in diffusers_name:
                        te_state_dict[diffusers_name] = value
                        te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict[lora_name_up]

        unet_state_dict = {f"{UNET_NAME}.{module_name}": params for module_name, params in unet_state_dict.items()}
        te_state_dict = {f"{TEXT_ENCODER_NAME}.{module_name}": params for module_name, params in te_state_dict.items()}
        new_state_dict = {**unet_state_dict, **te_state_dict}
        return new_state_dict, network_alpha




class AttnProcsLayers(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k

            raise ValueError(
                f"There seems to be a problem with the state_dict: {set(state_dict.keys())}. {key} has to have one of {self.split_keys}."
            )

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)

