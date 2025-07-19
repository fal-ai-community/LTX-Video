# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py
import copy
import inspect
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LTXVideoLoraLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import deprecate, logging, set_weights_and_activate_adapters
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    latent_to_pixel_coords,
    normalize_latents,
    un_normalize_latents,
    vae_decode,
    vae_encode,
)
from ltx_video.models.transformers.symmetric_patchifier import Patchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.schedulers.rf import TimestepShifter
from ltx_video.utils.prompt_enhance_utils import generate_cinematic_prompt
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

try:
    import diffusers.loaders.peft as peft_module

    peft_module._SET_ADAPTER_SCALE_FN_MAPPING["Transformer3DModel"] = (
        peft_module._SET_ADAPTER_SCALE_FN_MAPPING["LTXVideoTransformer3DModel"]
    )
except:
    raise

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


ASPECT_RATIO_1024_BIN = {
    "0.25": [512.0, 2048.0],
    "0.28": [512.0, 1856.0],
    "0.32": [576.0, 1792.0],
    "0.33": [576.0, 1728.0],
    "0.35": [576.0, 1664.0],
    "0.4": [640.0, 1600.0],
    "0.42": [640.0, 1536.0],
    "0.48": [704.0, 1472.0],
    "0.5": [704.0, 1408.0],
    "0.52": [704.0, 1344.0],
    "0.57": [768.0, 1344.0],
    "0.6": [768.0, 1280.0],
    "0.68": [832.0, 1216.0],
    "0.72": [832.0, 1152.0],
    "0.78": [896.0, 1152.0],
    "0.82": [896.0, 1088.0],
    "0.88": [960.0, 1088.0],
    "0.94": [960.0, 1024.0],
    "1.0": [1024.0, 1024.0],
    "1.07": [1024.0, 960.0],
    "1.13": [1088.0, 960.0],
    "1.21": [1088.0, 896.0],
    "1.29": [1152.0, 896.0],
    "1.38": [1152.0, 832.0],
    "1.46": [1216.0, 832.0],
    "1.67": [1280.0, 768.0],
    "1.75": [1344.0, 768.0],
    "2.0": [1408.0, 704.0],
    "2.09": [1472.0, 704.0],
    "2.4": [1536.0, 640.0],
    "2.5": [1600.0, 640.0],
    "3.0": [1728.0, 576.0],
    "4.0": [2048.0, 512.0],
}

ASPECT_RATIO_512_BIN = {
    "0.25": [256.0, 1024.0],
    "0.28": [256.0, 928.0],
    "0.32": [288.0, 896.0],
    "0.33": [288.0, 864.0],
    "0.35": [288.0, 832.0],
    "0.4": [320.0, 800.0],
    "0.42": [320.0, 768.0],
    "0.48": [352.0, 736.0],
    "0.5": [352.0, 704.0],
    "0.52": [352.0, 672.0],
    "0.57": [384.0, 672.0],
    "0.6": [384.0, 640.0],
    "0.68": [416.0, 608.0],
    "0.72": [416.0, 576.0],
    "0.78": [448.0, 576.0],
    "0.82": [448.0, 544.0],
    "0.88": [480.0, 544.0],
    "0.94": [480.0, 512.0],
    "1.0": [512.0, 512.0],
    "1.07": [512.0, 480.0],
    "1.13": [544.0, 480.0],
    "1.21": [544.0, 448.0],
    "1.29": [576.0, 448.0],
    "1.38": [576.0, 416.0],
    "1.46": [608.0, 416.0],
    "1.67": [640.0, 384.0],
    "1.75": [672.0, 384.0],
    "2.0": [704.0, 352.0],
    "2.09": [736.0, 352.0],
    "2.4": [768.0, 320.0],
    "2.5": [800.0, 320.0],
    "3.0": [864.0, 288.0],
    "4.0": [1024.0, 256.0],
}


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    skip_initial_inference_steps: int = 0,
    skip_final_inference_steps: int = 0,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
            timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
            must be `None`.
        max_timestep ('float', *optional*, defaults to 1.0):
            The initial noising level for image-to-image/video-to-video. The list if timestamps will be
            truncated to start with a timestamp greater or equal to this.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

        if (
            skip_initial_inference_steps < 0
            or skip_final_inference_steps < 0
            or skip_initial_inference_steps + skip_final_inference_steps
            >= num_inference_steps
        ):
            raise ValueError(
                "invalid skip inference step values: must be non-negative and the sum of skip_initial_inference_steps and skip_final_inference_steps must be less than the number of inference steps"
            )

        timesteps = timesteps[
            skip_initial_inference_steps : len(timesteps) - skip_final_inference_steps
        ]
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        num_inference_steps = len(timesteps)

    return timesteps, num_inference_steps


def linear_overlap_blend(
    existing_latents: torch.Tensor,
    new_latents: torch.Tensor,
    overlap_frames: int,
    temporal_axis: int = 2,
) -> torch.Tensor:
    """
    Blend new latents with existing latents using linear interpolation in the overlap region.

    Args:
        existing_latents: The existing latent tensor
        new_latents: The new latent tensor to blend in
        overlap_frames: Number of frames to blend over
        temporal_axis: The temporal dimension axis

    Returns:
        Blended latent tensor
    """
    if overlap_frames <= 0:
        # No overlap, just concatenate
        return torch.cat([existing_latents, new_latents], dim=temporal_axis)

    # Extract parts
    existing_non_overlap = existing_latents.narrow(
        temporal_axis, 0, existing_latents.size(temporal_axis) - overlap_frames
    )
    existing_overlap = existing_latents.narrow(
        temporal_axis,
        existing_latents.size(temporal_axis) - overlap_frames,
        overlap_frames,
    )
    new_overlap = new_latents.narrow(temporal_axis, 0, overlap_frames)
    new_non_overlap = new_latents.narrow(
        temporal_axis, overlap_frames, new_latents.size(temporal_axis) - overlap_frames
    )

    # Create blending weights
    blend_weights = torch.linspace(
        1.0, 0.0, overlap_frames, device=existing_latents.device
    )
    blend_weights = (
        blend_weights.view(1, 1, overlap_frames, 1, 1)
        if temporal_axis == 2
        else blend_weights
    )

    # Blend the overlap region
    blended_overlap = existing_overlap * blend_weights + new_overlap * (
        1.0 - blend_weights
    )

    # Concatenate all parts
    return torch.cat(
        [existing_non_overlap, blended_overlap, new_non_overlap], dim=temporal_axis
    )


@dataclass
class ConditioningItem:
    """
    Unified conditioning item for all types of conditioning in LTX-Video.

    This can represent:
    - Regular image/video conditioning (type="image" or "video")
    - IC-LoRA guiding video (type="guiding")
    - Any future conditioning types

    Attributes:
        media_item (torch.Tensor): The conditioning data in pixel space.
            - For all types: shape=(b, 3, f, h, w) where b=batch, f=frames, h=height, w=width
            - Guiding latents can be passed as pixel-space videos (depth/pose/etc) or already-encoded videos
        conditioning_type (str): Type of conditioning - "image", "video", or "guiding"
        media_frame_number (int): The start-frame number in the generated video
        conditioning_strength (float): The strength of the conditioning (0.0-1.0)
        media_x (Optional[int]): Optional left x coordinate (for spatial conditioning)
        media_y (Optional[int]): Optional top y coordinate (for spatial conditioning)
    """

    media_item: torch.Tensor
    conditioning_type: str = "image"  # "image", "video", or "guiding"
    media_frame_number: int = 0
    conditioning_strength: float = 1.0
    media_x: Optional[int] = None
    media_y: Optional[int] = None

    @property
    def is_latent_space(self) -> bool:
        """
        Returns true if the item is already in latent space.
        """
        return self.media_item.shape[1] not in [3, 1]


class LTXVideoPipeline(DiffusionPipeline, LTXVideoLoraLoaderMixin):
    r"""
    Pipeline for text-to-image generation using LTX-Video.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. This uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`Transformer2DModel`]):
            A text conditioned `Transformer2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )  # noqa

    _optional_components = [
        "tokenizer",
        "text_encoder",
        "prompt_enhancer_image_caption_model",
        "prompt_enhancer_image_caption_processor",
        "prompt_enhancer_llm_model",
        "prompt_enhancer_llm_tokenizer",
    ]
    model_cpu_offload_seq = "prompt_enhancer_image_caption_model->prompt_enhancer_llm_model->text_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: Transformer3DModel,
        scheduler: DPMSolverMultistepScheduler,
        patchifier: Patchifier,
        prompt_enhancer_image_caption_model: AutoModelForCausalLM,
        prompt_enhancer_image_caption_processor: AutoProcessor,
        prompt_enhancer_llm_model: AutoModelForCausalLM,
        prompt_enhancer_llm_tokenizer: AutoTokenizer,
        allowed_inference_steps: Optional[List[float]] = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            patchifier=patchifier,
            prompt_enhancer_image_caption_model=prompt_enhancer_image_caption_model,
            prompt_enhancer_image_caption_processor=prompt_enhancer_image_caption_processor,
            prompt_enhancer_llm_model=prompt_enhancer_llm_model,
            prompt_enhancer_llm_tokenizer=prompt_enhancer_llm_tokenizer,
        )

        self.video_scale_factor, self.vae_scale_factor, _ = get_vae_size_scale_factor(
            self.vae
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.allowed_inference_steps = allowed_inference_steps

    @classmethod
    def lora_state_dict(cls, pretrained_model_name_or_path_or_dict, **kwargs):
        state_dict_or_tuple = super().lora_state_dict(
            pretrained_model_name_or_path_or_dict, **kwargs
        )

        is_tuple = False
        metadata = None

        if isinstance(state_dict_or_tuple, tuple):
            state_dict, metadata = state_dict_or_tuple
            is_tuple = True
        else:
            state_dict = state_dict_or_tuple

        state_dict = {
            key.replace("diffusion_model.", "transformer."): value
            for key, value in state_dict.items()
        }

        if is_tuple:
            return state_dict, metadata

        return state_dict

    def mask_text_embeddings(self, emb, mask):
        if emb.shape[0] == 1:
            keep_index = mask.sum().item()
            return emb[:, :, :keep_index, :], keep_index
        else:
            masked_feature = emb * mask[:, None, :, None]
            return masked_feature, emb.shape[2]

    # Adapted from diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        text_encoder_max_tokens: int = 256,
        **kwargs,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                This should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings.
        """

        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # See Section 3.1. of the paper.
        max_length = (
            text_encoder_max_tokens  # TPU supports only lengths multiple of 128
        )
        if prompt_embeds is None:
            assert (
                self.text_encoder is not None
            ), "You should provide either prompt_embeds or self.text_encoder should not be None,"
            text_enc_device = next(self.text_encoder.parameters()).device
            prompt = self._text_preprocessing(prompt)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.to(text_enc_device)
            prompt_attention_mask = prompt_attention_mask.to(device)

            prompt_embeds = self.text_encoder(
                text_input_ids.to(text_enc_device), attention_mask=prompt_attention_mask
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        prompt_attention_mask = prompt_attention_mask.repeat(1, num_images_per_prompt)
        prompt_attention_mask = prompt_attention_mask.view(
            bs_embed * num_images_per_prompt, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = self._text_preprocessing(negative_prompt)
            uncond_tokens = uncond_tokens * batch_size
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = uncond_input.attention_mask
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(
                text_enc_device
            )

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(text_enc_device),
                attention_mask=negative_prompt_attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(
                1, num_images_per_prompt
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                bs_embed * num_images_per_prompt, -1
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        enhance_prompt=False,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError(
                "Must provide `prompt_attention_mask` when specifying `prompt_embeds`."
            )

        if (
            negative_prompt_embeds is not None
            and negative_prompt_attention_mask is None
        ):
            raise ValueError(
                "Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

        if enhance_prompt:
            assert (
                self.prompt_enhancer_image_caption_model is not None
            ), "Image caption model must be initialized if enhance_prompt is True"
            assert (
                self.prompt_enhancer_image_caption_processor is not None
            ), "Image caption processor must be initialized if enhance_prompt is True"
            assert (
                self.prompt_enhancer_llm_model is not None
            ), "Text prompt enhancer model must be initialized if enhance_prompt is True"
            assert (
                self.prompt_enhancer_llm_tokenizer is not None
            ), "Text prompt enhancer tokenizer must be initialized if enhance_prompt is True"

    def _text_preprocessing(self, text):
        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            text = text.strip()
            return text

        return [process(t) for t in text]

    @staticmethod
    def add_noise_to_image_conditioning_latents(
        t: float,
        init_latents: torch.Tensor,
        latents: torch.Tensor,
        noise_scale: float,
        conditioning_mask: torch.Tensor,
        generator,
        eps=1e-6,
    ):
        """
        Add timestep-dependent noise to the hard-conditioning latents.
        This helps with motion continuity, especially when conditioned on a single frame.
        """
        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        # Add noise only to hard-conditioning latents (conditioning_mask = 1.0)
        need_to_noise = (conditioning_mask > 1.0 - eps).unsqueeze(-1)
        noised_latents = init_latents + noise_scale * noise * (t**2)
        latents = torch.where(need_to_noise, noised_latents, latents)
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self,
        latents: torch.Tensor | None,
        media_items: torch.Tensor | None,
        timestep: float,
        latent_shape: torch.Size | Tuple[Any, ...],
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | List[torch.Generator],
        vae_per_channel_normalize: bool = True,
        encode_tiling: bool = False,
        encode_tile_size: tuple[int, int] = (512, 512),
        encode_tile_stride: tuple[int, int] = (256, 256),
    ):
        """
        Prepare the initial latent tensor to be denoised.
        The latents are either pure noise or a noised version of the encoded media items.
        Args:
            latents (`torch.FloatTensor` or `None`):
                The latents to use (provided by the user) or `None` to create new latents.
            media_items (`torch.FloatTensor` or `None`):
                An image or video to be updated using img2img or vid2vid. The media item is encoded and noised.
            timestep (`float`):
                The timestep to noise the encoded media_items to.
            latent_shape (`torch.Size`):
                The target latent shape.
            dtype (`torch.dtype`):
                The target dtype.
            device (`torch.device`):
                The target device.
            generator (`torch.Generator` or `List[torch.Generator]`):
                Generator(s) to be used for the noising process.
            vae_per_channel_normalize ('bool'):
                When encoding the media_items, whether to normalize the latents per-channel.
        Returns:
            `torch.FloatTensor`: The latents to be used for the denoising process. This is a tensor of shape
            (batch_size, num_channels, height, width).
        """
        if isinstance(generator, list) and len(generator) != latent_shape[0]:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {latent_shape[0]}. Make sure the batch size matches the length of the generators."
            )

        # Initialize the latents with the given latents or encoded media item, if provided
        assert (
            latents is None or media_items is None
        ), "Cannot provide both latents and media_items. Please provide only one of the two."

        assert (
            latents is None and media_items is None or timestep < 1.0
        ), "Input media_item or latents are provided, but they will be replaced with noise."

        if media_items is not None:
            latents = vae_encode(
                media_items.to(dtype=self.vae.dtype, device=self.vae.device),
                self.vae,
                vae_per_channel_normalize=vae_per_channel_normalize,
                use_tiling=encode_tiling,
                tile_size=encode_tile_size,
                tile_stride=encode_tile_stride,
            )
        if latents is not None:
            assert (
                latents.shape == latent_shape
            ), f"Latents have to be of shape {latent_shape} but are {latents.shape}."
            latents = latents.to(device=device, dtype=dtype)

        # For backward compatibility, generate in the "patchified" shape and rearrange
        b, c, f, h, w = latent_shape
        noise = randn_tensor(
            (b, f * h * w, c), generator=generator, device=device, dtype=dtype
        )
        noise = rearrange(noise, "b (f h w) c -> b c f h w", f=f, h=h, w=w)

        # scale the initial noise by the standard deviation required by the scheduler
        noise = noise * self.scheduler.init_noise_sigma

        if latents is None:
            latents = noise
        else:
            # Noise the latents to the required (first) timestep
            latents = timestep * noise + (1 - timestep) * latents

        return latents

    @staticmethod
    def classify_height_width_bin(
        height: int, width: int, ratios: dict
    ) -> Tuple[int, int]:
        """Returns binned height and width."""
        ar = float(height / width)
        closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
        default_hw = ratios[closest_ratio]
        return int(default_hw[0]), int(default_hw[1])

    @staticmethod
    def resize_and_crop_tensor(
        samples: torch.Tensor, new_width: int, new_height: int
    ) -> torch.Tensor:
        n_frames, orig_height, orig_width = samples.shape[-3:]

        # Check if resizing is needed
        if orig_height != new_height or orig_width != new_width:
            ratio = max(new_height / orig_height, new_width / orig_width)
            resized_width = int(orig_width * ratio)
            resized_height = int(orig_height * ratio)

            # Resize
            samples = LTXVideoPipeline.resize_tensor(
                samples, resized_height, resized_width
            )

            # Center Crop
            start_x = (resized_width - new_width) // 2
            end_x = start_x + new_width
            start_y = (resized_height - new_height) // 2
            end_y = start_y + new_height
            samples = samples[..., start_y:end_y, start_x:end_x]

        return samples

    @staticmethod
    def resize_tensor(media_items, height, width):
        n_frames = media_items.shape[2]
        if media_items.shape[-2:] != (height, width):
            media_items = rearrange(media_items, "b c n h w -> (b n) c h w")
            media_items = F.interpolate(
                media_items,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            media_items = rearrange(media_items, "(b n) c h w -> b c n h w", n=n_frames)
        return media_items

    @torch.no_grad()
    def __call__(
        self,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,
        timesteps: List[int] = None,
        guidance_scale: Union[float, List[float]] = 4.5,
        cfg_star_rescale: bool = False,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        skip_block_list: Optional[Union[List[List[int]], List[int]]] = None,
        stg_scale: Union[float, List[float]] = 1.0,
        rescaling_scale: Union[float, List[float]] = 0.7,
        guidance_timesteps: Optional[List[int]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        conditioning_items: Optional[List[ConditioningItem]] = None,
        decode_timestep: Union[List[float], float] = 0.0,
        decode_noise_scale: Optional[List[float]] = None,
        mixed_precision: bool = False,
        offload_to_cpu: bool = False,
        enhance_prompt: bool = False,
        text_encoder_max_tokens: int = 256,
        stochastic_sampling: bool = False,
        media_items: Optional[torch.Tensor] = None,
        # Temporal tiling parameters
        temporal_tile_size: Optional[int] = None,
        temporal_overlap: Optional[int] = None,
        temporal_overlap_strength_schedule: list[float] = [0.5],
        use_guiding_latents: bool = False,
        guiding_strength: float = 1.0,
        temporal_adain_factor: float = 0.0,
        decode_tiling: bool = False,
        decode_tile_size: Tuple[int, int] = (64, 64),
        decode_tile_stride: Tuple[int, int] = (32, 32),
        tone_map_compression_ratio: float = 0.0,
        encode_tiling: bool = False,
        encode_tile_size: Tuple[int, int] = (512, 512),
        encode_tile_stride: Tuple[int, int] = (256, 256),
        optional_negative_index_latents: Optional[torch.Tensor] = None,
        negative_index_strength: float = 1.0,
        negative_index_position: int = -1,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. If `timesteps` is provided, this parameter is ignored.
            skip_initial_inference_steps (`int`, *optional*, defaults to 0):
                The number of initial timesteps to skip. After calculating the timesteps, this number of timesteps will
                be removed from the beginning of the timesteps list. Meaning the highest-timesteps values will not run.
            skip_final_inference_steps (`int`, *optional*, defaults to 0):
                The number of final timesteps to skip. After calculating the timesteps, this number of timesteps will
                be removed from the end of the timesteps list. Meaning the lowest-timesteps values will not run.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            cfg_star_rescale (`bool`, *optional*, defaults to `False`):
                If set to `True`, applies the CFG star rescale. Scales the negative prediction according to dot
                product between positive and negative.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.FloatTensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. This negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            enhance_prompt (`bool`, *optional*, defaults to `False`):
                If set to `True`, the prompt is enhanced using a LLM model.
            text_encoder_max_tokens (`int`, *optional*, defaults to `256`):
                The maximum number of tokens to use for the text encoder.
            stochastic_sampling (`bool`, *optional*, defaults to `False`):
                If set to `True`, the sampling is stochastic. If set to `False`, the sampling is deterministic.
            media_items ('torch.Tensor', *optional*):
                The input media item used for image-to-image / video-to-video.
            conditioning_items (`List[ConditioningItem]`, *optional*):
                List of unified conditioning items that can include:
                - Regular image/video conditioning (type="image" or "video")
                - IC-LoRA guiding latents (type="guiding")
                Multiple IC-LoRA types (depth, pose, etc.) can be used simultaneously.
            temporal_tile_size (`int`, *optional*):
                Size of temporal tiles for long video generation using windowed sampling. If provided,
                the video will be generated in overlapping temporal chunks of this size.
            temporal_overlap (`int`, *optional*):
                Number of frames to overlap between temporal tiles. Required if temporal_tile_size is provided.
                Should be less than temporal_tile_size.
            temporal_overlap_strength (`float`, *optional*, defaults to `0.5`):
                Strength of conditioning from overlapping frames when extending video.
            use_guiding_latents (`bool`, *optional*, defaults to `False`):
                Whether to use conditioning_items as guiding latents for temporal tiling (e.g., IC-LoRA guides).
            guiding_strength (`float`, *optional*, defaults to `1.0`):
                Strength of guiding latent conditioning when use_guiding_latents is True.
            temporal_adain_factor (`float`, *optional*, defaults to `0.0`):
                Factor for AdaIN normalization to maintain temporal consistency across tiles (0.0 = disabled).
        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        is_video = kwargs.get("is_video", False)
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # Check for autoregressive pipeline
        if temporal_tile_size is not None:
            if temporal_overlap is None:
                raise ValueError(
                    "temporal_overlap must be provided when using temporal_tile_size"
                )
            if temporal_overlap >= temporal_tile_size:
                raise ValueError(
                    "temporal_overlap must be less than temporal_tile_size"
                )

            return self._autoregressive_pipeline(
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                skip_initial_inference_steps=skip_initial_inference_steps,
                skip_final_inference_steps=skip_final_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                cfg_star_rescale=cfg_star_rescale,
                skip_layer_strategy=skip_layer_strategy,
                skip_block_list=skip_block_list,
                stg_scale=stg_scale,
                rescaling_scale=rescaling_scale,
                guidance_timesteps=guidance_timesteps,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                output_type=output_type,
                return_dict=return_dict,
                callback_on_step_end=callback_on_step_end,
                conditioning_items=conditioning_items,
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                mixed_precision=mixed_precision,
                offload_to_cpu=offload_to_cpu,
                enhance_prompt=enhance_prompt,
                text_encoder_max_tokens=text_encoder_max_tokens,
                stochastic_sampling=stochastic_sampling,
                media_items=media_items,
                temporal_tile_size=temporal_tile_size,
                temporal_overlap=temporal_overlap,
                temporal_overlap_strength_schedule=temporal_overlap_strength_schedule,
                use_guiding_latents=use_guiding_latents,
                guiding_strength=guiding_strength,
                temporal_adain_factor=temporal_adain_factor,
                encode_tiling=encode_tiling,
                encode_tile_size=encode_tile_size,
                encode_tile_stride=encode_tile_stride,
                decode_tiling=decode_tiling,
                decode_tile_size=decode_tile_size,
                decode_tiling_stride=decode_tile_stride,
                optional_negative_index_latents=optional_negative_index_latents,
                negative_index_strength=negative_index_strength,
                negative_index_position=negative_index_position,
                tone_map_compression_ratio=tone_map_compression_ratio,
                **kwargs,
            )

        self.video_scale_factor = self.video_scale_factor if is_video else 1
        vae_per_channel_normalize = kwargs.get("vae_per_channel_normalize", True)
        image_cond_noise_scale = kwargs.get("image_cond_noise_scale", 0.0)

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_num_frames = num_frames // self.video_scale_factor

        if isinstance(self.vae, CausalVideoAutoencoder) and is_video:
            latent_num_frames += 1

        latent_shape = (
            batch_size * num_images_per_prompt,
            self.transformer.config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
        )

        # Prepare the list of denoising time-steps
        retrieve_timesteps_kwargs = {}
        if isinstance(self.scheduler, TimestepShifter):
            retrieve_timesteps_kwargs["samples_shape"] = latent_shape

        assert (
            skip_initial_inference_steps == 0
            or latents is not None
            or media_items is not None
        ), (
            f"skip_initial_inference_steps ({skip_initial_inference_steps}) is used for image-to-image/video-to-video - "
            "media_item or latents should be provided."
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            skip_initial_inference_steps=skip_initial_inference_steps,
            skip_final_inference_steps=skip_final_inference_steps,
            **retrieve_timesteps_kwargs,
        )

        if self.allowed_inference_steps is not None:
            for timestep in [round(x, 4) for x in timesteps.tolist()]:
                assert (
                    timestep in self.allowed_inference_steps
                ), f"Invalid inference timestep {timestep}. Allowed timesteps are {self.allowed_inference_steps}."

        if guidance_timesteps:
            guidance_mapping = []
            for timestep in timesteps:
                indices = [
                    i for i, val in enumerate(guidance_timesteps) if val <= timestep
                ]
                # assert len(indices) > 0, f"No guidance timestep found for {timestep}"
                guidance_mapping.append(
                    indices[0] if len(indices) > 0 else (len(guidance_timesteps) - 1)
                )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        if not isinstance(guidance_scale, List):
            guidance_scale = [guidance_scale] * len(timesteps)
        else:
            guidance_scale = [
                guidance_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        if not isinstance(stg_scale, List):
            stg_scale = [stg_scale] * len(timesteps)
        else:
            stg_scale = [stg_scale[guidance_mapping[i]] for i in range(len(timesteps))]

        if not isinstance(rescaling_scale, List):
            rescaling_scale = [rescaling_scale] * len(timesteps)
        else:
            rescaling_scale = [
                rescaling_scale[guidance_mapping[i]] for i in range(len(timesteps))
            ]

        # Normalize skip_block_list to always be None or a list of lists matching timesteps
        if skip_block_list is not None:
            # Convert single list to list of lists if needed
            if len(skip_block_list) == 0 or not isinstance(skip_block_list[0], list):
                skip_block_list = [skip_block_list] * len(timesteps)
            else:
                new_skip_block_list = []
                for i, timestep in enumerate(timesteps):
                    new_skip_block_list.append(skip_block_list[guidance_mapping[i]])
                skip_block_list = new_skip_block_list

        if enhance_prompt:
            self.prompt_enhancer_image_caption_model = (
                self.prompt_enhancer_image_caption_model.to(self._execution_device)
            )
            self.prompt_enhancer_llm_model = self.prompt_enhancer_llm_model.to(
                self._execution_device
            )

            prompt = generate_cinematic_prompt(
                self.prompt_enhancer_image_caption_model,
                self.prompt_enhancer_image_caption_processor,
                self.prompt_enhancer_llm_model,
                self.prompt_enhancer_llm_tokenizer,
                prompt,
                conditioning_items,
                max_new_tokens=text_encoder_max_tokens,
            )

        # 3. Encode input prompt
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self._execution_device)

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            True,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            text_encoder_max_tokens=text_encoder_max_tokens,
        )

        if offload_to_cpu and self.text_encoder is not None:
            self.text_encoder = self.text_encoder.cpu()

        self.transformer = self.transformer.to(self._execution_device)

        prompt_embeds_batch = prompt_embeds
        prompt_attention_mask_batch = prompt_attention_mask
        negative_prompt_embeds = (
            torch.zeros_like(prompt_embeds)
            if negative_prompt_embeds is None
            else negative_prompt_embeds
        )
        negative_prompt_attention_mask = (
            torch.zeros_like(prompt_attention_mask)
            if negative_prompt_attention_mask is None
            else negative_prompt_attention_mask
        )

        prompt_embeds_batch = torch.cat(
            [negative_prompt_embeds, prompt_embeds, prompt_embeds], dim=0
        )
        prompt_attention_mask_batch = torch.cat(
            [
                negative_prompt_attention_mask,
                prompt_attention_mask,
                prompt_attention_mask,
            ],
            dim=0,
        )
        # 4. Prepare the initial latents using the provided media and conditioning items

        # Prepare the initial latents tensor, shape = (b, c, f, h, w)
        latents = self.prepare_latents(
            latents=latents,
            media_items=media_items,
            timestep=timesteps[0],
            latent_shape=latent_shape,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            vae_per_channel_normalize=vae_per_channel_normalize,
            encode_tiling=encode_tiling,
            encode_tile_size=encode_tile_size,
            encode_tile_stride=encode_tile_stride,
        )

        # Handle all conditioning through the unified interface
        latents, pixel_coords, conditioning_mask, num_cond_latents = (
            self.prepare_combined_conditioning(
                conditioning_items=conditioning_items,
                init_latents=latents,
                num_frames=num_frames,
                height=height,
                width=width,
                vae_per_channel_normalize=vae_per_channel_normalize,
                encode_tiling=encode_tiling,
                encode_tile_size=encode_tile_size,
                encode_tile_stride=encode_tile_stride,
                generator=generator,
            )
        )

        # Used for image_cond_noise_update
        init_latents = latents.clone()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        orig_conditioning_mask = conditioning_mask

        # Befor compiling this code please be aware:
        # This code might generate different input shapes if some timesteps have no STG or CFG.
        # This means that the codes might need to be compiled mutliple times.
        # To avoid that, use the same STG and CFG values for all timesteps.

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                do_classifier_free_guidance = guidance_scale[i] > 1.0
                do_spatio_temporal_guidance = stg_scale[i] > 0
                do_rescaling = rescaling_scale[i] != 1.0

                num_conds = 1
                if do_classifier_free_guidance:
                    num_conds += 1
                if do_spatio_temporal_guidance:
                    num_conds += 1

                if do_classifier_free_guidance and do_spatio_temporal_guidance:
                    indices = slice(batch_size * 0, batch_size * 3)
                elif do_classifier_free_guidance:
                    indices = slice(batch_size * 0, batch_size * 2)
                elif do_spatio_temporal_guidance:
                    indices = slice(batch_size * 1, batch_size * 3)
                else:
                    indices = slice(batch_size * 1, batch_size * 2)

                # Prepare skip layer masks
                skip_layer_mask: Optional[torch.Tensor] = None
                if do_spatio_temporal_guidance:
                    if skip_block_list is not None:
                        skip_layer_mask = self.transformer.create_skip_layer_mask(
                            batch_size, num_conds, num_conds - 1, skip_block_list[i]
                        )

                batch_pixel_coords = torch.cat([pixel_coords] * num_conds)
                conditioning_mask = orig_conditioning_mask
                if conditioning_mask is not None and is_video:
                    assert num_images_per_prompt == 1
                    conditioning_mask = torch.cat([conditioning_mask] * num_conds)
                fractional_coords = batch_pixel_coords.to(torch.float32)
                fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)

                if conditioning_mask is not None and image_cond_noise_scale > 0.0:
                    latents = self.add_noise_to_image_conditioning_latents(
                        t,
                        init_latents,
                        latents,
                        image_cond_noise_scale,
                        orig_conditioning_mask,
                        generator,
                    )

                latent_model_input = (
                    torch.cat([latents] * num_conds) if num_conds > 1 else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor(
                        [current_timestep],
                        dtype=dtype,
                        device=latent_model_input.device,
                    )
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(
                        latent_model_input.device
                    )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(
                    latent_model_input.shape[0]
                ).unsqueeze(-1)

                if conditioning_mask is not None:
                    # Conditioning latents have an initial timestep and noising level of (1.0 - conditioning_mask)
                    # and will start to be denoised when the current timestep is lower than their conditioning timestep.
                    current_timestep = torch.min(
                        current_timestep, 1.0 - conditioning_mask
                    )

                # Choose the appropriate context manager based on `mixed_precision`
                if mixed_precision:
                    context_manager = torch.autocast(device.type, dtype=torch.bfloat16)
                else:
                    context_manager = nullcontext()  # Dummy context manager

                # predict noise model_output
                with context_manager:
                    noise_pred = self.transformer(
                        latent_model_input.to(self.transformer.dtype),
                        indices_grid=fractional_coords,
                        encoder_hidden_states=prompt_embeds_batch[indices].to(
                            self.transformer.dtype
                        ),
                        encoder_attention_mask=prompt_attention_mask_batch[indices],
                        timestep=current_timestep,
                        skip_layer_mask=skip_layer_mask,
                        skip_layer_strategy=skip_layer_strategy,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_spatio_temporal_guidance:
                    noise_pred_text, noise_pred_text_perturb = noise_pred.chunk(
                        num_conds
                    )[-2:]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_conds)[:2]

                    if cfg_star_rescale:
                        # Rescales the unconditional noise prediction using the projection of the conditional prediction onto it:
                        # α = (⟨ε_text, ε_uncond⟩ / ||ε_uncond||²), then ε_uncond ← α * ε_uncond
                        # where ε_text is the conditional noise prediction and ε_uncond is the unconditional one.
                        positive_flat = noise_pred_text.view(batch_size, -1)
                        negative_flat = noise_pred_uncond.view(batch_size, -1)
                        dot_product = torch.sum(
                            positive_flat * negative_flat, dim=1, keepdim=True
                        )
                        squared_norm = (
                            torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
                        )
                        alpha = dot_product / squared_norm
                        noise_pred_uncond = alpha * noise_pred_uncond

                    noise_pred = noise_pred_uncond + guidance_scale[i] * (
                        noise_pred_text - noise_pred_uncond
                    )
                elif do_spatio_temporal_guidance:
                    noise_pred = noise_pred_text
                if do_spatio_temporal_guidance:
                    noise_pred = noise_pred + stg_scale[i] * (
                        noise_pred_text - noise_pred_text_perturb
                    )
                    if do_rescaling and stg_scale[i] > 0.0:
                        noise_pred_text_std = noise_pred_text.view(batch_size, -1).std(
                            dim=1, keepdim=True
                        )
                        noise_pred_std = noise_pred.view(batch_size, -1).std(
                            dim=1, keepdim=True
                        )

                        factor = noise_pred_text_std / noise_pred_std
                        factor = rescaling_scale[i] * factor + (1 - rescaling_scale[i])

                        noise_pred = noise_pred * factor.view(batch_size, 1, 1)

                current_timestep = current_timestep[:1]
                # learned sigma
                if (
                    self.transformer.config.out_channels // 2
                    == self.transformer.config.in_channels
                ):
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # compute previous image: x_t -> x_t-1
                latents = self.denoising_step(
                    latents,
                    noise_pred,
                    current_timestep,
                    orig_conditioning_mask,
                    t,
                    extra_step_kwargs,
                    stochastic_sampling=stochastic_sampling,
                )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if callback_on_step_end is not None:
                    callback_on_step_end(self, i, t, {})

        if offload_to_cpu:
            self.transformer = self.transformer.cpu()
            if self._execution_device == "cuda":
                torch.cuda.empty_cache()

        # Remove the added conditioning latents
        latents = latents[:, num_cond_latents:]

        latents = self.patchifier.unpatchify(
            latents=latents,
            output_height=latent_height,
            output_width=latent_width,
            out_channels=self.transformer.in_channels
            // math.prod(self.patchifier.patch_size),
        )
        if output_type != "latent":
            latents = self.tone_map_latents(latents, tone_map_compression_ratio)
            image = self._vae_decode(
                latents,
                vae_per_channel_normalize=kwargs["vae_per_channel_normalize"],
                decode_timestep=decode_timestep,
                decode_tiling=decode_tiling,
                decode_tile_size=decode_tile_size,
                decode_tile_stride=decode_tile_stride,
            )

            image = self.image_processor.postprocess(image, output_type=output_type)

        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def denoising_step(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        current_timestep: torch.Tensor,
        conditioning_mask: torch.Tensor,
        t: float,
        extra_step_kwargs,
        t_eps=1e-6,
        stochastic_sampling=False,
    ):
        """
        Perform the denoising step for the required tokens, based on the current timestep and
        conditioning mask:
        Conditioning latents have an initial timestep and noising level of (1.0 - conditioning_mask)
        and will start to be denoised when the current timestep is equal or lower than their
        conditioning timestep.
        (hard-conditioning latents with conditioning_mask = 1.0 are never denoised)
        """
        # Denoise the latents using the scheduler
        denoised_latents = self.scheduler.step(
            noise_pred,
            t if current_timestep is None else current_timestep,
            latents,
            **extra_step_kwargs,
            return_dict=False,
            stochastic_sampling=stochastic_sampling,
        )[0]

        if conditioning_mask is None:
            return denoised_latents

        tokens_to_denoise_mask = (t - t_eps < (1.0 - conditioning_mask)).unsqueeze(-1)
        return torch.where(tokens_to_denoise_mask, denoised_latents, latents)

    def prepare_combined_conditioning(
        self,
        conditioning_items: Optional[List[ConditioningItem]],
        init_latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        vae_per_channel_normalize: bool = False,
        encode_tiling: bool = False,
        encode_tile_size: tuple = (512, 512),
        encode_tile_stride: tuple = (256, 256),
        generator=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Prepare all types of conditioning through the unified ConditioningItem interface.

        This method processes all conditioning items (regular image/video conditioning and
        IC-LoRA guiding latents) through a single unified interface.

        Args:
            conditioning_items (Optional[List[ConditioningItem]]): List of conditioning items that can include:
                - Regular image/video conditioning (type="image" or "video")
                - IC-LoRA guiding latents (type="guiding")
                Multiple IC-LoRA types (depth, pose, etc.) can be used simultaneously
            init_latents (torch.Tensor): Initial latent tensor
            num_frames (int): Number of frames to generate
            height (int): Height of the generated video
            width (int): Width of the generated video
            vae_per_channel_normalize (bool): Whether to normalize channels during VAE encoding
            generator: Random generator for noise blending

        Returns:
            Tuple containing updated latents, pixel coordinates, conditioning mask, and num extra latents
        """
        # Start with empty lists for all conditioning types
        extra_conditioning_latents = []
        extra_conditioning_pixel_coords = []
        extra_conditioning_mask = []
        extra_conditioning_num_latents = 0

        # Process all conditioning items through the unified interface
        if conditioning_items:
            assert isinstance(self.vae, CausalVideoAutoencoder)

            batch_size, _, num_latent_frames = init_latents.shape[:3]

            init_conditioning_mask = torch.zeros(
                init_latents[:, 0, :, :, :].shape,
                dtype=torch.float32,
                device=init_latents.device,
            )

            # Process each conditioning item
            for conditioning_item in conditioning_items:
                media_item = conditioning_item.media_item
                media_frame_number = conditioning_item.media_frame_number
                strength = conditioning_item.conditioning_strength
                conditioning_type = conditioning_item.conditioning_type

                assert media_item.ndim == 5  # (b, c, f, h, w)
                b, c, n_frames, h, w = media_item.shape

                # Check if media item is already in latent space (more than 3 channels indicates encoded)
                is_already_encoded = conditioning_item.is_latent_space
                logger.info(
                    f"Processing conditioning type={conditioning_type}, shape={media_item.shape}, channels={c}, is_encoded={is_already_encoded}"
                )

                if is_already_encoded:
                    # Media item is already in latent space, use directly
                    media_item_latents = media_item.to(dtype=init_latents.dtype)
                    logger.info(
                        f"Using pre-encoded guiding latents: {media_item.shape}, frame_number={media_frame_number}"
                    )
                else:
                    # Resize conditioning item if needed
                    conditioning_item = self._resize_conditioning_item(
                        conditioning_item, height, width
                    )
                    media_item = conditioning_item.media_item

                    # Validate dimensions for pixel-space items
                    assert (
                        height == media_item.shape[-2] and width == media_item.shape[-1]
                    ) or media_frame_number == 0, f"Dimensions do not match: {height}x{width} != {media_item.shape[-2]}x{media_item.shape[-1]} - allowed only when media_frame_number == 0"
                    assert n_frames % 8 == 1
                    assert (
                        media_frame_number >= 0
                        and media_frame_number + n_frames <= num_frames
                    )

                    # Encode pixel-space conditioning media items
                    media_item_latents = vae_encode(
                        media_item.to(dtype=self.vae.dtype, device=self.vae.device),
                        self.vae,
                        vae_per_channel_normalize=vae_per_channel_normalize,
                        use_tiling=encode_tiling,
                        tile_size=encode_tile_size,
                        tile_stride=encode_tile_stride,
                    ).to(dtype=init_latents.dtype)

                # Handle the different conditioning cases
                if conditioning_type in ["guiding", "guiding_latents"]:
                    # For guiding latents, we skip the spatial positioning logic
                    # and go directly to creating extra conditioning tokens

                    # Adjust frame number for latent space
                    latent_start_frame = media_frame_number // self.video_scale_factor

                    # Blend with noise if needed
                    if strength < 1.0:
                        noise = randn_tensor(
                            media_item_latents.shape,
                            generator=generator,
                            device=media_item_latents.device,
                            dtype=media_item_latents.dtype,
                        )
                        media_item_latents = torch.lerp(
                            noise, media_item_latents, strength
                        )

                    # Patchify and calculate pixel coordinates
                    media_item_latents, latent_coords = self.patchifier.patchify(
                        latents=media_item_latents
                    )
                    pixel_coords = latent_to_pixel_coords(
                        latent_coords,
                        self.vae,
                        causal_fix=self.transformer.config.causal_temporal_positioning,
                    )

                    # Update frame numbers to match the target frame number
                    pixel_coords[:, 0] += latent_start_frame
                    extra_conditioning_num_latents += media_item_latents.shape[1]

                    # Create conditioning mask
                    conditioning_mask = torch.full(
                        media_item_latents.shape[:2],
                        strength,
                        dtype=torch.float32,
                        device=init_latents.device,
                    )

                    extra_conditioning_latents.append(media_item_latents)
                    extra_conditioning_pixel_coords.append(pixel_coords)
                    extra_conditioning_mask.append(conditioning_mask)

                elif conditioning_type == "negative_index":
                    # Handle negative index latents - these are placed at negative temporal positions
                    # to provide long-term visual memory without being treated as part of the sequence

                    # Blend with noise based on strength
                    if strength < 1.0:
                        noise = randn_tensor(
                            media_item_latents.shape,
                            generator=generator,
                            device=media_item_latents.device,
                            dtype=media_item_latents.dtype,
                        )
                        media_item_latents = torch.lerp(
                            noise, media_item_latents, strength
                        )

                    # Patchify and calculate pixel coordinates
                    media_item_latents, latent_coords = self.patchifier.patchify(
                        latents=media_item_latents
                    )
                    pixel_coords = latent_to_pixel_coords(
                        latent_coords,
                        self.vae,
                        causal_fix=self.transformer.config.causal_temporal_positioning,
                    )

                    # Set negative frame indices for proper positional encoding
                    # This is what makes these latents receive negative positional embeddings
                    pixel_coords[:, 0] = (
                        media_frame_number  # Use the negative frame number directly
                    )
                    extra_conditioning_num_latents += media_item_latents.shape[1]

                    # Create conditioning mask
                    conditioning_mask = torch.full(
                        media_item_latents.shape[:2],
                        strength,
                        dtype=torch.float32,
                        device=init_latents.device,
                    )

                    extra_conditioning_latents.append(media_item_latents)
                    extra_conditioning_pixel_coords.append(pixel_coords)
                    extra_conditioning_mask.append(conditioning_mask)

                    logger.info(
                        f"Added negative index latents: shape={media_item_latents.shape}, "
                        f"negative_position={media_frame_number}, strength={strength}"
                    )

                elif media_frame_number == 0:
                    # Get the target spatial position of the latent conditioning item
                    media_item_latents, l_x, l_y = self._get_latent_spatial_position(
                        media_item_latents,
                        conditioning_item,
                        height,
                        width,
                        strip_latent_border=True,
                    )
                    b, c_l, f_l, h_l, w_l = media_item_latents.shape

                    # First frame or sequence - just update the initial noise latents and the mask
                    init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l] = (
                        torch.lerp(
                            init_latents[:, :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l],
                            media_item_latents,
                            strength,
                        )
                    )
                    init_conditioning_mask[
                        :, :f_l, l_y : l_y + h_l, l_x : l_x + w_l
                    ] = strength

                else:
                    # Non-first frame or sequence
                    if n_frames > 1:
                        # Handle non-first sequence.
                        (
                            init_latents,
                            init_conditioning_mask,
                            media_item_latents,
                        ) = self._handle_non_first_conditioning_sequence(
                            init_latents,
                            init_conditioning_mask,
                            media_item_latents,
                            media_frame_number,
                            strength,
                        )

                    # Single frame or sequence-prefix latents
                    if media_item_latents is not None:
                        noise = randn_tensor(
                            media_item_latents.shape,
                            generator=generator,
                            device=media_item_latents.device,
                            dtype=media_item_latents.dtype,
                        )

                        media_item_latents = torch.lerp(
                            noise, media_item_latents, strength
                        )

                        # Patchify the extra conditioning latents and calculate their pixel coordinates
                        media_item_latents, latent_coords = self.patchifier.patchify(
                            latents=media_item_latents
                        )
                        pixel_coords = latent_to_pixel_coords(
                            latent_coords,
                            self.vae,
                            causal_fix=self.transformer.config.causal_temporal_positioning,
                        )

                        # Update the frame numbers to match the target frame number
                        pixel_coords[:, 0] += media_frame_number
                        extra_conditioning_num_latents += media_item_latents.shape[1]

                        conditioning_mask = torch.full(
                            media_item_latents.shape[:2],
                            strength,
                            dtype=torch.float32,
                            device=init_latents.device,
                        )

                        extra_conditioning_latents.append(media_item_latents)
                        extra_conditioning_pixel_coords.append(pixel_coords)
                        extra_conditioning_mask.append(conditioning_mask)

        # 3. Patchify the initial latents
        init_latents, init_latent_coords = self.patchifier.patchify(
            latents=init_latents
        )
        init_pixel_coords = latent_to_pixel_coords(
            init_latent_coords,
            self.vae,
            causal_fix=self.transformer.config.causal_temporal_positioning,
        )

        # 4. Create conditioning mask for initial latents
        if conditioning_items:
            init_conditioning_mask, _ = self.patchifier.patchify(
                latents=init_conditioning_mask.unsqueeze(1)
            )
            init_conditioning_mask = init_conditioning_mask.squeeze(-1)
        else:
            # No regular conditioning, create zero mask for init latents
            init_conditioning_mask = torch.zeros(
                init_latents.shape[:2],
                dtype=torch.float32,
                device=init_latents.device,
            )

        # 5. Combine all conditioning types
        if extra_conditioning_latents:
            # Stack the extra conditioning latents, pixel coordinates and mask
            init_latents = torch.cat([*extra_conditioning_latents, init_latents], dim=1)
            init_pixel_coords = torch.cat(
                [*extra_conditioning_pixel_coords, init_pixel_coords], dim=2
            )
            init_conditioning_mask = torch.cat(
                [*extra_conditioning_mask, init_conditioning_mask], dim=1
            )

            if self.transformer.use_tpu_flash_attention:
                # When flash attention is used, keep the original number of tokens by removing tokens from the end.
                init_latents = init_latents[:, :-extra_conditioning_num_latents]
                init_pixel_coords = init_pixel_coords[
                    :, :, :-extra_conditioning_num_latents
                ]
                init_conditioning_mask = init_conditioning_mask[
                    :, :-extra_conditioning_num_latents
                ]

        # Return None for conditioning_mask if no conditioning was applied
        final_conditioning_mask = init_conditioning_mask if conditioning_items else None

        return (
            init_latents,
            init_pixel_coords,
            final_conditioning_mask,
            extra_conditioning_num_latents,
        )

    @staticmethod
    def _resize_conditioning_item(
        conditioning_item: ConditioningItem,
        height: int,
        width: int,
    ):
        if conditioning_item.media_x or conditioning_item.media_y:
            raise ValueError(
                "Provide media_item in the target size for spatial conditioning."
            )
        new_conditioning_item = copy.copy(conditioning_item)
        new_conditioning_item.media_item = LTXVideoPipeline.resize_tensor(
            conditioning_item.media_item, height, width
        )
        return new_conditioning_item

    def _get_latent_spatial_position(
        self,
        latents: torch.Tensor,
        conditioning_item: ConditioningItem,
        height: int,
        width: int,
        strip_latent_border,
    ):
        """
        Get the spatial position of the conditioning item in the latent space.
        If requested, strip the conditioning latent borders that do not align with target borders.
        (border latents look different then other latents and might confuse the model)
        """
        scale = self.vae_scale_factor
        h, w = conditioning_item.media_item.shape[-2:]

        if conditioning_item.is_latent_space:
            h = h * scale
            w = w * scale

        assert (
            h <= height and w <= width
        ), f"Conditioning item size {h}x{w} is larger than target size {height}x{width}"
        assert (
            h % scale == 0 and w % scale == 0
        ), f"Conditioning item size {h}x{w} must have both axes divisible by {scale}"

        # Compute the start and end spatial positions of the media item
        x_start, y_start = conditioning_item.media_x, conditioning_item.media_y
        x_start = (width - w) // 2 if x_start is None else x_start
        y_start = (height - h) // 2 if y_start is None else y_start
        x_end, y_end = x_start + w, y_start + h

        assert (
            x_end <= width and y_end <= height
        ), f"Conditioning item {x_start}:{x_end}x{y_start}:{y_end} is out of bounds for target size {width}x{height}"

        if strip_latent_border:
            # Strip one latent from left/right and/or top/bottom, update x, y accordingly
            if x_start > 0:
                x_start += scale
                latents = latents[:, :, :, :, 1:]

            if y_start > 0:
                y_start += scale
                latents = latents[:, :, :, 1:, :]

            if x_end < width:
                latents = latents[:, :, :, :, :-1]

            if y_end < height:
                latents = latents[:, :, :, :-1, :]

        return latents, x_start // scale, y_start // scale

    @staticmethod
    def _handle_non_first_conditioning_sequence(
        init_latents: torch.Tensor,
        init_conditioning_mask: torch.Tensor,
        latents: torch.Tensor,
        media_frame_number: int,
        strength: float,
        num_prefix_latent_frames: int = 2,
        prefix_latents_mode: str = "concat",
        prefix_soft_conditioning_strength: float = 0.15,
    ):
        """
        Special handling for a conditioning sequence that does not start on the first frame.
        The special handling is required to allow a short encoded video to be used as middle
        (or last) sequence in a longer video.
        Args:
            init_latents (torch.Tensor): The initial noise latents to be updated.
            init_conditioning_mask (torch.Tensor): The initial conditioning mask to be updated.
            latents (torch.Tensor): The encoded conditioning item.
            media_frame_number (int): The target frame number of the first frame in the conditioning sequence.
            strength (float): The conditioning strength for the conditioning latents.
            num_prefix_latent_frames (int, optional): The length of the sequence prefix, to be handled
                separately. Defaults to 2.
            prefix_latents_mode (str, optional): Special treatment for prefix (boundary) latents.
                - "drop": Drop the prefix latents.
                - "soft": Use the prefix latents, but with soft-conditioning
                - "concat": Add the prefix latents as extra tokens (like single frames)
            prefix_soft_conditioning_strength (float, optional): The strength of the soft-conditioning for
                the prefix latents, relevant if `prefix_latents_mode` is "soft". Defaults to 0.1.

        """
        f_l = latents.shape[2]
        f_l_p = num_prefix_latent_frames
        assert f_l >= f_l_p
        assert media_frame_number % 8 == 0
        if f_l > f_l_p:
            # Insert the conditioning latents **excluding the prefix** into the sequence
            f_l_start = media_frame_number // 8 + f_l_p
            f_l_end = f_l_start + f_l - f_l_p
            init_latents[:, :, f_l_start:f_l_end] = torch.lerp(
                init_latents[:, :, f_l_start:f_l_end],
                latents[:, :, f_l_p:],
                strength,
            )
            # Mark these latent frames as conditioning latents
            init_conditioning_mask[:, f_l_start:f_l_end] = strength

        # Handle the prefix-latents
        if prefix_latents_mode == "soft":
            if f_l_p > 1:
                # Drop the first (single-frame) latent and soft-condition the remaining prefix
                f_l_start = media_frame_number // 8 + 1
                f_l_end = f_l_start + f_l_p - 1
                strength = min(prefix_soft_conditioning_strength, strength)
                init_latents[:, :, f_l_start:f_l_end] = torch.lerp(
                    init_latents[:, :, f_l_start:f_l_end],
                    latents[:, :, 1:f_l_p],
                    strength,
                )
                # Mark these latent frames as conditioning latents
                init_conditioning_mask[:, f_l_start:f_l_end] = strength
            latents = None  # No more latents to handle
        elif prefix_latents_mode == "drop":
            # Drop the prefix latents
            latents = None
        elif prefix_latents_mode == "concat":
            # Pass-on the prefix latents to be handled as extra conditioning frames
            latents = latents[:, :, :f_l_p]
        else:
            raise ValueError(f"Invalid prefix_latents_mode: {prefix_latents_mode}")
        return (
            init_latents,
            init_conditioning_mask,
            latents,
        )

    def trim_conditioning_sequence(
        self, start_frame: int, sequence_num_frames: int, target_num_frames: int
    ):
        """
        Trim a conditioning sequence to the allowed number of frames.

        Args:
            start_frame (int): The target frame number of the first frame in the sequence.
            sequence_num_frames (int): The number of frames in the sequence.
            target_num_frames (int): The target number of frames in the generated video.

        Returns:
            int: updated sequence length
        """
        scale_factor = self.video_scale_factor
        num_frames = min(sequence_num_frames, target_num_frames - start_frame)
        # Trim down to a multiple of temporal_scale_factor frames plus 1
        num_frames = (num_frames - 1) // scale_factor * scale_factor + 1
        return num_frames

    def _combine_temporal_chunks(
        self,
        output_chunks: List,
        chunk_starts: List[int],
        chunk_ends: List[int],
        temporal_overlap: int,
        output_type: str = "pil",
    ):
        """
        Combine temporal chunks with proper overlap blending.

        Args:
            output_chunks: List of pipeline outputs for each chunk
            chunk_starts: Start frame for each chunk
            chunk_ends: End frame for each chunk
            temporal_overlap: Number of frames to overlap
            output_type: Output type (pil, pt, latent)

        Returns:
            Combined output
        """
        if len(output_chunks) == 1:
            # Single chunk, return as-is
            chunk = output_chunks[0]
            if hasattr(chunk, "images"):
                return chunk.images
            elif isinstance(chunk, tuple):
                return chunk[0]
            else:
                return chunk

        # Extract latents from each chunk
        chunk_latents = []
        for chunk in output_chunks:
            if hasattr(chunk, "images"):
                chunk_latents.append(chunk.images)
            elif isinstance(chunk, tuple):
                chunk_latents.append(chunk[0])
            else:
                chunk_latents.append(chunk)

        # Combine chunks with overlap blending in latent space
        combined_latents = chunk_latents[0]  # Start with first chunk

        for i in range(1, len(chunk_latents)):
            current_chunk = chunk_latents[i]

            # Calculate overlap in latent space
            latent_overlap_frames = temporal_overlap // self.video_scale_factor
            if isinstance(self.vae, CausalVideoAutoencoder):
                latent_overlap_frames = max(1, latent_overlap_frames)

            # Blend the chunks
            combined_latents = linear_overlap_blend(
                combined_latents, current_chunk, latent_overlap_frames, temporal_axis=2
            )

        return combined_latents

    def _autoregressive_pipeline(
        self,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        temporal_tile_size: int,
        temporal_overlap: int,
        temporal_overlap_strength_schedule: list[float] = 0.5,
        use_guiding_latents: bool = False,
        guiding_strength: float = 1.0,
        temporal_adain_factor: float = 0.0,
        optional_negative_index_latents: Optional[torch.Tensor] = None,
        negative_index_strength: float = 1.0,
        negative_index_position: int = -1,
        **kwargs,
    ):
        """
        Generate video using temporal tiling and autoregressive sampling.
        This properly implements the LTXVLoopingSampler approach by:
        1. Running complete denoising processes on temporal chunks
        2. Using previous chunk results as conditioning for subsequent chunks
        3. Each chunk goes through the full denoising pipeline (all timesteps)
        """
        is_video = kwargs.get("is_video", True)
        device = self._execution_device

        # Calculate temporal chunks (in pixel frames, not latent frames)
        if num_frames < temporal_tile_size:
            # No need for tiling if the video is smaller than tile size
            kwargs.pop("temporal_tile_size", None)
            kwargs.pop("temporal_overlap", None)
            return self.__call__(
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                **kwargs,
            )

        # Calculate chunk boundaries in pixel space
        chunk_starts = list(range(0, num_frames, temporal_tile_size - temporal_overlap))
        chunk_ends = [
            min(start + temporal_tile_size, num_frames) for start in chunk_starts
        ]

        # Remove last chunk if it's too small or adjust it
        if (
            len(chunk_starts) > 1
            and chunk_ends[-1] - chunk_starts[-1] < temporal_overlap
        ):
            chunk_starts = chunk_starts[:-1]
            chunk_ends = chunk_ends[:-1]
            chunk_ends[-1] = num_frames

        logger.info(f"Temporal tiling: {len(chunk_starts)} chunks")
        for i, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
            logger.info(f"  Chunk {i+1}: frames {start}-{end} ({end-start} frames)")

        # Store original generator state for reproducibility
        original_generator_state = None
        if kwargs.get("generator") is not None:
            original_generator_state = kwargs["generator"].get_state()

        output_chunks = []
        first_chunk_result = None  # For AdaIN reference

        num_chunks = len(chunk_starts)
        for chunk_idx, (start_frame, end_frame) in tqdm(
            enumerate(zip(chunk_starts, chunk_ends)),
            desc="Generating",
            unit="chunk",
            total=num_chunks,
        ):
            chunk_num_frames = end_frame - start_frame

            logger.info(
                f"Processing chunk {chunk_idx + 1}/{len(chunk_starts)}: frames {start_frame}-{end_frame}"
            )

            if chunk_idx > len(temporal_overlap_strength_schedule) - 1:
                temporal_overlap_strength = temporal_overlap_strength_schedule[-1]
            else:
                temporal_overlap_strength = temporal_overlap_strength_schedule[
                    chunk_idx
                ]

            # Prepare chunk-specific parameters
            chunk_kwargs = kwargs.copy()
            # Remove temporal tiling parameters to prevent recursion
            chunk_kwargs.pop("temporal_tile_size", None)
            chunk_kwargs.pop("temporal_overlap", None)
            chunk_kwargs.pop("temporal_overlap_strength", None)
            chunk_kwargs.pop("use_guiding_latents", None)
            chunk_kwargs.pop("guiding_strength", None)
            chunk_kwargs.pop("temporal_adain_factor", None)

            # Set deterministic seed for each chunk
            if kwargs.get("generator") is not None:
                chunk_kwargs["generator"].manual_seed(
                    kwargs["generator"].initial_seed() + start_frame
                )

            # Prepare conditioning for this chunk
            chunk_conditioning_items = []

            # Add overlap conditioning from previous chunk (if not first chunk)
            if chunk_idx > 0 and output_chunks:
                # Extract overlap region from previous chunk
                prev_chunk = output_chunks[-1]
                overlap_frames = temporal_overlap

                # Get the last 'overlap_frames' from previous chunk as conditioning
                if hasattr(prev_chunk, "images"):
                    prev_latents = prev_chunk.images
                else:
                    prev_latents = (
                        prev_chunk[0] if isinstance(prev_chunk, tuple) else prev_chunk
                    )

                # Get overlap region in latent space
                latent_overlap_frames = overlap_frames // self.video_scale_factor
                if isinstance(self.vae, CausalVideoAutoencoder):
                    # Account for causal VAE structure
                    latent_overlap_frames = max(1, latent_overlap_frames)

                overlap_latents = prev_latents[:, :, -latent_overlap_frames:]

                if use_guiding_latents:
                    try:
                        overlap_item = ConditioningItem(
                            media_item=overlap_latents.clone(),
                            conditioning_type="guiding",
                            media_frame_number=0,  # Condition at start of new chunk
                            conditioning_strength=temporal_overlap_strength,
                        )
                        chunk_conditioning_items.append(overlap_item)
                        logger.info(
                            f"  Added overlap conditioning: {overlap_item.media_item.shape} with strength {temporal_overlap_strength}"
                        )

                    except Exception as e:
                        logger.info(
                            f"  Warning: Failed to create overlap conditioning: {e}"
                        )

            # Add guiding latents for this chunk if specified
            if kwargs.get("conditioning_items"):
                for item in kwargs["conditioning_items"]:
                    if item.conditioning_type in ["guiding", "guiding_latents"]:
                        # First, encode the guiding video to latent space if not already done
                        if not hasattr(item, "_encoded_latents"):
                            if item.is_latent_space:
                                item._encoded_latents = item.media_item
                            else:
                                item._encoded_latents = vae_encode(
                                    item.media_item.to(
                                        dtype=self.vae.dtype, device=self.vae.device
                                    ),
                                    self.vae,
                                    vae_per_channel_normalize=kwargs.get(
                                        "vae_per_channel_normalize", True
                                    ),
                                    use_tiling=kwargs.get("encode_tiling", False),
                                    tile_size=kwargs.get(
                                        "encode_tile_size", (512, 512)
                                    ),
                                    tile_stride=kwargs.get(
                                        "encode_tile_stride", (256, 256)
                                    ),
                                ).to(dtype=torch.float32)
                                logger.info(
                                    f"  Encoded guiding video: {item.media_item.shape} -> {item._encoded_latents.shape}"
                                )

                        # Now work in latent space for all calculations
                        latent_item_start_frame = (
                            item.media_frame_number // self.video_scale_factor
                        )
                        latent_item_end_frame = (
                            latent_item_start_frame + item._encoded_latents.shape[2]
                        )
                        latent_chunk_start = start_frame // self.video_scale_factor
                        latent_chunk_end = (
                            latent_chunk_start
                            + (chunk_num_frames - 1) // self.video_scale_factor
                            + 1
                        )

                        # Check if this guiding item overlaps with current chunk (in latent space)
                        if (
                            latent_item_start_frame < latent_chunk_end
                            and latent_item_end_frame > latent_chunk_start
                        ):
                            # Calculate overlap region in latent space
                            chunk_start_in_item = max(
                                0, latent_chunk_start - latent_item_start_frame
                            )
                            chunk_end_in_item = min(
                                item._encoded_latents.shape[2],
                                chunk_start_in_item
                                + (latent_chunk_end - latent_chunk_start),
                            )

                            if chunk_end_in_item > chunk_start_in_item:
                                # Ensure frame count satisfies n_frames % 8 == 1
                                extracted_frames = (
                                    chunk_end_in_item - chunk_start_in_item
                                )
                                if extracted_frames % 8 != 1:
                                    adjusted_frames = (
                                        (extracted_frames - 1) // 8
                                    ) * 8 + 1
                                    if adjusted_frames < 1:
                                        adjusted_frames = 1
                                    chunk_end_in_item = (
                                        chunk_start_in_item + adjusted_frames
                                    )
                                    chunk_end_in_item = min(
                                        chunk_end_in_item, item.media_item.shape[2]
                                    )

                            if chunk_end_in_item > chunk_start_in_item:
                                # Extract the relevant portion from encoded latents
                                chunk_item = copy.deepcopy(item)
                                chunk_item.media_item = item._encoded_latents[
                                    :, :, chunk_start_in_item:chunk_end_in_item
                                ]
                                # Convert back to pixel space for frame number
                                chunk_item.media_frame_number = max(
                                    0,
                                    (latent_item_start_frame - latent_chunk_start)
                                    * self.video_scale_factor,
                                )
                                chunk_item.conditioning_strength = guiding_strength
                                chunk_conditioning_items.append(chunk_item)
                                logger.info(
                                    f"  Added guiding conditioning: {chunk_item.media_item.shape} (latent space, chunk {chunk_idx+1})"
                                )

            # Add other conditioning items (like first frame conditioning)
            if kwargs.get("conditioning_items"):
                for item in kwargs["conditioning_items"]:
                    if item.conditioning_type not in ["guiding", "guiding_latents"]:
                        # First frame conditioning should ONLY apply to the first chunk
                        if (
                            item.conditioning_type == "image"
                            and item.media_frame_number == 0
                        ):
                            # Only include first frame conditioning in the first chunk
                            if chunk_idx == 0:
                                chunk_conditioning_items.append(item)
                            # Skip first frame conditioning for all other chunks
                        else:
                            # Include other regular conditioning items if they apply to this chunk
                            # Adjust frame number relative to chunk start
                            item_frame_in_chunk = item.media_frame_number - start_frame
                            if 0 <= item_frame_in_chunk < chunk_num_frames:
                                # Create a copy with adjusted frame number
                                chunk_item = copy.deepcopy(item)
                                chunk_item.media_frame_number = item_frame_in_chunk
                                chunk_conditioning_items.append(chunk_item)
                    # Note: guiding items are handled above and should not be included here to avoid duplication

            chunk_kwargs["conditioning_items"] = (
                chunk_conditioning_items if chunk_conditioning_items else None
            )

            # Handle initialization latents for this chunk
            if kwargs.get("latents") is not None:
                # If we have initialization latents, slice them for this chunk
                full_latents = kwargs["latents"]

                # Calculate latent space dimensions for this chunk
                latent_height = height // self.vae_scale_factor
                latent_width = width // self.vae_scale_factor
                latent_chunk_start = start_frame // self.video_scale_factor
                latent_chunk_frames = chunk_num_frames // self.video_scale_factor
                if isinstance(self.vae, CausalVideoAutoencoder) and is_video:
                    latent_chunk_frames += 1

                latent_chunk_end = latent_chunk_start + latent_chunk_frames

                # Ensure we don't exceed the bounds of the full latents
                latent_chunk_end = min(latent_chunk_end, full_latents.shape[2])
                actual_latent_frames = latent_chunk_end - latent_chunk_start

                if actual_latent_frames > 0:
                    # Slice the latents for this chunk
                    chunk_latents = full_latents[
                        :, :, latent_chunk_start:latent_chunk_end
                    ]
                    chunk_kwargs["latents"] = chunk_latents
                    logger.info(
                        f"  Using sliced latents: {chunk_latents.shape} (from {latent_chunk_start}:{latent_chunk_end})"
                    )
                else:
                    # No latents for this chunk, remove the parameter
                    chunk_kwargs.pop("latents", None)
                    logger.info(f"  No latents available for chunk {chunk_idx+1}")
            else:
                # No initialization latents provided
                if chunk_idx > 0:
                    # For extension chunks, we can optionally provide initialization latents
                    # This is particularly useful when we want to initialize the new chunk
                    # with some prior knowledge or when extending from previous results

                    # Create empty latents for the new chunk if no other initialization is provided
                    latent_height = height // self.vae_scale_factor
                    latent_width = width // self.vae_scale_factor
                    latent_num_frames = chunk_num_frames // self.video_scale_factor
                    if isinstance(self.vae, CausalVideoAutoencoder) and is_video:
                        latent_num_frames += 1

                    batch_size = (
                        kwargs.get("prompt_embeds", torch.ones(1, 1, 1)).shape[0]
                        if kwargs.get("prompt_embeds") is not None
                        else 1
                    )
                    num_images_per_prompt = kwargs.get("num_images_per_prompt", 1)

                    init_latent_shape = (
                        batch_size * num_images_per_prompt,
                        self.transformer.config.in_channels,
                        latent_num_frames,
                        latent_height,
                        latent_width,
                    )

                    # Prepare initialization latents for this chunk
                    chunk_init_latents = self.prepare_latents(
                        latents=None,  # No existing latents for extension chunks
                        media_items=None,  # No media items for extension chunks
                        timestep=1.0,  # Full noise for new content
                        latent_shape=init_latent_shape,
                        dtype=torch.float32,
                        device=device,
                        generator=chunk_kwargs.get("generator"),
                        vae_per_channel_normalize=kwargs.get(
                            "vae_per_channel_normalize", True
                        ),
                        encode_tiling=kwargs.get("encode_tiling", False),
                        encode_tile_size=kwargs.get("encode_tile_size", (512, 512)),
                        encode_tile_stride=kwargs.get("encode_tile_stride", (256, 256)),
                    )

                    # Store initialization latents for potential use in conditioning
                    chunk_kwargs["_chunk_init_latents"] = chunk_init_latents

            # Handle negative index latents for temporal chunks (except the first one)
            if (
                chunk_idx > 0  # Only add to chunks after the first
                and optional_negative_index_latents is not None
                and negative_index_position >= 0
            ):
                # First, ensure the negative index latents are encoded to latent space
                if optional_negative_index_latents.shape[1] == 3:  # RGB channels
                    # Encode from pixel space to latent space
                    neg_idx_latents = vae_encode(
                        optional_negative_index_latents.to(
                            dtype=self.vae.dtype, device=self.vae.device
                        ),
                        self.vae,
                        vae_per_channel_normalize=kwargs.get(
                            "vae_per_channel_normalize", True
                        ),
                        use_tiling=kwargs.get("encode_tiling", False),
                        tile_size=kwargs.get("encode_tile_size", (512, 512)),
                        tile_stride=kwargs.get("encode_tile_stride", (256, 256)),
                    ).to(dtype=torch.float32)
                else:
                    # Already in latent space
                    neg_idx_latents = optional_negative_index_latents

                # Create a ConditioningItem for the negative index latents
                # These will be placed at a negative temporal position
                neg_idx_item = ConditioningItem(
                    media_item=neg_idx_latents.clone(),
                    conditioning_type="negative_index",  # Special type for negative index conditioning
                    media_frame_number=negative_index_position,  # Negative position
                    conditioning_strength=negative_index_strength,
                )

                # Add to chunk conditioning items
                if chunk_kwargs.get("conditioning_items") is None:
                    chunk_kwargs["conditioning_items"] = []
                chunk_kwargs["conditioning_items"].append(neg_idx_item)

                logger.info(
                    f"  Added negative index latents: {neg_idx_latents.shape} at position {negative_index_position} with strength {negative_index_strength}"
                )

            # Run complete denoising process for this chunk
            result = self.__call__(
                height=height,
                width=width,
                num_frames=chunk_num_frames,
                frame_rate=frame_rate,
                **chunk_kwargs,
            )

            # Apply AdaIN if enabled and we have a reference
            if (
                temporal_adain_factor > 0
                and chunk_idx > 0
                and first_chunk_result is not None
            ):
                if hasattr(result, "images"):
                    current_latents = result.images
                else:
                    current_latents = result[0] if isinstance(result, tuple) else result

                if hasattr(first_chunk_result, "images"):
                    reference_latents = first_chunk_result.images
                else:
                    reference_latents = (
                        first_chunk_result[0]
                        if isinstance(first_chunk_result, tuple)
                        else first_chunk_result
                    )

                # Apply AdaIN
                current_latents = adain_filter_latent(
                    current_latents, reference_latents, temporal_adain_factor
                )

                # Update result
                if hasattr(result, "images"):
                    result.images = current_latents
                elif isinstance(result, tuple):
                    result = (current_latents,) + result[1:]
                else:
                    result = current_latents

            output_chunks.append(result)

            # Store first chunk as reference for AdaIN
            if chunk_idx == 0:
                first_chunk_result = result

        # Restore original generator state
        if original_generator_state is not None:
            kwargs["generator"].set_state(original_generator_state)

        # Combine chunks with overlap blending
        logger.info("Combining chunks with temporal blending...")
        final_output = self._combine_temporal_chunks(
            output_chunks,
            chunk_starts,
            chunk_ends,
            temporal_overlap,
            output_type=kwargs.get("output_type", "pil"),
        )

        # Return in the same format as the original pipeline
        if kwargs.get("return_dict", True):
            return ImagePipelineOutput(images=final_output)
        else:
            return (final_output,)

    def _vae_decode(
        self,
        latents: torch.Tensor,
        decode_timestep: Union[List[float], float] = 0.0,
        decode_noise_scale: Optional[List[float]] = None,
        decode_tiling: bool = False,
        decode_tile_size: Tuple[int, int] = (64, 64),
        decode_tile_stride: Tuple[int, int] = (32, 32),
        vae_per_channel_normalize: bool = True,
    ) -> torch.Tensor:
        """
        Decodes.
        """
        if self.vae.decoder.timestep_conditioning:
            noise = torch.randn_like(latents)
            if not isinstance(decode_timestep, list):
                decode_timestep = [decode_timestep] * latents.shape[0]
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            elif not isinstance(decode_noise_scale, list):
                decode_noise_scale = [decode_noise_scale] * latents.shape[0]

            decode_timestep = torch.tensor(decode_timestep).to(latents.device)
            decode_noise_scale = torch.tensor(decode_noise_scale).to(latents.device)[
                :, None, None, None, None
            ]
            latents = latents * (1 - decode_noise_scale) + noise * decode_noise_scale
        else:
            decode_timestep = None

        image = vae_decode(
            latents,
            self.vae,
            True,
            vae_per_channel_normalize=vae_per_channel_normalize,
            timestep=decode_timestep,
            use_tiling=decode_tiling,
            tile_size=decode_tile_size,
            tile_stride=decode_tile_stride,
        )

        return image

    @staticmethod
    def tone_map_latents(
        latents: torch.Tensor,
        compression: float,
    ) -> torch.Tensor:
        """
        Applies a non-linear tone-mapping function to latent values to reduce their dynamic range
        in a perceptually smooth way using a sigmoid-based compression.
        This is useful for regularizing high-variance latents or for conditioning outputs
        during generation, especially when controlling dynamic behavior with a `compression` factor.
        Parameters:
        ----------
        latents : torch.Tensor
            Input latent tensor with arbitrary shape. Expected to be roughly in [-1, 1] or [0, 1] range.
        compression : float
            Compression strength in the range [0, 1].
            - 0.0: No tone-mapping (identity transform)
            - 1.0: Full compression effect
        Returns:
        -------
        torch.Tensor
            The tone-mapped latent tensor of the same shape as input.
        """
        if not (0 <= compression <= 1):
            raise ValueError("Compression must be in the range [0, 1]")

        # Remap [0-1] to [0-0.75] and apply sigmoid compression in one shot
        scale_factor = compression * 0.75
        abs_latents = torch.abs(latents)

        # Sigmoid compression: sigmoid shifts large values toward 0.2, small values stay ~1.0
        # When scale_factor=0, sigmoid term vanishes, when scale_factor=0.75, full effect
        sigmoid_term = torch.sigmoid(4.0 * scale_factor * (abs_latents - 1.0))
        scales = 1.0 - 0.8 * scale_factor * sigmoid_term

        filtered = latents * scales
        return filtered


def adain_filter_latent(
    latents: torch.Tensor, reference_latents: torch.Tensor, factor=1.0
):
    """
    Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on
    statistics from a reference latent tensor.

    Args:
        latent (torch.Tensor): Input latents to normalize
        reference_latent (torch.Tensor): The reference latents providing style statistics.
        factor (float): Blending factor between original and transformed latent.
                       Range: -10.0 to 10.0, Default: 1.0

    Returns:
        torch.Tensor: The transformed latent tensor
    """
    result = latents.clone()

    for i in range(latents.size(0)):
        for c in range(latents.size(1)):
            r_sd, r_mean = torch.std_mean(
                reference_latents[i, c], dim=None
            )  # index by original dim order
            i_sd, i_mean = torch.std_mean(result[i, c], dim=None)

            result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean

    result = torch.lerp(latents, result, factor)
    return result


class LTXMultiScalePipeline:
    def _upsample_latents(
        self, latest_upsampler: LatentUpsampler, latents: torch.Tensor
    ):
        assert latents.device == latest_upsampler.device

        latents = un_normalize_latents(
            latents, self.vae, vae_per_channel_normalize=True
        )
        upsampled_latents = latest_upsampler(latents)
        upsampled_latents = normalize_latents(
            upsampled_latents, self.vae, vae_per_channel_normalize=True
        )
        return upsampled_latents

    def __init__(
        self, video_pipeline: LTXVideoPipeline, latent_upsampler: LatentUpsampler
    ):
        self.video_pipeline = video_pipeline
        self.vae = video_pipeline.vae
        self.latent_upsampler = latent_upsampler

    def load_lora_weights(self, *args, **kwargs):
        return self.video_pipeline.load_lora_weights(*args, **kwargs)

    def unload_lora_weights(self, *args, **kwargs):
        return self.video_pipeline.unload_lora_weights(*args, **kwargs)

    def set_adapters(self, adapter_names, weights):
        weights = [w if w is not None else 1.0 for w in weights]
        set_weights_and_activate_adapters(
            self.video_pipeline.transformer, adapter_names, weights
        )

    def _resize_conditioning_items_for_pass(
        self, conditioning_items: Optional[List], target_height: int, target_width: int
    ) -> Optional[List]:
        """
        Resize conditioning items in pixel space for the target dimensions.
        This is crucial for multi-scale pipeline to avoid lossy latent interpolation.
        """
        if not conditioning_items:
            return None

        resized_items = []
        for item in conditioning_items:
            # Create a copy to avoid modifying the original
            new_item = copy.deepcopy(item)

            # Check if the conditioning item needs resizing
            current_height, current_width = item.media_item.shape[-2:]
            if current_height != target_height or current_width != target_width:
                # Only resize pixel-space items (3 channels or less)
                if item.media_item.shape[1] <= 3:
                    # Resize in pixel space using the pipeline's resize method
                    new_item.media_item = LTXVideoPipeline.resize_tensor(
                        item.media_item, target_height, target_width
                    )
                    logger.info(
                        f"  Resized conditioning {item.conditioning_type}: {item.media_item.shape} -> {new_item.media_item.shape}"
                    )
                else:
                    # For pre-encoded latents, we need to calculate the target latent size
                    # and resize in latent space as a fallback (though this is less ideal)
                    target_latent_height = (
                        target_height // self.video_pipeline.vae_scale_factor
                    )
                    target_latent_width = (
                        target_width // self.video_pipeline.vae_scale_factor
                    )

                    b, c, f, h, w = item.media_item.shape
                    # Reshape to (b*f, c, h, w) for interpolation
                    media_reshaped = item.media_item.permute(0, 2, 1, 3, 4).reshape(
                        b * f, c, h, w
                    )
                    # Resize using bilinear interpolation
                    media_resized = F.interpolate(
                        media_reshaped,
                        size=(target_latent_height, target_latent_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    # Reshape back to (b, c, f, h, w)
                    new_item.media_item = media_resized.reshape(
                        b, f, c, target_latent_height, target_latent_width
                    ).permute(0, 2, 1, 3, 4)
                    logger.info(
                        f"  Resized pre-encoded conditioning {item.conditioning_type}: {item.media_item.shape} -> {new_item.media_item.shape}"
                    )

            resized_items.append(new_item)

        return resized_items

    def __call__(
        self,
        downscale_factor: float,
        first_pass: dict,
        second_pass: dict,
        second_pass_noise_scale: float = 0.0,
        tone_map_compression_ratio: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_kwargs = kwargs.copy()
        original_output_type = kwargs["output_type"]
        original_width = kwargs["width"]
        original_height = kwargs["height"]
        original_conditioning_items = kwargs.get("conditioning_items")

        x_width = int(kwargs["width"] * downscale_factor)
        downscaled_width = x_width - (x_width % self.video_pipeline.vae_scale_factor)
        x_height = int(kwargs["height"] * downscale_factor)
        downscaled_height = x_height - (x_height % self.video_pipeline.vae_scale_factor)

        kwargs["output_type"] = "latent"
        kwargs["width"] = downscaled_width
        kwargs["height"] = downscaled_height
        kwargs["conditioning_items"] = self._resize_conditioning_items_for_pass(
            original_conditioning_items, downscaled_height, downscaled_width
        )

        kwargs.update(**first_pass)

        result = self.video_pipeline(*args, **kwargs)
        latents = result.images
        latents = self.video_pipeline.tone_map_latents(
            latents, tone_map_compression_ratio
        )

        upsampled_latents = self._upsample_latents(self.latent_upsampler, latents)
        upsampled_latents = adain_filter_latent(
            latents=upsampled_latents, reference_latents=latents
        )

        if second_pass_noise_scale > 0:
            noise = torch.randn(
                upsampled_latents.shape,
                device=upsampled_latents.device,
                dtype=upsampled_latents.dtype,
                generator=kwargs.get("generator"),
            )
            upsampled_latents = (
                upsampled_latents * (1 - second_pass_noise_scale)
                + noise * second_pass_noise_scale
            )

        kwargs = original_kwargs

        second_pass_height = downscaled_height * 2
        second_pass_width = downscaled_width * 2

        kwargs["latents"] = upsampled_latents
        kwargs["output_type"] = original_output_type
        kwargs["width"] = second_pass_width
        kwargs["height"] = second_pass_height
        kwargs["conditioning_items"] = self._resize_conditioning_items_for_pass(
            original_conditioning_items, second_pass_height, second_pass_width
        )

        kwargs.update(**second_pass)

        result = self.video_pipeline(*args, **kwargs)
        if original_output_type != "latent":
            num_frames = result.images.shape[2]
            videos = rearrange(result.images, "b c f h w -> (b f) c h w")

            videos = F.interpolate(
                videos,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            videos = rearrange(videos, "(b f) c h w -> b c f h w", f=num_frames)
            result.images = videos

        return result
