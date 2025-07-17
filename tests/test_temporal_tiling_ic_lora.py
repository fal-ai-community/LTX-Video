#!/usr/bin/env python3
"""
Temporal Tiling + IC-LoRA Test Script

This script tests the new temporal tiling functionality combined with IC-LoRA guidance
for generating long videos with temporal consistency.

Usage:
    source /scratch/benjamin/LTX-Trainer/.venv/bin/activate
    PYTHONPATH=. python tests/test_temporal_tiling_ic_lora.py --model_path /path/to/ltx-video
"""

import argparse
import logging
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from einops import rearrange
from ltx_video.inference import create_ltx_video_pipeline, get_device, seed_everething, create_latent_upsampler
from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem, LTXMultiScalePipeline
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_video_frames(
    video_path, target_height=None, target_width=None, target_frames=None
):
    """
    Load video frames with optional resizing and frame count adjustment.

    Args:
        video_path: Path to the video file
        target_height: Target height (None to keep original)
        target_width: Target width (None to keep original)
        target_frames: Target number of frames (None to keep all)

    Returns:
        torch.Tensor: Video tensor of shape (1, C, T, H, W)
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        f"Original video: {original_width}x{original_height}, {original_frame_count} frames, {original_fps:.2f} fps"
    )

    # Use original dimensions if targets not specified
    height = target_height or original_height
    width = target_width or original_width
    num_frames = target_frames or original_frame_count

    # Load frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        frames.append(frame)

    cap.release()

    # Adjust frame count
    current_frames = len(frames)
    if current_frames < num_frames:
        # Repeat last frame
        last_frame = (
            frames[-1] if frames else np.zeros((height, width, 3), dtype=np.uint8)
        )
        frames.extend([last_frame] * (num_frames - current_frames))
        logger.info(f"Extended video from {current_frames} to {num_frames} frames")
    elif current_frames > num_frames:
        # Truncate
        frames = frames[:num_frames]
        logger.info(f"Truncated video from {current_frames} to {num_frames} frames")

    # Convert to tensor format (T, H, W, C) -> (B, C, T, H, W)
    frames_array = np.stack(frames)  # (T, H, W, C)
    frames_tensor = (
        torch.from_numpy(frames_array).float() / 127.5
    )  # Normalize to [0, 2]
    frames_tensor = torch.clamp(frames_tensor - 1.0, -1.0, 1.0)  # Normalize to [-1, 1]
    frames_tensor = rearrange(frames_tensor, "t h w c -> 1 c t h w")
    return frames_tensor


def load_image(image_path):
    """
    Loads an image for conditioning.
    """
    image = Image.open(image_path).convert("RGB")
    image = (torch.from_numpy(np.array(image)).float() / 127.5) - 1.0
    image = rearrange(image, "h w c -> 1 c 1 h w")
    image = torch.clamp(image, -1.0, 1.0)
    print(f"{image.shape=} {image.dtype=}")
    return image


def run_temporal_tiling_ic_lora_test(args):
    """Run the temporal tiling + IC-LoRA test."""

    logger.info("🎬 Starting Temporal Tiling + IC-LoRA Test")
    logger.info("=" * 60)

    # Set seed early
    if args.seed is not None:
        seed_everething(args.seed)
        logger.info(f"🌱 Using seed: {args.seed}")

    # Determine device
    device = get_device() if args.device == "auto" else args.device
    logger.info(f"🖥️  Using device: {device}")

    # Load pipeline
    logger.info(f"📦 Loading LTX-Video pipeline from: {args.model_path}")
    text_encoder_path = args.text_encoder_path or "PixArt-alpha/PixArt-XL-2-1024-MS"

    pipeline = create_ltx_video_pipeline(
        ckpt_path=args.model_path,
        precision=args.precision,
        text_encoder_model_name_or_path=text_encoder_path,
        sampler=args.sampler,
        device=device,
        enhance_prompt=False,
    )

    if args.upsampler_model_path:
        logger.info(f"📦 Loading Latent Upsampler from {args.upsampler_model_path}")
        upsampler = create_latent_upsampler(
            args.upsampler_model_path,
            device=device
        )
        pipeline = LTXMultiScalePipeline(
            pipeline,
            upsampler
        )

    # Prepare conditioning items
    conditioning_items = []

    # Load first frame conditioning if provided
    if args.first_frame and Path(args.first_frame).exists():
        logger.info(f"🖼️  Loading first frame from: {args.first_frame}")
        first_frame = load_image(args.first_frame)
        first_frame_conditioning = ConditioningItem(
            media_item=first_frame,
            conditioning_type="image",
            media_frame_number=0,
            conditioning_strength=1.0,
        )
        conditioning_items.append(first_frame_conditioning)

    # Load and process depth IC-LoRA if provided
    if args.depth_video and Path(args.depth_video).exists():
        logger.info(f"🏔️  Loading IC-LoRA depth weights from: {args.ic_lora_depth_repo}")
        pipeline.load_lora_weights(
            args.ic_lora_depth_repo,
            weight_name=args.ic_lora_depth_filename,
            adapter_name="depth",
        )

        # Load depth video
        logger.info(f"📹 Loading depth video from: {args.depth_video}")
        depth_video = load_video_frames(
            args.depth_video,
            target_height=args.height,
            target_width=args.width,
            target_frames=args.num_frames,
        )

        # Create conditioning item for depth video (fixed: use "guiding" not "guiding_latents")
        depth_conditioning = ConditioningItem(
            media_item=depth_video,
            conditioning_type="guiding",  # Fixed: correct type for IC-LoRA
            media_frame_number=args.depth_start_frame,
            conditioning_strength=args.depth_strength,
        )
        conditioning_items.append(depth_conditioning)

    # Load and process pose IC-LoRA if provided
    if args.pose_video and Path(args.pose_video).exists():
        logger.info(f"🕺 Loading IC-LoRA pose weights from: {args.ic_lora_pose_repo}")
        pipeline.load_lora_weights(
            args.ic_lora_pose_repo,
            weight_name=args.ic_lora_pose_filename,
            adapter_name="pose",
        )

        # Load pose video
        logger.info(f"📹 Loading pose video from: {args.pose_video}")
        pose_video = load_video_frames(
            args.pose_video,
            target_height=args.height,
            target_width=args.width,
            target_frames=args.num_frames,
        )

        # Create conditioning item for pose video
        pose_conditioning = ConditioningItem(
            media_item=pose_video,
            conditioning_type="guiding",  # Fixed: correct type for IC-LoRA
            media_frame_number=args.pose_start_frame,
            conditioning_strength=args.pose_strength,
        )
        conditioning_items.append(pose_conditioning)

    # Set adapter weights if we have multiple LoRAs
    if args.depth_video and args.pose_video and Path(args.depth_video).exists() and Path(args.pose_video).exists():
        logger.info(
            f"⚖️  Setting adapter weights - depth: {args.depth_lora_weight}, pose: {args.pose_lora_weight}"
        )
        pipeline.set_adapters(
            ["depth", "pose"], [args.depth_lora_weight, args.pose_lora_weight]
        )
    elif args.depth_video and Path(args.depth_video).exists():
        pipeline.set_adapters(["depth"], [1.0])
    elif args.pose_video and Path(args.pose_video).exists():
        pipeline.set_adapters(["pose"], [1.0])

    # Display temporal tiling configuration
    logger.info("🧩 Temporal Tiling Configuration:")
    logger.info(f"   📏 Total frames: {args.num_frames}")
    logger.info(f"   🔲 Tile size: {args.temporal_tile_size} frames")
    logger.info(f"   📐 Overlap: {args.temporal_overlap} frames")
    logger.info(f"   💪 Overlap strength: {args.temporal_overlap_strength}")
    logger.info(f"   🎯 Use guiding latents: {args.use_guiding_latents}")
    logger.info(f"   🎚️  Guiding strength: {args.guiding_strength}")
    logger.info(f"   🔄 AdaIN factor: {args.temporal_adain_factor}")

    # Calculate expected chunks for demonstration
    if args.temporal_tile_size and args.temporal_overlap:
        video_scale_factor = 8  # LTX Video temporal compression
        latent_tile_size = args.temporal_tile_size // video_scale_factor
        latent_overlap = args.temporal_overlap // video_scale_factor
        total_latent_frames = args.num_frames // video_scale_factor + 1
        
        chunk_starts = list(range(0, total_latent_frames, latent_tile_size - latent_overlap))
        chunk_ends = [min(start + latent_tile_size, total_latent_frames) for start in chunk_starts]
        
        logger.info(f"   🔢 Expected chunks: {len(chunk_starts)}")
        for i, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
            chunk_frames = (end - start - 1) * video_scale_factor + 1
            logger.info(f"      Chunk {i+1}: latent {start}-{end} ({chunk_frames} pixel frames)")

# {'device': 'cuda', 'generator': <torch._C.Generator object at 0x7f9f71c644b0>, 'callback_on_step_end': None, 'output_type': 'pt', 'num_images_per_prompt': 1, 'is_video': True, 'mixed_precision': True, 'offload_to_cpu': False, 'enhance_prompt': False, 'vae_per_channel_normalize': True, 'temporal_tile_size': 80, 'temporal_overlap': 24, 'use_guiding_latents': True, 'temporal_adain_factor': 0.0, 'temporal_overlap_strength': 0.5, 'guiding_strength': 1.0, 'decode_tiling': True, 'decode_tile_size': (16, 16), 'decode_tile_stride': (14, 14), 'height': 704, 'width': 1280, 'num_frames': 193, 'frame_rate': 25, 'prompt': "A cinematic fast-tracking shot follows a vintage, teal camper van as it descends a winding mountain trail. The van, slightly weathered but well-maintained, is the central focus, its retro design emphasized by the motion blur. Medium shot reveals the dusty, ochre trail, edged with vibrant green pine trees. Close-up on the van's tires shows the gravel spraying, highlighting the speed and rugged terrain. Sunlight filters through the trees, casting dappled shadows on the van and the trail. The background is a hazy, majestic mountain range bathed in warm, golden light. The overall mood is adventurous and exhilarating. High resolution 4k movie scene.", 'negative_prompt': 'worst quality, inconsistent motion, blurry, jittery, distorted', 'skip_layer_strategy': <SkipLayerStrategy.AttentionValues: 2>, 'conditioning_items': None, 'stochastic_sampling': False, 'decode_timestep': 0.05, 'decode_noise_scale': 0.025, 'downscale_factor': 0.6666666, 'first_pass': {'guidance_timesteps': [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725], 'guidance_scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'stg_scale': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'rescaling_scale': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 'skip_block_list': [[42], [42], [42], [42], [42], [42], [42]], 'num_inference_steps': 8, 'skip_initial_inference_steps': 0, 'skip_final_inference_steps': 1, 'cfg_star_rescale': False}, 'second_pass': {'guidance_timesteps': [0.9094, 0.725, 0.4219], 'guidance_scale': [1.0, 1.0, 1.0], 'stg_scale': [0.0, 0.0, 0.0], 'rescaling_scale': [1.0, 1.0, 1.0], 'skip_block_list': [[42], [42], [42]], 'num_inference_steps': 8, 'skip_initial_inference_steps': 4, 'skip_final_inference_steps': 0, 'cfg_star_rescale': False, 'tone_map_compression_ratio': 0.6}}##

    # Generation parameters
    generation_params = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "prompt": args.prompt,
        "negative_prompt": 'worst quality, inconsistent motion, blurry, jittery, distorted',
        "conditioning_items": conditioning_items,
        "is_video": True,
        "vae_per_channel_normalize": True,
        "output_type": "latent",  # Get tensors directly
        # NEW: Temporal tiling parameters
        "temporal_tile_size": args.temporal_tile_size,
        "temporal_overlap": args.temporal_overlap,
        "temporal_overlap_strength": args.temporal_overlap_strength,
        "use_guiding_latents": args.use_guiding_latents,
        "guiding_strength": args.guiding_strength,
        "temporal_adain_factor": args.temporal_adain_factor,
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.25,
        "guidance_scale": 1.0,
        "num_inference_steps": args.num_inference_steps,
        "stg_scale": 0.0,
    }

    if args.upsampler_model_path:
        generation_params["downscale_factor"] = 0.66667
        generation_params["first_pass"] = {
            #"skip_final_inference_steps": 1,
            #"guidance_timesteps": [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725],
            #"num_inference_steps": 8,
            #"guidance_scale": [1.0] * 7,
            #"stg_scale": [0.0] * 7,
            #"rescaling_scale": [1.0] * 7,
            #"skip_block_list": [[42] for i in range(7)],
        }
        generation_params["second_pass"] = {
            #"skip_initial_inference_steps": 5,
            #"guidance_timesteps": [0.9094, 0.725, 0.4219],
            #"num_inference_steps": 8,
            #"guidance_scale": [1.0] * 3,
            #"stg_scale": [0.0] * 3,
            #"rescaling_scale": [1.0] * 3,
            #"skip_block_list": [[42] for i in range(3)],
        }

    # Add optional parameters
    if args.negative_prompt:
        generation_params["negative_prompt"] = args.negative_prompt

    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        generation_params["generator"] = generator

    # Generate video with temporal tiling
    logger.info("🎬 Generating video with Temporal Tiling + IC-LoRA...")
    logger.info(f"💬 Prompt: {args.prompt}")
    logger.info(f"🎛️  Total conditioning items: {len(conditioning_items)}")
    for i, item in enumerate(conditioning_items):
        logger.info(
            f"   {i+1}. Type: {item.conditioning_type}, Strength: {item.conditioning_strength}, Frame: {item.media_frame_number}"
        )

    result = pipeline(**generation_params)

    # Save result
    output_path = Path(args.output_dir) / f"temporal_tiling_ic_lora_{args.seed or 'random'}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert latent result to video frames for saving
    if hasattr(result, "images"):
        video_latents = result.images[0]  # (C, T, H, W) in latent space
    elif isinstance(result, tuple):
        video_latents = result[0][0] if isinstance(result[0], tuple) else result[0]
    else:
        video_latents = result

    # Decode latents to pixel space
    print(f"Received result of shape {video_latents.shape=}")
    logger.info("🎞️  Decoding latents to video frames...")
    from ltx_video.models.autoencoders.vae_encode import vae_decode
    video_frames = vae_decode(
        video_latents.unsqueeze(0),  # Add batch dimension if needed
        pipeline.vae,
        is_video=True,
        vae_per_channel_normalize=True,
        timestep=torch.tensor([0.0], device=video_latents.device),  # Default timestep for decoding
        use_tiling=True,
        tile_size=(16, 16),
        tile_stride=(14, 14),
    )
    
    # Convert from (B, C, T, H, W) to (T, H, W, C)
    if video_frames.dim() == 5:
        video_frames = video_frames[0]  # Remove batch dimension
    video_frames = video_frames.permute(1, 2, 3, 0).float().cpu().detach().numpy()

    # Normalize to 0-255 range
    video_frames = (video_frames + 1.0) / 2.0  # From [-1,1] to [0,1]
    video_frames = np.clip(video_frames * 255, 0, 255).astype(np.uint8)

    # Save as MP4
    imageio.mimsave(
        str(output_path), video_frames, fps=args.frame_rate, codec="libx264"
    )
    logger.info(f"💾 Saved generated video to: {output_path}")

    # Unload LoRA for cleanup
    pipeline.unload_lora_weights()
    logger.info("✅ Temporal Tiling + IC-LoRA test completed successfully!")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Temporal Tiling + IC-LoRA Test")

    # Model and data paths
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to LTX-Video model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--upsampler_model_path",
        type=str,
        default="",
        help="Path to LTX-Video upsampler model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="PixArt-alpha/PixArt-XL-2-1024-MS",
        help="Path to text encoder model",
    )

    # IC-LoRA configurations for depth
    parser.add_argument(
        "--ic_lora_depth_repo",
        type=str,
        default="Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7",
        help="HuggingFace repo for depth IC-LoRA weights",
    )
    parser.add_argument(
        "--ic_lora_depth_filename",
        type=str,
        default="ltxv-097-ic-lora-depth-control-comfyui.safetensors",
        help="Filename for depth IC-LoRA weights",
    )
    parser.add_argument(
        "--depth_video",
        type=str,
        default="/scratch/benjamin/depth.mp4",
        help="Path to depth/structure video",
    )
    parser.add_argument(
        "--depth_strength",
        type=float,
        default=1.0,
        help="Strength of depth guiding latents (0.0-1.0)",
    )
    parser.add_argument(
        "--depth_start_frame",
        type=int,
        default=0,
        help="Start frame for depth guidance",
    )
    parser.add_argument(
        "--depth_lora_weight",
        type=float,
        default=1.0,
        help="Weight for depth LoRA when using multiple LoRAs",
    )

    # IC-LoRA configurations for pose
    parser.add_argument(
        "--ic_lora_pose_repo",
        type=str,
        default="Lightricks/LTX-Video-ICLoRA-pose-13b-0.9.7",
        help="HuggingFace repo for pose IC-LoRA weights",
    )
    parser.add_argument(
        "--ic_lora_pose_filename",
        type=str,
        default="ltxv-097-ic-lora-pose-control-comfyui.safetensors",
        help="Filename for pose IC-LoRA weights",
    )
    parser.add_argument(
        "--pose_video",
        type=str,
        default="/scratch/benjamin/pose.mp4",
        help="Path to pose video",
    )
    parser.add_argument(
        "--pose_strength",
        type=float,
        default=1.0,
        help="Strength of pose guiding latents (0.0-1.0)",
    )
    parser.add_argument(
        "--pose_start_frame", type=int, default=0, help="Start frame for pose guidance"
    )
    parser.add_argument(
        "--pose_lora_weight",
        type=float,
        default=1.0,
        help="Weight for pose LoRA when using multiple LoRAs",
    )

    # First frame conditioning
    parser.add_argument(
        "--first_frame",
        type=str,
        default="/scratch/benjamin/first-frame.jpg",
        help="Path to first frame image",
    )

    # Temporal tiling parameters
    parser.add_argument(
        "--temporal_tile_size",
        type=int,
        default=48,  # Smaller than 81 to ensure windowing
        help="Size of temporal tiles (pixel frames)",
    )
    parser.add_argument(
        "--temporal_overlap",
        type=int,
        default=16,  # Reasonable overlap
        help="Overlap between temporal tiles (pixel frames)",
    )
    parser.add_argument(
        "--temporal_overlap_strength",
        type=float,
        default=0.5,
        help="Conditioning strength from previous chunk",
    )
    parser.add_argument(
        "--use_guiding_latents",
        type=bool,
        default=True,
        help="Whether to use conditioning_items as IC-LoRA guides",
    )
    parser.add_argument(
        "--guiding_strength",
        type=float,
        default=1.0,
        help="Strength of IC-LoRA conditioning",
    )
    parser.add_argument(
        "--temporal_adain_factor",
        type=float,
        default=0.2,
        help="AdaIN factor for temporal consistency (0.0=disabled)",
    )

    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="A confident woman strides toward the camera down a sun-drenched, empty street. Her vibrant summer dress, a flowing emerald green with delicate white floral embroidery, billows slightly in the gentle breeze. The mood is optimistic and serene, emphasizing the woman's independence and carefree spirit. High resolution 4k",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt", type=str, default="", help="Negative prompt"
    )

    # Video dimensions (same as IC-LoRA test)
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=704, help="Video height")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--frame_rate", type=int, default=15, help="Frame rate")

    # Guidance parameters
    parser.add_argument(
        "--guidance_scale", type=float, default=4.5, help="CFG guidance scale"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=20, help="Number of denoising steps"
    )

    # Technical parameters
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Model precision",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="from_checkpoint",
        choices=["from_checkpoint", "uniform", "linearquadratic"],
        help="Sampler type",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for generated videos",
    )

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    if args.temporal_tile_size <= args.temporal_overlap:
        raise ValueError("temporal_tile_size must be greater than temporal_overlap")

    if args.temporal_tile_size >= args.num_frames:
        logger.warning(f"temporal_tile_size ({args.temporal_tile_size}) >= num_frames ({args.num_frames}). No tiling will occur.")

    # Run test
    try:
        output_path = run_temporal_tiling_ic_lora_test(args)
        print(f"\n🎉 Success! Generated video with temporal tiling saved to: {output_path}")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
