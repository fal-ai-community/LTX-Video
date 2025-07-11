#!/usr/bin/env python3
"""
Complete IC-LoRA + Guiding Latents Test Script

This script provides a production-ready example of using IC-LoRA with
the new guiding latents feature in LTX-Video pipeline.

Usage:
    python test_ic_lora_complete.py --model_path /path/to/ltx-video --depth_video /path/to/depth.mp4
"""

import argparse
import logging
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from ltx_video.inference import create_ltx_video_pipeline, get_device, seed_everething
from ltx_video.models.autoencoders.vae_encode import vae_encode

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

        # Convert BGR to RGB
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
        torch.from_numpy(frames_array).float() / 255.0
    )  # Normalize to [0, 1]
    frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)

    logger.info(f"Final tensor shape: {frames_tensor.shape}")
    return frames_tensor


def create_guiding_latents_from_depth_video(pipeline, depth_video_tensor):
    """
    Create guiding latents from a depth video using IC-LoRA.

    This function demonstrates the proper workflow:
    1. Encode depth video to latents
    2. Apply IC-LoRA processing (when available)
    3. Return guiding latents

    Args:
        pipeline: LTX-Video pipeline with IC-LoRA loaded
        depth_video_tensor: Depth video tensor (1, C, T, H, W)

    Returns:
        torch.Tensor: Guiding latents for the pipeline
    """
    logger.info("Creating guiding latents from depth video...")

    with torch.no_grad():
        # Move to device and correct dtype
        device = pipeline.device if hasattr(pipeline, "device") else get_device()
        depth_video_tensor = depth_video_tensor.to(device, dtype=pipeline.vae.dtype)

        # Encode the depth video using the VAE
        logger.info("Encoding depth video with VAE...")
        print(f"{depth_video_tensor.shape=}")
        latents = vae_encode(
            depth_video_tensor, pipeline.vae, vae_per_channel_normalize=True
        )

        # The IC-LoRA would process these latents to create structure-aware guidance
        # For now, we return the encoded latents directly
        # In a real IC-LoRA setup, additional processing would happen here

        guiding_latents = latents.to(dtype=torch.float32)
        logger.info(f"Generated guiding latents with shape: {guiding_latents.shape}")

        return guiding_latents


def run_ic_lora_generation(args):
    """Run the complete IC-LoRA + guiding latents generation."""

    logger.info("🚀 Starting IC-LoRA guided video generation...")

    # Set seed early
    if args.seed is not None:
        seed_everething(args.seed)
        logger.info(f"Using seed: {args.seed}")

    # Determine device
    device = get_device() if args.device == "auto" else args.device
    logger.info(f"Using device: {device}")

    # Load the pipeline using the proper method from inference.py
    logger.info(f"Loading LTX-Video pipeline from: {args.model_path}")

    # Default text encoder path - you may need to adjust this
    text_encoder_path = args.text_encoder_path or "google/t5-v1_1-xxl"

    pipeline = create_ltx_video_pipeline(
        ckpt_path=args.model_path,
        precision=args.precision,
        text_encoder_model_name_or_path=text_encoder_path,
        sampler=args.sampler,
        device=device,
        enhance_prompt=False,  # Can be made configurable
    )

    # Load IC-LoRA weights
    logger.info(f"Loading IC-LoRA weights from: {args.ic_lora_repo}")
    pipeline.load_lora_weights(
        args.ic_lora_repo,
        weight_name=args.ic_lora_filename,
    )

    # Load depth video
    logger.info(f"Loading depth video from: {args.depth_video}")
    depth_video = load_video_frames(
        args.depth_video,
        target_height=args.height,
        target_width=args.width,
        target_frames=args.num_frames,
    )

    # Create guiding latents
    guiding_latents = create_guiding_latents_from_depth_video(pipeline, depth_video)

    # Generation parameters
    generation_params = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "prompt": args.prompt,
        "guiding_latents": guiding_latents,
        "guiding_latents_strength": args.guidance_strength,
        "guiding_latents_start_frame": args.start_frame,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "is_video": True,
        "vae_per_channel_normalize": True,
        "output_type": "pt",  # Get tensors directly
    }

    # Add optional parameters
    if args.negative_prompt:
        generation_params["negative_prompt"] = args.negative_prompt

    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        generation_params["generator"] = generator

    # Generate video
    logger.info("🎬 Generating video with IC-LoRA guidance...")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Guidance strength: {args.guidance_strength}")

    result = pipeline(**generation_params)

    # Save result
    output_path = Path(args.output_dir) / f"ic_lora_guided_{args.seed or 'random'}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert result to numpy for saving
    # The pipeline returns ImagePipelineOutput with .images attribute or a tuple
    if hasattr(result, "images"):
        video_frames = result.images[0]  # (C, T, H, W)
    elif isinstance(result, tuple):
        # If it's a tuple, take the first element
        video_frames = result[0]
    else:
        # Direct tensor result
        video_frames = result

    # Convert from (C, T, H, W) to (T, H, W, C)
    if torch.is_tensor(video_frames):
        video_frames = video_frames.permute(1, 2, 3, 0).float().cpu().numpy()

    # Ensure frames are in correct format and range
    if video_frames.max() <= 1.0:
        video_frames = (video_frames * 255).astype(np.uint8)

    # Save as MP4
    imageio.mimsave(
        str(output_path), video_frames, fps=args.frame_rate, codec="libx264"
    )
    logger.info(f"💾 Saved generated video to: {output_path}")

    # Unload LoRA for cleanup
    pipeline.unload_lora_weights()
    logger.info("✅ Generation completed successfully!")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="IC-LoRA + Guiding Latents Test")

    # Model and data paths
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to LTX-Video model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="PixArt-alpha/PixArt-XL-2-1024-MS",
        help="Path to text encoder model",
    )
    parser.add_argument(
        "--ic_lora_repo",
        type=str,
        default="Lightricks/LTX-Video-ICLoRA-depth-13b-0.9.7",
        help="HuggingFace repo for IC-LoRA weights",
    )
    parser.add_argument(
        "--ic_lora_filename",
        type=str,
        default="ltxv-097-ic-lora-depth-control-comfyui.safetensors",
        help="Filename in HuggingFace repo for IC-LoRA weights",
    )
    parser.add_argument(
        "--depth_video",
        type=str,
        default="/scratch/benjamin/depth.mp4",
        help="Path to depth/structure video",
    )

    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        default="A confident woman strides toward the camera down a sun-drenched, empty street. Her vibrant summer dress, a flowing emerald green with delicate white floral embroidery, billows slightly in the gentle breeze. She carries a stylish, woven straw bag, its natural tan contrasting beautifully with the dress. The dress's fabric shimmers subtly, catching the light. The white embroidery is intricate, each tiny flower meticulously detailed. Her expression is focused, yet relaxed, radiating self-assuredness. Her auburn hair, partially pulled back in a loose braid, catches the sunlight, creating warm highlights. The street itself is paved with warm, grey cobblestones, reflecting the bright sun. The mood is optimistic and serene, emphasizing the woman's independence and carefree spirit. High resolution 4k",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt", type=str, default="", help="Negative prompt"
    )

    # Video dimensions
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--frame_rate", type=int, default=15, help="Frame rate")

    # Guidance parameters
    parser.add_argument(
        "--guidance_strength",
        type=float,
        default=0.9,
        help="Strength of guiding latents (0.0-1.0)",
    )
    parser.add_argument(
        "--start_frame", type=int, default=0, help="Start frame for guidance"
    )
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
        "--seed", type=int, default=None, help="Random seed for reproducibility"
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

    if not Path(args.depth_video).exists():
        raise FileNotFoundError(f"Depth video not found: {args.depth_video}")

    # Run generation
    try:
        output_path = run_ic_lora_generation(args)
        print(f"\n🎉 Success! Generated video saved to: {output_path}")

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
