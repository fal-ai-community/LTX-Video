from typing import Tuple

import torch
from diffusers import AutoencoderKL
from einops import rearrange, repeat
from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.video_autoencoder import (
    Downsample3D,
    VideoAutoencoder,
)
from torch import Tensor
from tqdm import tqdm

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


def sliding_2d_windows(
    height: int,
    width: int,
    tile_size: int | tuple[int, int],
    tile_stride: int | tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """
    Gets windows over a height/width using a square tile.

    :param height: The height of the area.
    :param width: The width of the area.
    :param tile_size: The size of the tile.
    :param tile_stride: The stride of the tile.

    :return: A list of tuples representing the windows in the format (top, bottom, left, right).
    """
    if isinstance(tile_size, tuple):
        tile_width, tile_height = tile_size
    else:
        tile_width = tile_height = int(tile_size)

    if isinstance(tile_stride, tuple):
        tile_stride_width, tile_stride_height = tile_stride
    else:
        tile_stride_width = tile_stride_height = int(tile_stride)

    height_list = list(range(0, height - tile_height + 1, tile_stride_height))
    if (height - tile_height) % tile_stride_height != 0:
        height_list.append(height - tile_height)

    width_list = list(range(0, width - tile_width + 1, tile_stride_width))
    if (width - tile_width) % tile_stride_width != 0:
        width_list.append(width - tile_width)

    coords: list[tuple[int, int, int, int]] = []
    for height in height_list:
        for width in width_list:
            coords.append((height, height + tile_height, width, width + tile_width))

    return coords


def build_1d_mask(
    length: int, left_bound: bool, right_bound: bool, border_width: int
) -> torch.Tensor:
    """
    Builds a 1D mask.

    :param length: length
    :param left_bound: left bound
    :param right_bound: right bound
    :param border_width: border width
    :return: mask
    """
    x = torch.ones((length,))

    if not left_bound:
        x[:border_width] = (torch.arange(border_width) + 1) / border_width
    if not right_bound:
        x[-border_width:] = torch.flip(
            (torch.arange(border_width) + 1) / border_width, dims=(0,)
        )

    return x


def build_mask(
    data: torch.Tensor,
    is_bound: tuple[bool, bool, bool, bool],
    border_width: tuple[int, int],
) -> torch.Tensor:
    """
    :param data: data tensor [B, C, T, H, W] or [B, C, H, W]
    :param is_bound: bound toggle for each side (top, bottom, left, right)
    :param border_width: border width (h, w)
    :return: mask tensor [1, 1, 1, H, W] or [1, 1, H, W]
    """
    if data.dim() == 5:  # Video tensor [B, C, T, H, W]
        _, _, _, H, W = data.shape
        h = build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
    else:  # Image tensor [B, C, H, W]
        _, _, H, W = data.shape
        h = build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 H W")

    return mask


@torch.no_grad()
def tiled_encode(
    media_items: Tensor,
    vae: AutoencoderKL,
    tile_size: tuple[int, int] = (512, 512),
    tile_stride: tuple[int, int] = (256, 256),
    vae_per_channel_normalize: bool = False,
) -> Tensor:
    """
    Tiled encoding for large media items to save memory.

    Args:
        media_items: Input tensor [B, C, H, W] or [B, C, T, H, W]
        vae: VAE model
        tile_size: Size of each tile (height, width)
        tile_stride: Stride between tiles (height, width)
        vae_per_channel_normalize: Whether to use per-channel normalization

    Returns:
        Tensor: Encoded latents
    """
    is_video_shaped = media_items.dim() == 5
    device = media_items.device
    dtype = media_items.dtype

    if is_video_shaped:
        batch_size, channels, num_frames, height, width = media_items.shape
        # For video VAEs, process the full video; for image VAEs, process frame by frame
        if isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)):
            # Video VAE can handle temporal dimension
            H, W = height, width
            size_h, size_w = tile_size
            stride_h, stride_w = tile_stride

            size_h = min(size_h, H)
            size_w = min(size_w, W)
            stride_h = min(stride_h, size_h)
            stride_w = min(stride_w, size_w)

            # Create tiles
            tasks = sliding_2d_windows(
                height=H,
                width=W,
                tile_size=(size_w, size_h),
                tile_stride=(stride_w, stride_h),
            )

            # Get output dimensions from VAE scaling
            temporal_scale, spatial_scale, _ = get_vae_size_scale_factor(vae)
            out_h = H // spatial_scale
            out_w = W // spatial_scale

            # Determine actual latent dimensions by encoding a test tile
            test_input = media_items[:1, :, :, :size_h, :size_w].to(vae.dtype)
            with torch.no_grad():
                test_latent = vae.encode(test_input).latent_dist.sample()
                latent_channels = test_latent.shape[1]
                out_t = test_latent.shape[2]  # Use actual temporal dimension from VAE

            # Initialize accumulation tensors
            weight = torch.zeros(
                (1, 1, out_t, out_h, out_w), dtype=dtype, device=device
            )
            values = torch.zeros(
                (batch_size, latent_channels, out_t, out_h, out_w),
                dtype=dtype,
                device=device,
            )

            # Process each tile
            for h, h_, w, w_ in tqdm(
                tasks, desc="Encoding", total=len(tasks), unit="tile"
            ):
                # Extract tile
                tile = media_items[:, :, :, h:h_, w:w_].to(vae.dtype)

                # Encode tile
                tile_latent = vae.encode(tile).latent_dist.sample()
                tile_latent = normalize_latents(
                    tile_latent, vae, vae_per_channel_normalize
                )
                tile_latent = tile_latent.to(dtype).to(device)

                # Create mask for blending
                mask = build_mask(
                    tile_latent,
                    is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                    border_width=(
                        (size_h - stride_h) // spatial_scale,
                        (size_w - stride_w) // spatial_scale,
                    ),
                ).to(dtype=dtype, device=device)

                # Add to accumulation
                target_h = h // spatial_scale
                target_w = w // spatial_scale
                tile_h, tile_w = tile_latent.shape[-2:]

                values[
                    :, :, :, target_h : target_h + tile_h, target_w : target_w + tile_w
                ] += (tile_latent * mask)
                weight[
                    :, :, :, target_h : target_h + tile_h, target_w : target_w + tile_w
                ] += mask

            # Normalize by weights
            latents = values / weight
            return latents
        else:
            # Image VAE - process frame by frame then tile spatially
            media_items = rearrange(media_items, "b c t h w -> (b t) c h w")
            latents = tiled_encode(
                media_items, vae, tile_size, tile_stride, vae_per_channel_normalize
            )
            # Reshape back to video format
            latents = rearrange(latents, "(b t) c h w -> b c t h w", b=batch_size)
            return latents
    else:
        # Image case [B, C, H, W]
        batch_size, channels, height, width = media_items.shape
        H, W = height, width
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        size_h = min(size_h, H)
        size_w = min(size_w, W)
        stride_h = min(stride_h, size_h)
        stride_w = min(stride_w, size_w)

        # Create tiles
        tasks = sliding_2d_windows(
            height=H,
            width=W,
            tile_size=(size_w, size_h),
            tile_stride=(stride_w, stride_h),
        )

        # Get output dimensions
        _, spatial_scale, _ = get_vae_size_scale_factor(vae)
        out_h = H // spatial_scale
        out_w = W // spatial_scale

        # Determine latent channels
        test_input = media_items[:1, :, :size_h, :size_w].to(vae.dtype)
        with torch.no_grad():
            test_latent = vae.encode(test_input).latent_dist.sample()
            latent_channels = test_latent.shape[1]

        # Initialize accumulation tensors
        weight = torch.zeros((1, 1, out_h, out_w), dtype=dtype, device=device)
        values = torch.zeros(
            (batch_size, latent_channels, out_h, out_w), dtype=dtype, device=device
        )

        # Process each tile
        for h, h_, w, w_ in tasks:
            # Extract tile
            tile = media_items[:, :, h:h_, w:w_].to(vae.dtype)

            # Encode tile
            tile_latent = vae.encode(tile).latent_dist.sample()
            tile_latent = normalize_latents(tile_latent, vae, vae_per_channel_normalize)
            tile_latent = tile_latent.to(dtype).to(device)

            # Create mask for blending
            mask = build_mask(
                tile_latent,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) // spatial_scale,
                    (size_w - stride_w) // spatial_scale,
                ),
            ).to(dtype=dtype, device=device)

            # Add to accumulation
            target_h = h // spatial_scale
            target_w = w // spatial_scale
            tile_h, tile_w = tile_latent.shape[-2:]

            values[
                :, :, target_h : target_h + tile_h, target_w : target_w + tile_w
            ] += (tile_latent * mask)
            weight[
                :, :, target_h : target_h + tile_h, target_w : target_w + tile_w
            ] += mask

        # Normalize by weights
        latents = values / weight
        return latents


@torch.no_grad()
def tiled_decode(
    latents: Tensor,
    vae: AutoencoderKL,
    tile_size: tuple[int, int] = (64, 64),
    tile_stride: tuple[int, int] = (32, 32),
    is_video: bool = True,
    vae_per_channel_normalize: bool = False,
    timestep=None,
) -> Tensor:
    """
    Tiled decoding for large latents to save memory.

    Args:
        latents: Input latent tensor [B, C, H, W] or [B, C, T, H, W]
        vae: VAE model
        tile_size: Size of each tile (height, width) in latent space
        tile_stride: Stride between tiles (height, width) in latent space
        is_video: Whether the input represents video
        vae_per_channel_normalize: Whether to use per-channel normalization
        timestep: Optional timestep for VAE decoding

    Returns:
        Tensor: Decoded media
    """
    is_video_shaped = latents.dim() == 5
    device = latents.device
    dtype = latents.dtype

    if is_video_shaped:
        batch_size, channels, num_frames, height, width = latents.shape

        if isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)):
            # Video VAE can handle temporal dimension
            H, W = height, width
            size_h, size_w = tile_size
            stride_h, stride_w = tile_stride

            size_h = min(size_h, H)
            size_w = min(size_w, W)
            stride_h = min(stride_h, size_h)
            stride_w = min(stride_w, size_w)

            # Create tiles
            tasks = sliding_2d_windows(
                height=H,
                width=W,
                tile_size=(size_w, size_h),
                tile_stride=(stride_w, stride_h),
            )

            # Get output dimensions from actual VAE decode
            temporal_scale, spatial_scale, _ = get_vae_size_scale_factor(vae)
            out_h = H * spatial_scale
            out_w = W * spatial_scale

            # Determine actual output dimensions by decoding a test tile
            test_latent = latents[:1, :, :, :size_h, :size_w]
            test_latent = un_normalize_latents(
                test_latent, vae, vae_per_channel_normalize
            )

            decode_kwargs = {"return_dict": False}
            if timestep is not None:
                decode_kwargs["timestep"] = timestep
            if isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)):
                decode_kwargs["target_shape"] = (
                    1,
                    3,
                    test_latent.shape[2] * temporal_scale if is_video else 1,
                    test_latent.shape[3] * spatial_scale,
                    test_latent.shape[4] * spatial_scale,
                )

            with torch.no_grad():
                test_decoded = vae.decode(test_latent.to(vae.dtype), **decode_kwargs)[0]
                out_t = test_decoded.shape[2]  # Use actual temporal dimension from VAE

            # Initialize accumulation tensors
            weight = torch.zeros(
                (1, 1, out_t, out_h, out_w), dtype=dtype, device=device
            )
            values = torch.zeros(
                (batch_size, 3, out_t, out_h, out_w), dtype=dtype, device=device
            )

            # Process each tile
            for h, h_, w, w_ in tqdm(
                tasks, desc="Decoding", total=len(tasks), unit="tile"
            ):
                # Extract tile
                tile_latent = latents[:, :, :, h:h_, w:w_]

                # Decode tile
                tile_latent = un_normalize_latents(
                    tile_latent, vae, vae_per_channel_normalize
                )

                decode_kwargs = {"return_dict": False}
                if timestep is not None:
                    decode_kwargs["timestep"] = timestep
                if isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)):
                    decode_kwargs["target_shape"] = (
                        1,
                        3,
                        tile_latent.shape[2] * temporal_scale if is_video else 1,
                        tile_latent.shape[3] * spatial_scale,
                        tile_latent.shape[4] * spatial_scale,
                    )

                tile_decoded = vae.decode(tile_latent.to(vae.dtype), **decode_kwargs)[0]
                tile_decoded = tile_decoded.to(dtype).to(device)

                # Create mask for blending
                mask = build_mask(
                    tile_decoded,
                    is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                    border_width=(
                        (size_h - stride_h) * spatial_scale,
                        (size_w - stride_w) * spatial_scale,
                    ),
                ).to(dtype=dtype, device=device)

                # Add to accumulation
                target_h = h * spatial_scale
                target_w = w * spatial_scale
                tile_h, tile_w = tile_decoded.shape[-2:]

                values[
                    :, :, :, target_h : target_h + tile_h, target_w : target_w + tile_w
                ] += (tile_decoded * mask)
                weight[
                    :, :, :, target_h : target_h + tile_h, target_w : target_w + tile_w
                ] += mask

            # Normalize by weights
            decoded = values / weight
            return decoded
        else:
            # Image VAE - process frame by frame
            latents = rearrange(latents, "b c t h w -> (b t) c h w")
            decoded = tiled_decode(
                latents,
                vae,
                tile_size,
                tile_stride,
                is_video,
                vae_per_channel_normalize,
                timestep,
            )
            # Reshape back to video format
            decoded = rearrange(decoded, "(b t) c h w -> b c t h w", b=batch_size)
            return decoded
    else:
        # Image case [B, C, H, W]
        batch_size, channels, height, width = latents.shape
        H, W = height, width
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        size_h = min(size_h, H)
        size_w = min(size_w, W)
        stride_h = min(stride_h, size_h)
        stride_w = min(stride_w, size_w)

        # Create tiles
        tasks = sliding_2d_windows(
            height=H,
            width=W,
            tile_size=(size_w, size_h),
            tile_stride=(stride_w, stride_h),
        )

        # Get output dimensions
        _, spatial_scale, _ = get_vae_size_scale_factor(vae)
        out_h = H * spatial_scale
        out_w = W * spatial_scale

        # Initialize accumulation tensors
        weight = torch.zeros((1, 1, out_h, out_w), dtype=dtype, device=device)
        values = torch.zeros((batch_size, 3, out_h, out_w), dtype=dtype, device=device)

        # Process each tile
        for h, h_, w, w_ in tasks:
            # Extract tile
            tile_latent = latents[:, :, h:h_, w:w_]

            # Decode tile
            tile_latent = un_normalize_latents(
                tile_latent, vae, vae_per_channel_normalize
            )
            tile_decoded = vae.decode(tile_latent.to(vae.dtype), return_dict=False)[0]
            tile_decoded = tile_decoded.to(dtype).to(device)

            # Create mask for blending
            mask = build_mask(
                tile_decoded,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) * spatial_scale,
                    (size_w - stride_w) * spatial_scale,
                ),
            ).to(dtype=dtype, device=device)

            # Add to accumulation
            target_h = h * spatial_scale
            target_w = w * spatial_scale
            tile_h, tile_w = tile_decoded.shape[-2:]

            values[
                :, :, target_h : target_h + tile_h, target_w : target_w + tile_w
            ] += (tile_decoded * mask)
            weight[
                :, :, target_h : target_h + tile_h, target_w : target_w + tile_w
            ] += mask

        # Normalize by weights
        decoded = values / weight
        return decoded


def vae_encode(
    media_items: Tensor,
    vae: AutoencoderKL,
    split_size: int = 1,
    vae_per_channel_normalize=False,
    use_tiling: bool = False,
    tile_size: tuple[int, int] = (512, 512),
    tile_stride: tuple[int, int] = (256, 256),
) -> Tensor:
    """
    Encodes media items (images or videos) into latent representations using a specified VAE model.
    The function supports processing batches of images or video frames and can handle the processing
    in smaller sub-batches if needed.

    Args:
        media_items (Tensor): A torch Tensor containing the media items to encode. The expected
            shape is (batch_size, channels, height, width) for images or (batch_size, channels,
            frames, height, width) for videos.
        vae (AutoencoderKL): An instance of the `AutoencoderKL` class from the `diffusers` library,
            pre-configured and loaded with the appropriate model weights.
        split_size (int, optional): The number of sub-batches to split the input batch into for encoding.
            If set to more than 1, the input media items are processed in smaller batches according to
            this value. Defaults to 1, which processes all items in a single batch.
        use_tiling (bool, optional): Whether to use tiled encoding for memory efficiency. Defaults to False.
        tile_size (tuple[int, int], optional): Size of tiles for tiled encoding in pixels. Defaults to (512, 512).
        tile_stride (tuple[int, int], optional): Stride between tiles for tiled encoding in pixels. Defaults to (256, 256).

    Returns:
        Tensor: A torch Tensor of the encoded latent representations. The shape of the tensor is adjusted
            to match the input shape, scaled by the model's configuration.

    Examples:
        >>> import torch
        >>> from diffusers import AutoencoderKL
        >>> vae = AutoencoderKL.from_pretrained('your-model-name')
        >>> images = torch.rand(10, 3, 8 256, 256)  # Example tensor with 10 videos of 8 frames.
        >>> latents = vae_encode(images, vae)
        >>> print(latents.shape)  # Output shape will depend on the model's latent configuration.

    Note:
        In case of a video, the function encodes the media item frame-by frame.
    """
    is_video_shaped = media_items.dim() == 5
    batch_size, channels = media_items.shape[0:2]

    if channels != 3:
        raise ValueError(f"Expects tensors with 3 channels, got {channels}.")

    if use_tiling:
        latents = tiled_encode(
            media_items, vae, tile_size, tile_stride, vae_per_channel_normalize
        )
        return latents

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        media_items = rearrange(media_items, "b c n h w -> (b n) c h w")
    if split_size > 1:
        if len(media_items) % split_size != 0:
            raise ValueError(
                "Error: The batch size must be divisible by 'train.vae_bs_split"
            )
        encode_bs = len(media_items) // split_size
        # latents = [vae.encode(image_batch).latent_dist.sample() for image_batch in media_items.split(encode_bs)]
        latents = []
        if media_items.device.type == "xla":
            xm.mark_step()
        for image_batch in media_items.split(encode_bs):
            latents.append(vae.encode(image_batch).latent_dist.sample())
            if media_items.device.type == "xla":
                xm.mark_step()
        latents = torch.cat(latents, dim=0)
    else:
        latents = vae.encode(media_items).latent_dist.sample()

    latents = normalize_latents(latents, vae, vae_per_channel_normalize)
    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        latents = rearrange(latents, "(b n) c h w -> b c n h w", b=batch_size)
    return latents


def vae_decode(
    latents: Tensor,
    vae: AutoencoderKL,
    is_video: bool = True,
    split_size: int = 1,
    vae_per_channel_normalize=False,
    timestep=None,
    use_tiling: bool = False,
    tile_size: tuple[int, int] = (64, 64),
    tile_stride: tuple[int, int] = (32, 32),
) -> Tensor:
    is_video_shaped = latents.dim() == 5
    batch_size = latents.shape[0]

    if use_tiling:
        images = tiled_decode(
            latents,
            vae,
            tile_size,
            tile_stride,
            is_video,
            vae_per_channel_normalize,
            timestep,
        )
        return images

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        latents = rearrange(latents, "b c n h w -> (b n) c h w")
    if split_size > 1:
        if len(latents) % split_size != 0:
            raise ValueError(
                "Error: The batch size must be divisible by 'train.vae_bs_split"
            )
        encode_bs = len(latents) // split_size
        image_batch = [
            _run_decoder(
                latent_batch, vae, is_video, vae_per_channel_normalize, timestep
            )
            for latent_batch in latents.split(encode_bs)
        ]
        images = torch.cat(image_batch, dim=0)
    else:
        images = _run_decoder(
            latents, vae, is_video, vae_per_channel_normalize, timestep
        )

    if is_video_shaped and not isinstance(
        vae, (VideoAutoencoder, CausalVideoAutoencoder)
    ):
        images = rearrange(images, "(b n) c h w -> b c n h w", b=batch_size)
    return images


def _run_decoder(
    latents: Tensor,
    vae: AutoencoderKL,
    is_video: bool,
    vae_per_channel_normalize=False,
    timestep=None,
) -> Tensor:
    if isinstance(vae, (VideoAutoencoder, CausalVideoAutoencoder)):
        *_, fl, hl, wl = latents.shape
        temporal_scale, spatial_scale, _ = get_vae_size_scale_factor(vae)
        latents = latents.to(vae.dtype)
        vae_decode_kwargs = {}
        if timestep is not None:
            vae_decode_kwargs["timestep"] = timestep
        image = vae.decode(
            un_normalize_latents(latents, vae, vae_per_channel_normalize),
            return_dict=False,
            target_shape=(
                1,
                3,
                fl * temporal_scale if is_video else 1,
                hl * spatial_scale,
                wl * spatial_scale,
            ),
            **vae_decode_kwargs,
        )[0]
    else:
        image = vae.decode(
            un_normalize_latents(latents, vae, vae_per_channel_normalize),
            return_dict=False,
        )[0]
    return image


def get_vae_size_scale_factor(vae: AutoencoderKL) -> Tuple[int, int, int]:
    if isinstance(vae, CausalVideoAutoencoder):
        spatial = vae.spatial_downscale_factor
        temporal = vae.temporal_downscale_factor
    else:
        down_blocks = len(
            [
                block
                for block in vae.encoder.down_blocks
                if isinstance(block.downsample, Downsample3D)
            ]
        )
        spatial = vae.config.patch_size * 2**down_blocks
        temporal = (
            vae.config.patch_size_t * 2**down_blocks
            if isinstance(vae, VideoAutoencoder)
            else 1
        )

    return (temporal, spatial, spatial)


def latent_to_pixel_coords(
    latent_coords: Tensor, vae: AutoencoderKL, causal_fix: bool = False
) -> Tensor:
    """
    Converts latent coordinates to pixel coordinates by scaling them according to the VAE's
    configuration.

    Args:
        latent_coords (Tensor): A tensor of shape [batch_size, 3, num_latents]
        containing the latent corner coordinates of each token.
        vae (AutoencoderKL): The VAE model
        causal_fix (bool): Whether to take into account the different temporal scale
            of the first frame. Default = False for backwards compatibility.
    Returns:
        Tensor: A tensor of pixel coordinates corresponding to the input latent coordinates.
    """

    scale_factors = get_vae_size_scale_factor(vae)
    causal_fix = isinstance(vae, CausalVideoAutoencoder) and causal_fix
    pixel_coords = latent_to_pixel_coords_from_factors(
        latent_coords, scale_factors, causal_fix
    )
    return pixel_coords


def latent_to_pixel_coords_from_factors(
    latent_coords: Tensor, scale_factors: Tuple, causal_fix: bool = False
) -> Tensor:
    pixel_coords = (
        latent_coords
        * torch.tensor(scale_factors, device=latent_coords.device)[None, :, None]
    )
    if causal_fix:
        # Fix temporal scale for first frame to 1 due to causality
        pixel_coords[:, 0] = (pixel_coords[:, 0] + 1 - scale_factors[0]).clamp(min=0)
    return pixel_coords


def normalize_latents(
    latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
) -> Tensor:
    return (
        (latents - vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1))
        / vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        if vae_per_channel_normalize
        else latents * vae.config.scaling_factor
    )


def un_normalize_latents(
    latents: Tensor, vae: AutoencoderKL, vae_per_channel_normalize: bool = False
) -> Tensor:
    return (
        latents * vae.std_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        + vae.mean_of_means.to(latents.dtype).view(1, -1, 1, 1, 1)
        if vae_per_channel_normalize
        else latents / vae.config.scaling_factor
    )
