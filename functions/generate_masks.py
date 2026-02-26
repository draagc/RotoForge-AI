"""
RotoForge AI - Mask Generation

Handles image pixel extraction, crop/uncrop logic, and delegates
actual SAM3 inference to the server via SAM3Client.

No torch or sam3 imports — runs entirely in Blender's Python.
"""

import bpy
import os
import shutil
import numpy as np
import PIL.Image
import PIL.ImageFilter

from .prompt_utils import fake_logits, calculate_bounding_box
from .data_manager import save_sequential_mask, save_singular_mask, get_rotoforge_dir


def bpyimg_to_HWCuint8(source_image):
    """Convert a Blender image to HxWx4 uint8 numpy array."""
    source_pixels = np.zeros(len(source_image.pixels), dtype=np.float32)
    source_image.pixels.foreach_get(source_pixels)

    width = source_image.size[0]
    height = source_image.size[1]

    pixels_HWC_uint8 = (source_pixels.reshape(height, width, 4) * 255).astype(np.uint8)
    return pixels_HWC_uint8


def get_cropped_image(pixels_uint8_rgba, guide_mask, input_points, input_box):
    """Crop image and prompts to the bounding box region for efficiency."""
    cropping_radius = 0.05
    width = pixels_uint8_rgba.shape[1]
    height = pixels_uint8_rgba.shape[0]

    img = PIL.Image.fromarray(pixels_uint8_rgba)
    img = img.convert('RGB')

    if input_box is not None:
        mask = PIL.Image.fromarray(guide_mask) if guide_mask is not None else None
        cropping_box = input_box + np.array([
            -width * cropping_radius, -height * cropping_radius,
            width * cropping_radius, height * cropping_radius
        ])
        img = img.crop(cropping_box)
        if mask is not None:
            mask = mask.crop(cropping_box)
        if input_points is not None:
            input_points = input_points - [cropping_box[0], cropping_box[1]]
        input_box = np.array([
            width * cropping_radius, height * cropping_radius,
            input_box[2] - input_box[0] + width * cropping_radius,
            input_box[3] - input_box[1] + height * cropping_radius
        ])
    else:
        cropping_box = None

    pixels_uint8_rgb = np.asarray(img)
    return pixels_uint8_rgb, cropping_box, input_box, input_points


# ---------------------------------------------------------------------------
# Point/box prompt generation
# ---------------------------------------------------------------------------

def generate_mask(
    source_image,
    used_mask,
    client,
    guide_mask=None,
    guide_strength=10,
    blur_radius=0.2,
    input_points=None,
    input_labels=None,
    input_box=None,
):
    """Generate a single-frame mask using point/box prompts via the server."""

    print('loading image')
    pixels_uint8_rgba = bpyimg_to_HWCuint8(source_image)
    pixels_uint8_rgb, cropping_box, input_box, input_points = get_cropped_image(
        pixels_uint8_rgba, guide_mask, input_points, input_box
    )
    print('loaded image')

    print('predicting masks (point prompt)')
    best_mask, best_logits = _predict_and_select(
        client, pixels_uint8_rgb, guide_mask, guide_strength,
        input_points, input_labels, input_box,
    )
    print('predicted masks')

    if best_mask is None:
        print('No mask generated, skipping save')
        return

    print('saving mask')
    save_singular_mask(source_image, used_mask, best_mask, cropping_box, blur_radius)
    print('saved mask')


def track_mask(
    source_image,
    used_mask,
    client,
    guide_mask=None,
    guide_strength=10,
    blur_radius=0.2,
    search_radius=10,
    input_points=None,
    input_labels=None,
    input_box=None,
):
    """Track a mask across one frame using point/box prompts."""

    pixels_uint8_rgba = bpyimg_to_HWCuint8(source_image)
    pixels_uint8_rgb, cropping_box, input_box, input_points = get_cropped_image(
        pixels_uint8_rgba, guide_mask, input_points, input_box
    )

    best_mask, best_logits = _predict_and_select(
        client, pixels_uint8_rgb, guide_mask, guide_strength,
        input_points, input_labels, input_box,
    )

    if best_mask is None:
        print('No mask generated during tracking')
        return None, None, None, None

    overlay_l = save_sequential_mask(source_image, used_mask, best_mask, cropping_box, blur_radius)

    new_box = calculate_bounding_box(best_mask)
    if new_box is not None:
        new_box = np.array([
            new_box[0] - search_radius, new_box[1] - search_radius,
            new_box[2] + search_radius, new_box[3] + search_radius
        ])
        if cropping_box is not None:
            new_box = np.array([
                new_box[0] + cropping_box[0], new_box[1] + cropping_box[1],
                new_box[2] + cropping_box[0], new_box[3] + cropping_box[1]
            ])

    return best_mask, new_box, overlay_l, best_logits


def _predict_and_select(client, pixels_uint8_rgb, guide_mask, guide_strength,
                        input_points, input_labels, input_box):
    """Call the server's point-prompt endpoint and pick the best mask."""

    masks, scores, low_res = client.predict_points(
        image_rgb=pixels_uint8_rgb,
        input_points=input_points,
        input_labels=input_labels,
        input_box=input_box,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        return None, None

    best_score = float('-inf')
    best_mask = None
    best_logits = None
    cropped_area = pixels_uint8_rgb.size / 3

    if guide_mask is not None:
        sum_guide = float(np.sum(guide_mask))

    for i in range(len(scores)):
        current = float(scores[i])
        if guide_mask is not None:
            current += -abs(sum_guide - np.sum(masks[i])) / cropped_area * guide_strength
        if current > best_score:
            best_score = current
            best_mask = masks[i]
            best_logits = low_res[i] if low_res is not None else None

    return best_mask, best_logits


# ---------------------------------------------------------------------------
# Text prompt generation — multi-instance support
# ---------------------------------------------------------------------------

def _select_text_masks(masks, scores, multi_mode):
    """Apply multi-instance mode to text prompt results.

    Args:
        masks: NxHxW bool/uint8 array of detected instance masks.
        scores: N array of confidence scores.
        multi_mode: 'best', 'union', or 'separate'.

    Returns:
        list of masks (each HxW). For 'best'/'union' this is a single-element
        list; for 'separate' it's one mask per detection, sorted by score.
    """
    if masks is None or len(masks) == 0:
        return []

    if multi_mode == 'best':
        best_idx = int(np.argmax(scores))
        return [masks[best_idx].astype(np.uint8) * 255]

    if multi_mode == 'union':
        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            combined = np.maximum(combined, m.astype(np.uint8) * 255)
        return [combined]

    # 'separate' — return each mask individually, highest score first
    order = np.argsort(scores)[::-1]
    return [(masks[i].astype(np.uint8) * 255) for i in order]


def generate_mask_text(
    source_image,
    used_mask,
    client,
    text_prompt,
    blur_radius=0.2,
    confidence_threshold=0.5,
    multi_mode='union',
):
    """Generate mask(s) using a text prompt via the server.

    For 'best' and 'union' modes, writes one mask to used_mask.
    For 'separate' mode, returns a list of masks (caller creates layers).
    """

    print('loading image')
    pixels_uint8_rgba = bpyimg_to_HWCuint8(source_image)
    img = PIL.Image.fromarray(pixels_uint8_rgba).convert('RGB')
    pixels_uint8_rgb = np.asarray(img)
    print('loaded image')

    print(f'predicting masks (text: "{text_prompt}", mode: {multi_mode})')
    masks, boxes, scores = client.predict_text(
        image_rgb=pixels_uint8_rgb,
        prompt=text_prompt,
        confidence_threshold=confidence_threshold,
    )
    print(f'predicted {len(masks) if masks is not None else 0} instance(s)')

    selected = _select_text_masks(masks, scores, multi_mode)

    if not selected:
        print('No mask generated for text prompt, skipping save')
        return []

    if multi_mode == 'separate':
        # Return the masks — the caller (operator) creates separate layers
        return selected

    print('saving mask')
    save_singular_mask(source_image, used_mask, selected[0], None, blur_radius)
    print('saved mask')
    return selected


def track_mask_text(
    source_image,
    used_mask,
    client,
    text_prompt,
    blur_radius=0.2,
    search_radius=10,
    confidence_threshold=0.5,
    multi_mode='union',
):
    """Track a mask across one frame using a text prompt."""

    pixels_uint8_rgba = bpyimg_to_HWCuint8(source_image)
    img = PIL.Image.fromarray(pixels_uint8_rgba).convert('RGB')
    pixels_uint8_rgb = np.asarray(img)

    masks, boxes, scores = client.predict_text(
        image_rgb=pixels_uint8_rgb,
        prompt=text_prompt,
        confidence_threshold=confidence_threshold,
    )

    selected = _select_text_masks(masks, scores, multi_mode)

    if not selected:
        print('No mask generated during text tracking')
        return None, None, None

    # For tracking we always use a single combined mask (even in 'separate' mode,
    # union them for the tracking frame since we need one overlay / one bounding box)
    if len(selected) > 1:
        combined = np.zeros_like(selected[0], dtype=np.uint8)
        for m in selected:
            combined = np.maximum(combined, m)
        result_mask = combined
    else:
        result_mask = selected[0]

    overlay_l = save_sequential_mask(source_image, used_mask, result_mask, None, blur_radius)

    new_box = calculate_bounding_box(result_mask)
    if new_box is not None:
        new_box = np.array([
            new_box[0] - search_radius, new_box[1] - search_radius,
            new_box[2] + search_radius, new_box[3] + search_radius
        ])

    return result_mask, new_box, overlay_l


# ---------------------------------------------------------------------------
# Video tracking — SAM3 native temporal tracking
# ---------------------------------------------------------------------------

def export_frames_to_jpeg(source_image, frame_start, frame_end):
    """Export Blender image sequence frames to a temp JPEG directory.

    Must be called from the main thread (accesses bpy.context).
    Returns the path to the directory and the frame-to-index mapping.
    The video predictor expects JPEG files named 00000.jpg, 00001.jpg, etc.
    """
    frames_dir = os.path.join(get_rotoforge_dir(), "video_frames_tmp")
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    context = bpy.context
    space = context.space_data
    original_frame = context.scene.frame_current

    frame_to_idx = {}
    idx = 0

    iu = space.image_user
    for frame_num in range(frame_start, frame_end + 1):
        context.scene.frame_current = frame_num
        # Compute the image-sequence-relative frame from the scene frame
        iu.frame_current = frame_num - iu.frame_start + 1 + iu.frame_offset
        space.display_channels = space.display_channels

        pixels_rgba = bpyimg_to_HWCuint8(source_image)
        img = PIL.Image.fromarray(pixels_rgba).convert('RGB')

        filename = f"{idx:05d}.jpg"
        img.save(os.path.join(frames_dir, filename), quality=95)
        frame_to_idx[frame_num] = idx
        idx += 1

    context.scene.frame_current = original_frame
    print(f'Exported {idx} frames to {frames_dir}')
    return frames_dir, frame_to_idx


def _idx_to_frame(frame_to_idx):
    """Invert the frame_to_idx mapping."""
    return {v: k for k, v in frame_to_idx.items()}


def _save_propagated_masks(
    all_frames, idx_to_frame, used_mask,
    frame_end, scene_frame_end, blur_radius,
    status_callback=None,
):
    """Save propagated masks to disk — thread-safe (no bpy access)."""
    saved = 0
    total = len(all_frames)
    max_frame = max(frame_end, scene_frame_end)
    padding = len(str(max_frame))

    for video_idx, frame_data in sorted(all_frames.items()):
        if video_idx not in idx_to_frame:
            continue

        blender_frame = idx_to_frame[video_idx]
        masks = frame_data["masks"]

        if not masks:
            continue

        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for m in masks:
            combined = np.maximum(combined, m.astype(np.uint8) * 255)

        if blur_radius > 0:
            pil_mask = PIL.Image.fromarray(combined).convert('RGBA')
            pil_mask = pil_mask.filter(PIL.ImageFilter.BoxBlur(radius=blur_radius))
        else:
            pil_mask = PIL.Image.fromarray(combined).convert('RGBA')

        img_seq_dir = os.path.join(get_rotoforge_dir('masksequences'), used_mask)
        if not os.path.isdir(img_seq_dir):
            os.makedirs(img_seq_dir)

        frame_str = str(blender_frame).zfill(padding)
        flipped = pil_mask.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        flipped.save(os.path.join(img_seq_dir, f"{frame_str}.png"))
        saved += 1

        if status_callback:
            status_callback(f"Saving masks ({saved}/{total})...")

    print(f'Saved {saved} tracked frames')
    return saved


def server_track_video_text(
    client, local_frames_dir, frame_to_idx,
    used_mask, text_prompt,
    frame_start, frame_end, prompt_frame,
    blur_radius=0.2, confidence_threshold=0.5,
    direction="both", scene_frame_end=0,
    fill_hole_area=16,
    status_callback=None,
):
    """Server-side video text tracking — thread-safe (no bpy access).

    Call export_frames_to_jpeg() on the main thread first, then pass
    the results here. Safe to call from a background thread.
    """
    idx_to_frame = _idx_to_frame(frame_to_idx)

    if status_callback:
        status_callback("Loading model...")
    if not client.is_video_model_loaded():
        client.load_video_model()

    if status_callback:
        status_callback("Uploading frames...")
    if client._is_remote:
        server_frames_dir = client.video_upload_frames(local_frames_dir)
    else:
        server_frames_dir = local_frames_dir

    if status_callback:
        status_callback("Starting video session...")
    session_id = client.video_start_session(server_frames_dir)

    try:
        if prompt_frame not in frame_to_idx:
            prompt_frame = max(frame_start, min(prompt_frame, frame_end))
        prompt_idx = frame_to_idx[prompt_frame]
        print(f'Adding text prompt "{text_prompt}" on frame {prompt_frame} (idx {prompt_idx})')

        if status_callback:
            status_callback(f'Prompting: "{text_prompt}"...')
        result = client.video_add_prompt(
            session_id=session_id,
            frame_index=prompt_idx,
            text=text_prompt,
            confidence_threshold=confidence_threshold,
        )
        if not result["masks"]:
            print('No objects detected for text prompt on the initial frame')
            return 0

        n_obj = len(result["obj_ids"])
        print(f'Detected {n_obj} object(s), propagating {direction}...')
        if status_callback:
            status_callback(f"Propagating ({n_obj} objects)...")
        all_frames = client.video_propagate(session_id, direction=direction,
                                            fill_hole_area=fill_hole_area)

        return _save_propagated_masks(
            all_frames, idx_to_frame, used_mask,
            frame_end, scene_frame_end, blur_radius,
            status_callback=status_callback,
        )

    finally:
        client.video_close_session(session_id)
        if os.path.isdir(local_frames_dir):
            shutil.rmtree(local_frames_dir)


def track_video_text(
    source_image,
    used_mask,
    client,
    text_prompt,
    frame_start,
    frame_end,
    prompt_frame,
    blur_radius=0.2,
    confidence_threshold=0.5,
    direction="both",
    progress_callback=None,
):
    """Convenience wrapper — exports frames then tracks. Blocks the caller."""
    print(f'Exporting frames {frame_start}-{frame_end}...')
    local_frames_dir, frame_to_idx = export_frames_to_jpeg(
        source_image, frame_start, frame_end
    )
    return server_track_video_text(
        client=client,
        local_frames_dir=local_frames_dir,
        frame_to_idx=frame_to_idx,
        used_mask=used_mask,
        text_prompt=text_prompt,
        frame_start=frame_start,
        frame_end=frame_end,
        prompt_frame=prompt_frame,
        blur_radius=blur_radius,
        confidence_threshold=confidence_threshold,
        direction=direction,
        scene_frame_end=bpy.context.scene.frame_end,
    )


def server_track_video_points(
    client, local_frames_dir, frame_to_idx,
    used_mask, image_size,
    frame_start, frame_end, prompt_frame,
    input_points=None, input_labels=None,
    blur_radius=0.2, direction="both",
    scene_frame_end=0, fill_hole_area=16,
    status_callback=None,
):
    """Server-side video point tracking — thread-safe (no bpy access).

    Call export_frames_to_jpeg() on the main thread first, then pass
    the results here. Safe to call from a background thread.
    """
    idx_to_frame = _idx_to_frame(frame_to_idx)

    if status_callback:
        status_callback("Loading model...")
    if not client.is_video_model_loaded():
        client.load_video_model()

    if status_callback:
        status_callback("Uploading frames...")
    if client._is_remote:
        server_frames_dir = client.video_upload_frames(local_frames_dir)
    else:
        server_frames_dir = local_frames_dir

    if status_callback:
        status_callback("Starting video session...")
    session_id = client.video_start_session(server_frames_dir)

    try:
        if prompt_frame not in frame_to_idx:
            prompt_frame = max(frame_start, min(prompt_frame, frame_end))
        prompt_idx = frame_to_idx[prompt_frame]
        width, height = image_size

        norm_points = None
        if input_points is not None:
            norm_points = [[float(p[0]) / width, float(p[1]) / height] for p in input_points]
        norm_labels = None
        if input_labels is not None:
            norm_labels = [int(l) for l in input_labels]

        print(f'Adding point prompt on frame {prompt_frame} (idx {prompt_idx})')
        if status_callback:
            status_callback("Prompting with points...")
        result = client.video_add_prompt(
            session_id=session_id,
            frame_index=prompt_idx,
            points=norm_points,
            labels=norm_labels,
        )
        if not result["masks"]:
            print('No objects detected for point prompt on the initial frame')
            return 0

        print(f'Propagating {direction}...')
        if status_callback:
            status_callback("Propagating...")
        all_frames = client.video_propagate(session_id, direction=direction,
                                            fill_hole_area=fill_hole_area)

        return _save_propagated_masks(
            all_frames, idx_to_frame, used_mask,
            frame_end, scene_frame_end, blur_radius,
            status_callback=status_callback,
        )

    finally:
        client.video_close_session(session_id)
        if os.path.isdir(local_frames_dir):
            shutil.rmtree(local_frames_dir)


def track_video_points(
    source_image,
    used_mask,
    client,
    frame_start,
    frame_end,
    prompt_frame,
    input_points=None,
    input_labels=None,
    blur_radius=0.2,
    direction="both",
    progress_callback=None,
):
    """Convenience wrapper — exports frames then tracks. Blocks the caller."""
    print(f'Exporting frames {frame_start}-{frame_end}...')
    local_frames_dir, frame_to_idx = export_frames_to_jpeg(
        source_image, frame_start, frame_end
    )
    return server_track_video_points(
        client=client,
        local_frames_dir=local_frames_dir,
        frame_to_idx=frame_to_idx,
        used_mask=used_mask,
        image_size=tuple(source_image.size),
        frame_start=frame_start,
        frame_end=frame_end,
        prompt_frame=prompt_frame,
        input_points=input_points,
        input_labels=input_labels,
        blur_radius=blur_radius,
        direction=direction,
        scene_frame_end=bpy.context.scene.frame_end,
    )
