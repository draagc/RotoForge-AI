import bpy

import numpy as np
import PIL.Image
import torch

from .prompt_utils import fake_logits, calculate_bounding_box
from .data_manager import save_sequential_mask, save_singular_mask


def get_device():
    """Detect and return the best available device (CUDA > MPS > CPU).

    Returns:
        tuple: (device_str, device_name) e.g. ("cuda", "CUDA acceleration")
    """
    if torch.cuda.is_available():
        return "cuda", "CUDA acceleration"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", "MPS acceleration (Apple Silicon)"
    else:
        return "cpu", "CPU"


def empty_cache():
    """Empty the memory cache for the current device."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_predictor(model_type=None):
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from .install_dependencies import get_install_folder
    except ImportError as e:
        print(f'Failed to import SAM3: {e}')
        print('SAM3 may not be properly installed or has known import issues.')
        print('Please try: pip install git+https://github.com/facebookresearch/sam3.git')
        print('See: https://github.com/facebookresearch/sam3/issues/225')
        raise e
    
    # Empty the memory cache before to clean up any mess that's been handed over
    empty_cache()

    # Debug info
    print("PyTorch version: ", torch.__version__)

    # Device selection: CUDA > MPS > CPU
    device, device_name = get_device()
    print(f"Using {device_name}")

    # Fetch predictor
    print('loading SAM3 model')
    sam_checkpoint = f"{get_install_folder('sam3_weights')}/sam3.0.pt"

    # Build SAM3 model - note that SAM3 doesn't use model_type in the same way
    # It loads the complete model checkpoint directly
    try:
        model = build_sam3_image_model()
        # Load checkpoint with device mapping
        checkpoint_dict = torch.load(sam_checkpoint, map_location=device)
        model.load_state_dict(checkpoint_dict)
        model.to(device=device)
        model.eval()
        
        # Create processor
        processor = Sam3Processor(model)
        
        print('loaded SAM3 processor')
        
        # Empty the memory cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return processor
        
    except Exception as e:
        print(f'Error loading SAM3 model: {e}')
        print('Make sure the model checkpoint is downloaded and accessible')
        raise e




def bpyimg_to_HWCuint8(source_image):
    # Get the image pixel data as a numpy array:
    source_pixels = np.zeros(len(source_image.pixels), dtype=np.float32)
    source_image.pixels.foreach_get(source_pixels)

    # Determine the dimensions of the image
    width = source_image.size[0]
    height = source_image.size[1]

    # Reshape the pixel data into HWC uint8 format
    channels = 4
    pixels_HWC_uint8 = (np.array(source_pixels).reshape(height, width, channels)* 255).astype(np.uint8)
    return pixels_HWC_uint8



def get_cropped_image(pixels_uint8_rgba, guide_mask, input_points, input_box, input_logits):
    # Determine the dimensions of the image
    cropping_radius = 0.05
    width = pixels_uint8_rgba.shape[1]
    height = pixels_uint8_rgba.shape[0]
    
    # Load data into PIL
    img = PIL.Image.fromarray(pixels_uint8_rgba)
    img = img.convert('RGB')
    
    # Crop to box if box is supported
    if input_box is not None:
        mask = PIL.Image.fromarray(guide_mask)
        cropping_box = input_box + np.array([-width*cropping_radius, -height*cropping_radius, width*cropping_radius, height*cropping_radius])
        img = img.crop(cropping_box)
        mask = mask.crop(cropping_box)
        if input_points is not None:
            input_points = input_points - [cropping_box[0], cropping_box[1]]
        input_box = np.array([width*cropping_radius, height*cropping_radius, input_box[2]-input_box[0] + width*cropping_radius, input_box[3]-input_box[1] + height*cropping_radius])
        
        if input_logits is not None:
            input_logits = np.array([input_logits])
        else:
            input_logits = fake_logits(mask)
    else:
        input_logits = None
        cropping_box = None
        
    
    pixels_uint8_rgb = np.asarray(img)

    return pixels_uint8_rgb, cropping_box, input_logits, input_box, input_points






def predict_mask(pixels_uint8_rgb, processor, guide_mask, guide_strength, input_points, input_labels, input_box, input_logits):
    from PIL import Image
    
    # Convert numpy array to PIL Image for SAM3 processor
    image = Image.fromarray(pixels_uint8_rgb)
    
    # Set image in processor
    inference_state = processor.set_image(image)
    
    # For SAM3 point/box prompting, we need to use the instance interactivity mode
    masks_list = []
    scores_list = []
    logits_list = []
    
    # Handle different prompt types
    if input_points is not None and len(input_points) > 0:
        # Point prompting - convert to format SAM3 expects
        point_coords = [[int(point[0]), int(point[1])] for point in input_points]
        point_labels = [int(label) if label is not None else 1 for label in (input_labels or [1] * len(point_coords))]
        
        # Use SAM3's point prompting
        output = processor.set_point_prompt(
            state=inference_state,
            point_coords=point_coords,
            point_labels=point_labels
        )
        
        if output and "masks" in output:
            masks_list.extend(output["masks"])
            scores_list.extend(output["scores"])
            logits_list.extend(output["logits"])
    
    # If we have masks from prompts, select the best one
    if masks_list:
        # Initialize variables outside the loop
        best_score = float('-inf')
        cropped_area = len(pixels_uint8_rgb.flatten())/3
        best_mask = None
        best_logits = None
        
        # Calculate sums outside the loop if they don't change
        if guide_mask is not None:
            sum_guide_mask = np.sum(guide_mask)
            
        for i, score in enumerate(scores_list):
            current_score = score
            if guide_mask is not None:
                current_score += -abs(sum_guide_mask - np.sum(masks_list[i])) / cropped_area * guide_strength
            if current_score > best_score:
                best_score = current_score
                best_mask = masks_list[i]
                best_logits = logits_list[i]
        
        # Empty the memory cache after using SAM3
        empty_cache()
        return best_mask, best_logits
    else:
        # No masks generated, return empty results
        empty_cache()
        return None, None





# Debug func for testing model input
def save_singular_logits(source_image, input_logits, sam_logits):
    
    # Create new image
    new_name = source_image.name
    if new_name.rfind('.') == -1:
        new_name = new_name + '_FAKElogits'
    else:
        new_name = new_name[:new_name.rfind('.')] + '_FAKElogits' + new_name[new_name.rfind('.'):]
    logits_image = bpy.data.images.new(new_name, width=256, height=256, is_data=True, alpha=False, float_buffer=True)
    print('Fake logits')
    print('Shape: ', str(input_logits.shape))
    print('Min: ', str(np.min(input_logits)))
    print('Max: ', str(np.max(input_logits)))
    # Convert Binary Mask to image data
    best_logits_flat = np.array(input_logits).flatten()
    logits_data_data = np.where(best_logits_flat[:, None], [1, 1, 1, 1], [0, 0, 0, 1])
    np_logits_data = np.array(logits_data_data, dtype=np.float32).flatten()
    # Write mask to image
    logits_image.pixels.foreach_set(np_logits_data)
    # Save the image
    logits_image.pack()
    logits_image.update()
    
    
    
    # Create new image
    new_name = source_image.name
    if new_name.rfind('.') == -1:
        new_name = new_name + '_SAMlogits'
    else:
        new_name = new_name[:new_name.rfind('.')] + '_SAMlogits' + new_name[new_name.rfind('.'):]
    logits_image = bpy.data.images.new(source_image.name + "_SAMlogits", width=256, height=256, is_data=True, alpha=False, float_buffer=True)
    
    print('Sam logits')
    print('Shape: ', str(sam_logits.shape))
    print('Min: ', str(np.min(sam_logits)))
    print('Max: ', str(np.max(sam_logits)))
    logits_data = (sam_logits-np.min(sam_logits))/np.max(sam_logits-np.min(sam_logits))
    logits_data =  np.expand_dims(logits_data, axis=2)  # Add an additional dimension
    logits_data = np.concatenate([logits_data]*3, axis=2)
    # Set the alpha channel to 1 for all pixels
    alpha_channel = np.ones_like(logits_data[:, :, :1])  # Set alpha channel to 1 (fully opaque)
    logits_data = np.concatenate([logits_data, alpha_channel], axis=2)  # Concatenate alpha channel
    np_logits_data = np.array(logits_data, dtype=np.float32).flatten()
    # Write mask to image
    logits_image.pixels.foreach_set(np_logits_data)
    # Save the image
    logits_image.pack()
    logits_image.update()










def generate_mask(
    source_image, 
    used_mask,
    processor, 
    guide_mask = None,
    guide_strength = 10,
    blur_radius = 0.2,
    input_points = None,
    input_labels = None,
    input_box = None,
    debug_logits = False,
):

    

    print('loading image')
    pixels_uint8_rgba = bpyimg_to_HWCuint8(source_image)
    pixels_uint8_rgb, cropping_box, input_logits, input_box, input_points = get_cropped_image(pixels_uint8_rgba, guide_mask, input_points, input_box, None)
    print('loaded image')

    print('predicting masks')
    best_mask, best_logits = predict_mask(pixels_uint8_rgb, processor, guide_mask, guide_strength, input_points, input_labels, input_box, input_logits)
    print('predicted masks')

    if best_mask is None:
        print('No mask generated, skipping save')
        return

    print('saving mask')
    save_singular_mask(source_image, used_mask, best_mask, cropping_box, blur_radius)
    print('saved mask')
    
    if debug_logits:
        print('saving logits')
        save_singular_logits(source_image, input_logits, best_logits)
        print('saved logits')
        







def track_mask(
    source_image, 
    used_mask,
    processor, 
    guide_mask = None,
    guide_strength = 10,
    blur_radius = 0.2,
    search_radius = 10,
    input_points = None,
    input_labels = None,
    input_box = None,
    input_logits = None
):
    
    #Process the frame
    pixels_uint8_rgba = bpyimg_to_HWCuint8(source_image)
    pixels_uint8_rgb, cropping_box, input_logits, input_box, input_points = get_cropped_image(pixels_uint8_rgba, guide_mask, input_points, input_box, input_logits)
    
    best_mask, best_logits = predict_mask(pixels_uint8_rgb, processor, guide_mask, guide_strength, input_points, input_labels, input_box, input_logits)
    
    if best_mask is None:
        print('No mask generated during tracking')
        return None, None, None, None
    
    overlay_l = save_sequential_mask(source_image, used_mask, best_mask, cropping_box, blur_radius)

    #Set input data for next frame
    input_box = calculate_bounding_box(best_mask)

    # Handle case where no bounding box is found (empty mask)
    if input_box is not None:
        input_box = np.array([input_box[0] - search_radius, input_box[1] - search_radius, input_box[2] + search_radius, input_box[3] + search_radius])
        if cropping_box is not None:
            input_box = np.array([input_box[0] + cropping_box[0], input_box[1] + cropping_box[1], input_box[2] + cropping_box[0], input_box[3] + cropping_box[1]])

    return best_mask, input_box, overlay_l, best_logits
