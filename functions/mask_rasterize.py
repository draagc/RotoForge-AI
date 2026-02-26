import bpy
import numpy as np


def ensure_scene(context):
    original_scene = context.scene

    SCENE_NAME = ".RotoForge_MaskBakeScene"
    COMP_GROUP_NAME = ".RotoForge_CompositingGroup"

    if SCENE_NAME in bpy.data.scenes:
        new_scene = bpy.data.scenes.get(SCENE_NAME)
    else:
        new_scene = bpy.data.scenes.new(SCENE_NAME)

    if bpy.app.version >= (5, 0, 0):
        if new_scene.compositing_node_group is None:
            new_scene.compositing_node_group = bpy.data.node_groups.new(COMP_GROUP_NAME, 'CompositorNodeTree')
        node_tree = new_scene.compositing_node_group
    else:
        new_scene.use_nodes = True
        node_tree = new_scene.node_tree

    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    mask_node = nodes.new(type="CompositorNodeMask")
    viewer_node = nodes.new(type="CompositorNodeViewer")
    links.new(mask_node.outputs.get("Mask"), viewer_node.inputs.get("Image"))

    return original_scene, new_scene, mask_node


def rasterize_active_mask():
    context = bpy.context
    space = context.space_data
    mask = space.mask

    original_scene, new_scene, mask_node = ensure_scene(context)
    mask_node.mask = mask

    new_scene.frame_current = original_scene.frame_current

    width, height = space.image.size
    new_scene.render.resolution_x = width
    new_scene.render.resolution_y = height

    context.window.scene = new_scene
    context.window.scene = original_scene

    rasterized_img = np.zeros((height, width), dtype=np.float32)

    prev_layer_settings = dict()
    for other_layers in mask.layers:
        prev_layer_settings[other_layers.name] = (other_layers.hide_render,
                                                  other_layers.blend,
                                                  other_layers.alpha,
                                                  other_layers.invert)
        other_layers.hide_render = True
        other_layers.blend = 'ADD'
        other_layers.alpha = 1.0
        other_layers.invert = False

    for layer in mask.layers:
        if prev_layer_settings[layer.name][0]:
            continue

        rf_props = mask.rotoforge_maskgencontrols.get(layer.name)
        image_name = f"{mask.name}/MaskLayers/{layer.name}"

        if rf_props.is_rflayer and image_name in bpy.data.images:
            render_result = bpy.data.images.get(image_name)

            current_image = space.image
            space.image = render_result
            space.display_channels = space.display_channels
            space.image = current_image
        else:
            layer.hide_render = False

            bpy.ops.render.render(scene=new_scene.name)

            render_result = bpy.data.images.get("Viewer Node")

            layer.hide_render = True

        len_pixels = len(render_result.pixels)
        if len_pixels < 1:
            continue
        pixels = np.zeros(len_pixels, dtype=np.float32)
        render_result.pixels.foreach_get(pixels)

        rgba = pixels.reshape((height, width, 4))
        rasterized_layer = rgba[:, :, 0]

        hide, blend, alpha, invert = prev_layer_settings[layer.name]

        if blend != 'REPLACE':
            rasterized_layer *= alpha
        if invert:
            rasterized_layer = 1.0 - rasterized_layer

        match blend:
            case 'MERGE_ADD':
                rasterized_img = 1 - (1 - rasterized_img) * (1 - rasterized_layer)
            case 'MERGE_SUBTRACT':
                rasterized_img = np.clip(1 - (1 - rasterized_img) * (1 - rasterized_layer) - rasterized_layer, 0, 1)
            case 'ADD':
                rasterized_img = np.clip(rasterized_img + rasterized_layer, 0, 1)
            case 'SUBTRACT':
                rasterized_img = np.clip(rasterized_img - rasterized_layer, 0, 1)
            case 'LIGHTEN':
                rasterized_img = np.maximum(rasterized_img, rasterized_layer)
            case 'DARKEN':
                rasterized_img = np.minimum(rasterized_img, rasterized_layer)
            case 'MUL':
                rasterized_img *= rasterized_layer
            case 'REPLACE':
                rasterized_img = rasterized_layer * alpha + rasterized_img * (1 - alpha)
            case 'DIFFERENCE':
                rasterized_img = np.absolute(rasterized_img - rasterized_layer)

    for other_layers in mask.layers:
        other_layers.hide_render, other_layers.blend, other_layers.alpha, other_layers.invert = prev_layer_settings[other_layers.name]

    return rasterized_img * 255


def rasterize_layer_of_active_mask(
    layer,
    resolution,
    rf_allowed=False,
    hide_uncyclic=True,
    use_255_range=False,
):
    context = bpy.context
    space = context.space_data
    mask = space.mask

    original_scene, new_scene, mask_node = ensure_scene(context)
    mask_node.mask = mask

    new_scene.frame_current = original_scene.frame_current

    new_scene.render.resolution_x = resolution[0]
    new_scene.render.resolution_y = resolution[1]

    prev_layer_settings = dict()
    for other_layers in mask.layers:
        prev_layer_settings[other_layers.name] = (other_layers.hide_render,
                                                  other_layers.blend,
                                                  other_layers.alpha,
                                                  other_layers.invert)
        other_layers.hide_render = True
        other_layers.blend = 'ADD'
        other_layers.alpha = 1.0
        other_layers.invert = False

    if hide_uncyclic:
        prev_spline_settings = dict()
        for i, spline in layer.splines.items():
            if spline.use_cyclic == False:
                arr = np.zeros((3, 2 * len(spline.points)), dtype=np.float32)
                spline.points.foreach_get('co', arr[0])
                spline.points.foreach_get('handle_left', arr[1])
                spline.points.foreach_get('handle_right', arr[2])
                prev_spline_settings[i] = arr
                arr = np.ones(2 * len(spline.points), dtype=np.float32) * 100
                spline.points.foreach_set('co', arr)
                spline.points.foreach_set('handle_left', arr)
                spline.points.foreach_set('handle_right', arr)

    context.window.scene = new_scene
    context.window.scene = original_scene

    rf_props = mask.rotoforge_maskgencontrols.get(layer.name)
    image_name = f"{mask.name}/MaskLayers/{layer.name}"

    if rf_allowed and rf_props.is_rflayer and image_name in bpy.data.images:
        render_result = bpy.data.images.get(image_name)

        current_image = space.image
        space.image = render_result
        space.display_channels = space.display_channels
        space.image = current_image
    else:
        layer.hide_render = False

        bpy.ops.render.render(scene=new_scene.name)

        render_result = bpy.data.images.get("Viewer Node")

    len_pixels = len(render_result.pixels)
    if len_pixels < 1:
        rasterized_layer = np.zeros((resolution[1], resolution[0]), dtype=np.float32)
    else:
        pixels = np.zeros(len_pixels, dtype=np.float32)
        render_result.pixels.foreach_get(pixels)
        rgba = pixels.reshape((resolution[1], resolution[0], 4))
        rasterized_layer = rgba[:, :, 0]

    for other_layers in mask.layers:
        other_layers.hide_render, other_layers.blend, other_layers.alpha, other_layers.invert = prev_layer_settings[other_layers.name]

    if hide_uncyclic:
        for i, spline in layer.splines.items():
            if spline.use_cyclic == False:
                arr = prev_spline_settings[i]
                spline.points.foreach_set('co', arr[0])
                spline.points.foreach_set('handle_left', arr[1])
                spline.points.foreach_set('handle_right', arr[2])

    if use_255_range:
        return rasterized_layer * 255
    else:
        return rasterized_layer
