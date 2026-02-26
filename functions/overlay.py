import bpy
from bpy.app.handlers import persistent
import numpy as np
import os
import gpu
import gpu_extras.batch
from mathutils import Matrix


# ---------------------------------------------------------------------------
# Shader / batch — created once during register()
# ---------------------------------------------------------------------------

shader = None
batch = None


def _create_shader():
    vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
    vert_out.smooth('VEC2', "uvInterp")

    shader_info = gpu.types.GPUShaderCreateInfo()
    shader_info.push_constant('MAT4', "ModelViewProjectionMatrix")
    shader_info.push_constant('VEC4', "overlayColor")
    shader_info.sampler(0, 'FLOAT_2D', "image")
    shader_info.vertex_in(0, 'VEC2', "position")
    shader_info.vertex_in(1, 'VEC2', "uv")
    shader_info.vertex_out(vert_out)
    shader_info.fragment_out(0, 'VEC4', "FragColor")

    shader_info.vertex_source(
        "void main()"
        "{"
        "  uvInterp = uv;"
        "  gl_Position = ModelViewProjectionMatrix * vec4(position, 0.0, 1.0);"
        "}"
    )

    shader_info.fragment_source(
        "void main()"
        "{"
        "  ivec2 texSize = textureSize(image, 0);"
        "  vec2 texelSize = vec2(1.0) / vec2(texSize);"
        "  vec2 nearestUV = floor(uvInterp / texelSize) * texelSize + texelSize * 0.5;"
        "  vec4 texColor = texture(image, nearestUV);"
        "  FragColor = vec4(texColor.rgb * overlayColor.rgb, overlayColor.a * (1.0 - texColor.r));"
        "}"
    )

    created_shader = gpu.shader.create_from_info(shader_info)
    del vert_out
    del shader_info

    created_batch = gpu_extras.batch.batch_for_shader(
        created_shader, 'TRI_FAN',
        {
            "position": ((0, 0), (1, 0), (1, 1), (0, 1)),
            "uv": ((0, 0), (1, 0), (1, 1), (0, 1)),
        },
    )

    return created_shader, created_batch


# ---------------------------------------------------------------------------
# Deferred rasterization cache
#
# The draw callback never calls bpy.ops.render.render().
# Instead, a one-shot bpy.app.timers callback reads RF-layer images
# directly from bpy.data.images and stores the result here.
# The draw callback only reads from _cached_rgba.
# ---------------------------------------------------------------------------

_cached_rgba = None       # flat float32 RGBA array, ready for GPU texture
_cached_size = (0, 0)     # (width, height) matching _cached_rgba
_dirty = False            # True → timer will re-rasterize on next tick
_timer_scheduled = False  # prevents duplicate timer registrations


def invalidate_overlay_cache():
    """Mark the overlay as needing re-rasterization.

    Called by operators after mask data changes. Safe to call from any
    context — the actual work is deferred to a timer.
    """
    global _dirty
    _dirty = True
    _schedule_raster_timer()


def _schedule_raster_timer():
    """Register a one-shot timer if one isn't already pending."""
    global _timer_scheduled
    if _timer_scheduled:
        return
    _timer_scheduled = True
    bpy.app.timers.register(_raster_timer_callback, first_interval=0.0)


def _find_image_editor_space():
    """Find an IMAGE_EDITOR space in mask mode with a valid mask + image.

    Timer callbacks don't have space_data in bpy.context, so we search
    through all windows/areas manually.
    """
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != 'IMAGE_EDITOR':
                continue
            space = area.spaces.active
            if space and getattr(space, 'mask', None) and getattr(space, 'image', None):
                return space, area, window
    return None, None, None


def _find_frame_file(seq_dir, frame_num):
    """Return the path to *frame_num* inside *seq_dir*, or None if not found."""
    if not os.path.isdir(seq_dir):
        return None

    files = sorted(os.listdir(seq_dir))
    if not files:
        return None

    for f in files:
        name = os.path.splitext(f)[0]
        try:
            if int(name) == frame_num:
                return os.path.join(seq_dir, f)
        except ValueError:
            continue

    return None


def _read_rf_layer_pixels(mask, layer, scene):
    """Read an RF-layer mask frame directly from disk (no bpy.ops.render)."""
    rf_props = mask.rotoforge_maskgencontrols.get(layer.name)
    if rf_props is None or not rf_props.is_rflayer:
        return None

    used_mask = f"{mask.name}/MaskLayers/{layer.name}"
    seq_dir = os.path.join(bpy.app.tempdir, "RotoForge", "masksequences", used_mask)
    frame_path = _find_frame_file(seq_dir, scene.frame_current)
    if frame_path is None:
        return None

    temp_name = ".rotoforge_overlay_temp"
    if temp_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[temp_name])

    try:
        temp_img = bpy.data.images.load(frame_path)
    except Exception:
        return None
    temp_img.name = temp_name

    n = len(temp_img.pixels)
    if n < 4:
        bpy.data.images.remove(temp_img)
        return None

    buf = np.zeros(n, dtype=np.float32)
    temp_img.pixels.foreach_get(buf)
    img_h, img_w = temp_img.size[1], temp_img.size[0]
    bpy.data.images.remove(temp_img)

    return buf.reshape((img_h, img_w, 4))[:, :, 0]


def _blend_layer(dst, src, blend, alpha, invert):
    """Apply a single layer with blend mode onto *dst* (in-place)."""
    layer = src.copy()
    if blend != 'REPLACE':
        layer *= alpha
    if invert:
        layer = 1.0 - layer

    match blend:
        case 'MERGE_ADD':
            dst[:] = 1 - (1 - dst) * (1 - layer)
        case 'MERGE_SUBTRACT':
            dst[:] = np.clip(1 - (1 - dst) * (1 - layer) - layer, 0, 1)
        case 'ADD':
            dst[:] = np.clip(dst + layer, 0, 1)
        case 'SUBTRACT':
            dst[:] = np.clip(dst - layer, 0, 1)
        case 'LIGHTEN':
            np.maximum(dst, layer, out=dst)
        case 'DARKEN':
            np.minimum(dst, layer, out=dst)
        case 'MUL':
            dst *= layer
        case 'REPLACE':
            dst[:] = layer * alpha + dst * (1 - alpha)
        case 'DIFFERENCE':
            dst[:] = np.absolute(dst - layer)


def _raster_timer_callback():
    """Reads RF-layer frames from disk — never calls bpy.ops.render."""
    global _cached_rgba, _cached_size, _dirty, _timer_scheduled
    _timer_scheduled = False

    if not _dirty:
        return None

    _dirty = False

    try:
        space, area, window = _find_image_editor_space()
        if space is None:
            return None

        scene = window.scene
        overlay_controls = scene.rotoforge_overlaycontrols
        if not overlay_controls.active_overlay or space.mode != 'MASK':
            return None

        mask = space.mask
        only_active_layer = overlay_controls.only_active_layer
        width, height = space.image.size

        source_pixels_rgba = None

        # --- Path 1: Single active layer ------------------------------------
        if only_active_layer:
            layer_px = _read_rf_layer_pixels(mask, mask.layers.active, scene)
            if layer_px is not None:
                grey = layer_px.flatten()
                source_pixels_rgba = np.ones((grey.size, 4), dtype=np.float32)
                source_pixels_rgba[:, :3] = grey[:, None]
                source_pixels_rgba = source_pixels_rgba.flatten()

        # --- Path 2: Blend all RF layers from disk --------------------------
        if source_pixels_rgba is None and not only_active_layer:
            blended = None
            found_any = False

            for layer in mask.layers:
                if layer.hide_render:
                    continue
                layer_px = _read_rf_layer_pixels(mask, layer, scene)
                if layer_px is None:
                    continue
                if blended is None:
                    blended = np.zeros(layer_px.shape, dtype=np.float32)
                found_any = True
                _blend_layer(blended, layer_px,
                             layer.blend, layer.alpha, layer.invert)

            if found_any and blended is not None:
                grey = blended.flatten()
                source_pixels_rgba = np.ones((grey.size, 4), dtype=np.float32)
                source_pixels_rgba[:, :3] = grey[:, None]
                source_pixels_rgba = source_pixels_rgba.flatten()

        if source_pixels_rgba is None:
            _cached_rgba = None
            _cached_size = None
            for win in bpy.context.window_manager.windows:
                for a in win.screen.areas:
                    if a.type == 'IMAGE_EDITOR':
                        a.tag_redraw()
            return None

        _cached_rgba = source_pixels_rgba
        _cached_size = (width, height)

        for win in bpy.context.window_manager.windows:
            for a in win.screen.areas:
                if a.type == 'IMAGE_EDITOR':
                    a.tag_redraw()

    except Exception as e:
        print(f"RotoForge AI: Overlay rasterization error: {e}")
        import traceback
        traceback.print_exc()

    return None


# ---------------------------------------------------------------------------
# GPU draw callback — lightweight, never rasterizes
# ---------------------------------------------------------------------------

def rotoforge_overlay_shader():
    context = bpy.context
    space = context.space_data
    if space is None or space.image is None:
        return

    mask = space.mask
    if mask is None:
        return

    overlay_controls = context.scene.rotoforge_overlaycontrols

    color = overlay_controls.overlay_color
    alpha = overlay_controls.overlay_opacity
    color = (color[0], color[1], color[2], alpha)

    custom_img = rotoforge_overlay_shader.custom_img

    if custom_img is not None:
        # Live tracking preview — build RGBA on the fly from the numpy array
        source_pixels = custom_img.flatten() / 255.0
        source_pixels_rgba = np.ones((source_pixels.size, 4), dtype=np.float32)
        source_pixels_rgba[:, :3] = source_pixels[:, None]
        rgba_flat = source_pixels_rgba.flatten()
        size = tuple(space.image.size)

    elif overlay_controls.active_overlay and space.mode == 'MASK':
        if _cached_rgba is None:
            # No cached data yet — schedule rasterization and skip this frame
            if not _dirty:
                invalidate_overlay_cache()
            return
        rgba_flat = _cached_rgba
        size = _cached_size
    else:
        return

    try:
        buf = gpu.types.Buffer('FLOAT', len(rgba_flat), rgba_flat)
        texture = gpu.types.GPUTexture(size, layers=0, is_cubemap=False,
                                       format='RGBA8', data=buf)
    except Exception:
        return

    region = context.region
    view2d = region.view2d

    translation = view2d.view_to_region(0, 0, clip=False)
    scale = (np.array(view2d.view_to_region(1, 1, clip=False))
             - np.array(view2d.view_to_region(0, 0, clip=False)))

    transform = np.eye(4, dtype=np.float32)
    transform[0, 3] = translation[0]
    transform[1, 3] = translation[1]
    transform[0, 0] = scale[0]
    transform[1, 1] = scale[1]

    gpu.matrix.load_matrix(Matrix(transform))

    shader.uniform_float("overlayColor", color)
    shader.uniform_sampler("image", texture)
    batch.draw(shader)


# Attribute used by operators for live tracking previews
rotoforge_overlay_shader.custom_img = None


# ---------------------------------------------------------------------------
# Frame change handler — marks dirty so the timer re-rasterizes
# ---------------------------------------------------------------------------

@persistent
def _on_frame_change(scene, depsgraph):
    global _dirty
    _dirty = True
    _schedule_raster_timer()


# ---------------------------------------------------------------------------
# Property group and panel
# ---------------------------------------------------------------------------

class OverlayControls(bpy.types.PropertyGroup):
    def _on_overlay_setting_changed(self, context):
        invalidate_overlay_cache()

    active_overlay: bpy.props.BoolProperty(
        name="Activate Overlay",
        default=False,
        update=_on_overlay_setting_changed,
    )  # type: ignore

    only_active_layer: bpy.props.BoolProperty(
        name="Only Render active layer",
        default=True,
        update=_on_overlay_setting_changed,
    )  # type: ignore

    use_combined: bpy.props.BoolProperty(
        name="Use baked Mask",
        default=True,
        update=_on_overlay_setting_changed,
    )  # type: ignore

    overlay_opacity: bpy.props.FloatProperty(
        name="Opacity",
        default=0,
        min=0.0, max=1.0,
        soft_min=0.0, soft_max=1.0,
    )  # type: ignore

    overlay_color: bpy.props.FloatVectorProperty(
        name="Overlay Color",
        subtype="COLOR",
        min=0, max=1,
        size=3,
        default=(1, 0, 0),
    )  # type: ignore

    @classmethod
    def register(cls):
        bpy.types.Scene.rotoforge_overlaycontrols = bpy.props.PointerProperty(type=cls)

    @classmethod
    def unregister(cls):
        if hasattr(bpy.types.Scene, 'rotoforge_overlaycontrols'):
            del bpy.types.Scene.rotoforge_overlaycontrols


def _any_mask_generated(context):
    """Check if any RF layer in the active mask has a generated image."""
    space = context.space_data
    mask = getattr(space, 'mask', None)
    if mask is None:
        return False
    for layer in mask.layers:
        image_name = f"{mask.name}/MaskLayers/{layer.name}"
        if image_name in bpy.data.images:
            return True
    return False


class OverlayPanel(bpy.types.Panel):
    """Overlay Panel"""
    bl_label = "Overlay"
    bl_idname = "ROTOFORGE_PT_OverlayPanel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "RotoForge"

    @classmethod
    def poll(cls, context):
        space_data = context.space_data
        return space_data.mask and space_data.mode == 'MASK'

    def draw_header_preset(self, context):
        layout = self.layout
        rotoforge_props = context.scene.rotoforge_overlaycontrols
        has_mask = _any_mask_generated(context)
        row = layout.row()
        row.enabled = has_mask
        row.prop(rotoforge_props, "active_overlay", text="Active", icon='OVERLAY')

    def draw(self, context):
        layout = self.layout
        rotoforge_props = context.scene.rotoforge_overlaycontrols

        if not _any_mask_generated(context):
            layout.label(text="No generated masks yet.", icon='INFO')
            return

        row = layout.row(align=True)
        row.prop(rotoforge_props, "only_active_layer")
        if not rotoforge_props.only_active_layer:
            row.prop(rotoforge_props, "use_combined")
        layout.template_color_picker(rotoforge_props, "overlay_color", value_slider=True)
        layout.prop(rotoforge_props, "overlay_opacity", text="Opacity", slider=True)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

overlay_handler = None

classes = [OverlayControls, OverlayPanel]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    global overlay_handler, shader, batch
    if overlay_handler is None:
        try:
            shader, batch = _create_shader()
            rotoforge_overlay_shader.custom_img = None
            overlay_handler = bpy.types.SpaceImageEditor.draw_handler_add(
                rotoforge_overlay_shader, (), 'WINDOW', 'POST_PIXEL')
        except Exception as e:
            print(f'RotoForge AI: Warning - Could not create overlay shader: {e}')
            print('RotoForge AI: Overlay functionality will be disabled')

    if _on_frame_change not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_on_frame_change)


def unregister():
    if _on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_on_frame_change)

    global overlay_handler
    if overlay_handler is not None:
        bpy.types.SpaceImageEditor.draw_handler_remove(overlay_handler, 'WINDOW')
        overlay_handler = None

    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
