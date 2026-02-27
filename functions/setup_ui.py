"""
RotoForge AI - UI Operators and Panels

Manages the SAM3 server lifecycle and exposes mask generation
operators for both point/box and text prompting.
"""

import bpy
import os
import threading
from time import process_time

from .install_dependencies import get_venv_python, get_server_script, get_install_folder

# Lazy imports — these modules need numpy/PIL which may not be installed yet.
# They're only actually used when an operator runs, not at registration time.
SAM3Client = None
generate_masks = None
prompt_utils = None
overlay = None
mask_rasterize = None
data_manager = None


def _ensure_modules():
    global SAM3Client, generate_masks, prompt_utils, overlay, mask_rasterize, data_manager
    if generate_masks is not None:
        return
    from .sam3_client import SAM3Client as _SC
    from . import generate_masks as _gm
    from . import prompt_utils as _pu
    from . import overlay as _ov
    from . import mask_rasterize as _mr
    from . import data_manager as _dm
    SAM3Client = _SC
    generate_masks = _gm
    prompt_utils = _pu
    overlay = _ov
    mask_rasterize = _mr
    data_manager = _dm

# ---------------------------------------------------------------------------
# Server / client singleton
# ---------------------------------------------------------------------------

_client = None  # SAM3Client instance, created on demand

# Tracking progress state — read by the panel to show loading UI
_tracking_busy = False
_tracking_status = ""


def _get_prefs():
    """Return the addon preferences."""
    return bpy.context.preferences.addons[__package__.rsplit('.', 1)[0]].preferences


def _get_client() -> SAM3Client:
    """Return the module-level SAM3Client, creating it if needed.

    Reads server_mode / server_host / server_port from addon preferences
    and recreates the client if the connection target has changed.
    """
    global _client
    _ensure_modules()
    prefs = _get_prefs()
    host = prefs.server_host if prefs.server_mode == "remote" else "127.0.0.1"
    port = prefs.server_port

    if _client is not None and (_client.host != host or _client.port != port):
        try:
            _client.stop_server()
        except Exception:
            pass
        _client = None

    if _client is None:
        _client = SAM3Client(host=host, port=port)
        _client._is_remote = (prefs.server_mode == "remote")
    return _client


def _ensure_server_and_model(context):
    """Make sure the server is running and the model is loaded.

    In local mode, launches the subprocess if needed.
    In remote mode, pings the remote server.

    Returns the client, or raises RuntimeError.
    """
    _ensure_modules()
    prefs = _get_prefs()
    client = _get_client()

    if not client.is_alive():
        if prefs.server_mode == "remote":
            client.connect_remote(timeout=10.0)
        else:
            python_exe = get_venv_python()
            server_script = get_server_script()

            if not os.path.isfile(python_exe):
                raise RuntimeError(
                    "SAM3 venv not found. Please install dependencies in the addon preferences."
                )

            checkpoint = os.path.join(get_install_folder(), "sam3.pt")
            if not os.path.isfile(checkpoint):
                checkpoint = None

            client.start_server(python_exe, server_script, checkpoint=checkpoint)

    if not client.is_model_loaded():
        client.load_model()

    return client


def free_server():
    """Free the model on the server (keep server running).

    Works for both local and remote servers — just frees GPU memory.
    """
    global _client
    if _client is None:
        return
    if _client.is_alive():
        try:
            _client.free_model()
        except Exception:
            pass


def stop_server():
    """Stop the server process entirely.

    For remote servers this only drops the client reference — it does
    NOT send /shutdown to a server we don't own.
    """
    global _client
    if _client is not None:
        _client.stop_server()
        _client = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def time_checkpoint(start, name):
    elapsed = process_time() - start
    minutes = int(elapsed // 60)
    seconds = round(elapsed % 60, 2)
    print(f"{name} finished in {minutes} min {seconds} sec")


# ---------------------------------------------------------------------------
# Operators — point/box prompt
# ---------------------------------------------------------------------------

class GenerateSingularMaskOperator(bpy.types.Operator):
    """Generates a singular .png mask using point/box prompts"""
    bl_idname = "rotoforge.generate_singular_mask"
    bl_label = "Generate Mask"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.space_data.image is not None

    def execute(self, context):
        space = context.space_data
        mask = space.mask
        layer = mask.layers.active
        image = space.image
        props = mask.rotoforge_maskgencontrols[layer.name]

        if len(layer.splines) == 0:
            self.report({'ERROR'}, 'No mask splines found! Please draw a mask first.')
            return {'CANCELLED'}

        try:
            client = _ensure_server_and_model(context)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        start = process_time()

        resolution = tuple(image.size)
        guide_mask = mask_rasterize.rasterize_layer_of_active_mask(layer, resolution)
        prompt_points, prompt_labels = prompt_utils.extract_prompt_points(mask, resolution)
        bounding_box = prompt_utils.calculate_bounding_box(guide_mask)

        if bounding_box is None and prompt_points is None:
            self.report({'ERROR'}, 'No valid mask input! The mask appears to be empty or invisible.')
            return {'CANCELLED'}

        used_mask = f"{mask.name}/MaskLayers/{layer.name}"

        generate_masks.generate_mask(
            source_image=image,
            used_mask=used_mask,
            client=client,
            guide_mask=guide_mask,
            guide_strength=props.guide_strength,
            blur_radius=props.feather_radius,
            input_points=prompt_points,
            input_labels=prompt_labels,
            input_box=bounding_box,
        )
        data_manager.update_maskseq(used_mask)
        overlay.invalidate_overlay_cache()

        self.report({'INFO'}, f'Saved mask layer as image: {used_mask}')
        time_checkpoint(start, 'Mask generation')
        return {'FINISHED'}

    def invoke(self, context, event):
        if context.space_data.image.source in ['SEQUENCE', 'MOVIE']:
            wm = context.window_manager
            return wm.invoke_confirm(self, event, title="This file is animated!",
                                     message="You're currently trying to create a static (not animated) mask based on animated footage. Do you wish to continue?",
                                     confirm_text="Process anyways", translate=True)
        return self.execute(context)


# ---------------------------------------------------------------------------
# Operators — text prompt
# ---------------------------------------------------------------------------

class GenerateTextMaskOperator(bpy.types.Operator):
    """Generates a mask using a text prompt (SAM3 concept segmentation)"""
    bl_idname = "rotoforge.generate_text_mask"
    bl_label = "Generate Text Mask"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.space_data.image is None:
            return False
        space = context.space_data
        if not (space.mask and space.mask.layers.active):
            return False
        props = space.mask.rotoforge_maskgencontrols[space.mask.layers.active.name]
        return bool(props.text_prompt.strip())

    def execute(self, context):
        space = context.space_data
        mask = space.mask
        layer = mask.layers.active
        image = space.image
        props = mask.rotoforge_maskgencontrols[layer.name]

        try:
            client = _ensure_server_and_model(context)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        start = process_time()
        multi_mode = props.text_multi_mode
        used_mask = f"{mask.name}/MaskLayers/{layer.name}"

        result = generate_masks.generate_mask_text(
            source_image=image,
            used_mask=used_mask,
            client=client,
            text_prompt=props.text_prompt.strip(),
            blur_radius=props.feather_radius,
            confidence_threshold=props.text_confidence,
            multi_mode=multi_mode,
        )

        if multi_mode == 'separate' and len(result) > 0:
            # Create a new layer for each detected instance
            for i, instance_mask in enumerate(result):
                layer_name = f"{layer.name}_{i+1}"
                # Create the mask layer if it doesn't exist
                if layer_name not in [l.name for l in mask.layers]:
                    bpy.ops.mask.layer_new()
                    mask.layers.active.name = layer_name

                instance_path = f"{mask.name}/MaskLayers/{layer_name}"
                from .data_manager import save_singular_mask as _save
                _save(image, instance_path, instance_mask, None, props.feather_radius)
                data_manager.update_maskseq(instance_path)

            self.report({'INFO'}, f'Created {len(result)} instance layers from text prompt')
        else:
            data_manager.update_maskseq(used_mask)
            count = len(result) if result else 0
            self.report({'INFO'}, f'Saved text-prompted mask ({count} instance(s)): {used_mask}')

        overlay.invalidate_overlay_cache()
        time_checkpoint(start, 'Text mask generation')
        return {'FINISHED'}

    def invoke(self, context, event):
        if context.space_data.image.source in ['SEQUENCE', 'MOVIE']:
            wm = context.window_manager
            return wm.invoke_confirm(self, event, title="This file is animated!",
                                     message="Creating a static mask on animated footage. Continue?",
                                     confirm_text="Process anyways", translate=True)
        return self.execute(context)


class TrackTextMaskOperator(bpy.types.Operator):
    """Tracks a mask using text prompts across frames"""
    bl_idname = "rotoforge.track_text_mask"
    bl_label = "Track Text Mask"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _next_processed_frame = None
    _used_mask_dir = None
    _running = False

    backwards: bpy.props.BoolProperty(name="Backwards", default=False)  # type: ignore

    @classmethod
    def poll(cls, context):
        if context.space_data.image is None:
            return False
        if context.space_data.image.source not in ['SEQUENCE', 'MOVIE']:
            return False
        space = context.space_data
        if not (space.mask and space.mask.layers.active):
            return False
        props = space.mask.rotoforge_maskgencontrols[space.mask.layers.active.name]
        return bool(props.text_prompt.strip())

    def modal(self, context, event):
        if event.type == 'TIMER':
            space = context.space_data
            mask = space.mask
            layer = mask.layers.active
            image = space.image
            props = mask.rotoforge_maskgencontrols[layer.name]

            context.scene.frame_current = self._next_processed_frame
            space.image_user.frame_current = self._next_processed_frame
            space.display_channels = space.display_channels

            print(f'----Info---- Frame: {self._next_processed_frame}')

            try:
                client = _ensure_server_and_model(context)
            except RuntimeError as e:
                self.report({'ERROR'}, str(e))
                self.cancel(context)
                return {'CANCELLED'}

            result = generate_masks.track_mask_text(
                source_image=image,
                used_mask=self._used_mask_dir,
                client=client,
                text_prompt=props.text_prompt.strip(),
                blur_radius=props.feather_radius,
                search_radius=props.search_radius,
                confidence_threshold=props.text_confidence,
                multi_mode=props.text_multi_mode,
            )

            if result[0] is None:
                self.report({'WARNING'}, f'Text tracking lost at frame {self._next_processed_frame}! Stopping.')
                self.cancel(context)
                return {'CANCELLED'}

            _, _, overlay_l = result
            overlay.rotoforge_overlay_shader.custom_img = overlay_l

            endframe = mask.frame_end if not self.backwards else mask.frame_start
            if self._next_processed_frame == endframe:
                self.cancel(context)
                return {'CANCELLED'}

            self._next_processed_frame += -1 if self.backwards else 1
            return {'PASS_THROUGH'}

        if event.type in ['ESC', 'RIGHTMOUSE']:
            self.cancel(context)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        if self._running:
            return {'CANCELLED'}

        space = context.space_data
        mask = space.mask
        layer = mask.layers.active

        try:
            _ensure_server_and_model(context)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        self._used_mask_dir = f"{mask.name}/MaskLayers/{layer.name}"
        self._next_processed_frame = context.scene.frame_current
        self._running = True
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        self._running = False
        overlay.rotoforge_overlay_shader.custom_img = None

        overlaycontrols = context.scene.rotoforge_overlaycontrols
        overlay_was_active = overlaycontrols.active_overlay
        overlaycontrols.active_overlay = False
        context.area.tag_redraw()

        space = context.space_data
        mask = space.mask

        mask_images_to_remove = [
            img.name for img in bpy.data.images
            if img.source == 'SEQUENCE' and f"{mask.name}/MaskLayers/" in img.name
        ]
        for img_name in mask_images_to_remove:
            if img_name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[img_name], do_unlink=True)

        maskseq_dir = data_manager.get_rotoforge_dir('masksequences')
        for layer in mask.layers:
            layer_image_name = f"{mask.name}/MaskLayers/{layer.name}"
            layer_dir = os.path.join(maskseq_dir, layer_image_name)
            if os.path.isdir(layer_dir):
                data_manager.update_maskseq(layer_image_name)

        overlaycontrols.active_overlay = overlay_was_active
        overlay.invalidate_overlay_cache()
        overlaycontrols.used_mask = self._used_mask_dir

        context.scene.frame_current = self._next_processed_frame
        self.report({'INFO'}, f'Saved text-tracked mask sequence: {self._used_mask_dir}')
        print("Quitting...")


# ---------------------------------------------------------------------------
# Operators — point/box tracking
# ---------------------------------------------------------------------------

class TrackMaskOperator(bpy.types.Operator):
    """Tracks a mask across frames using point/box prompts"""
    bl_idname = "rotoforge.track_mask"
    bl_label = "Track Mask"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _next_processed_frame = None
    _used_mask_dir = None
    _running = False

    guide_mask = None
    prompt_points, prompt_labels = None, None
    bounding_box = None

    backwards: bpy.props.BoolProperty(name="Backwards", default=False)  # type: ignore

    @classmethod
    def poll(cls, context):
        if context.space_data.image is None:
            return False
        return context.space_data.image.source in ['SEQUENCE', 'MOVIE']

    def modal(self, context, event):
        if event.type == 'TIMER':
            space = context.space_data
            mask = space.mask
            layer = mask.layers.active
            image = space.image
            props = mask.rotoforge_maskgencontrols[layer.name]

            context.scene.frame_current = self._next_processed_frame
            space.image_user.frame_current = self._next_processed_frame
            space.display_channels = space.display_channels

            print(f'----Info---- Frame: {self._next_processed_frame}')

            try:
                client = _ensure_server_and_model(context)
            except RuntimeError as e:
                self.report({'ERROR'}, str(e))
                self.cancel(context)
                return {'CANCELLED'}

            if not props.tracking and self.prompt_points is None:
                resolution = tuple(image.size)
                self.guide_mask = mask_rasterize.rasterize_layer_of_active_mask(layer, resolution)
                self.prompt_points, self.prompt_labels = prompt_utils.extract_prompt_points(mask, resolution)
                self.bounding_box = prompt_utils.calculate_bounding_box(self.guide_mask)

            self.guide_mask, self.bounding_box, overlay_l, _ = generate_masks.track_mask(
                source_image=image,
                used_mask=self._used_mask_dir,
                client=client,
                guide_mask=self.guide_mask,
                guide_strength=props.guide_strength,
                blur_radius=props.feather_radius,
                search_radius=props.search_radius,
                input_points=self.prompt_points,
                input_labels=self.prompt_labels,
                input_box=self.bounding_box,
            )

            if self.bounding_box is None:
                self.report({'WARNING'}, f'Tracking lost at frame {self._next_processed_frame}! Stopping.')
                self.cancel(context)
                return {'CANCELLED'}

            overlay.rotoforge_overlay_shader.custom_img = overlay_l
            self.prompt_points = None
            self.prompt_labels = None

            endframe = mask.frame_end if not self.backwards else mask.frame_start
            if self._next_processed_frame == endframe:
                self.cancel(context)
                return {'CANCELLED'}

            self._next_processed_frame += -1 if self.backwards else 1
            return {'PASS_THROUGH'}

        if event.type in ['ESC', 'RIGHTMOUSE']:
            self.cancel(context)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        if self._running:
            return {'CANCELLED'}

        space = context.space_data
        mask = space.mask
        layer = mask.layers.active
        image = space.image
        props = mask.rotoforge_maskgencontrols[layer.name]

        if len(layer.splines) == 0:
            self.report({'ERROR'}, 'No mask splines found! Please draw a mask first.')
            return {'CANCELLED'}

        try:
            client = _ensure_server_and_model(context)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        resolution = tuple(image.size)
        self.guide_mask = mask_rasterize.rasterize_layer_of_active_mask(layer, resolution)
        self.prompt_points, self.prompt_labels = prompt_utils.extract_prompt_points(mask, resolution)
        self.bounding_box = prompt_utils.calculate_bounding_box(self.guide_mask)

        if self.bounding_box is None and self.prompt_points is None:
            self.report({'ERROR'}, 'No valid mask input! The mask appears to be empty or invisible.')
            return {'CANCELLED'}

        self._used_mask_dir = f"{mask.name}/MaskLayers/{layer.name}"
        self._next_processed_frame = context.scene.frame_current
        self._running = True
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        self._running = False

        overlay.rotoforge_overlay_shader.custom_img = None

        overlaycontrols = context.scene.rotoforge_overlaycontrols
        overlay_was_active = overlaycontrols.active_overlay
        overlaycontrols.active_overlay = False
        context.area.tag_redraw()

        space = context.space_data
        mask = space.mask

        mask_images_to_remove = [
            img.name for img in bpy.data.images
            if img.source == 'SEQUENCE' and f"{mask.name}/MaskLayers/" in img.name
        ]
        for img_name in mask_images_to_remove:
            if img_name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[img_name], do_unlink=True)

        maskseq_dir = data_manager.get_rotoforge_dir('masksequences')
        for layer in mask.layers:
            layer_image_name = f"{mask.name}/MaskLayers/{layer.name}"
            layer_dir = os.path.join(maskseq_dir, layer_image_name)
            if os.path.isdir(layer_dir):
                data_manager.update_maskseq(layer_image_name)

        overlaycontrols.active_overlay = overlay_was_active
        overlay.invalidate_overlay_cache()
        overlaycontrols.used_mask = self._used_mask_dir

        context.scene.frame_current = self._next_processed_frame

        self.guide_mask = None
        self.prompt_points, self.prompt_labels = None, None
        self.bounding_box = None

        self.report({'INFO'}, f'Saved mask layer as image sequence: {self._used_mask_dir}')
        print("Quitting...")


# ---------------------------------------------------------------------------
# Operators — video tracking (SAM3 native temporal tracking)
# ---------------------------------------------------------------------------

class TrackVideoTextOperator(bpy.types.Operator):
    """Track with text prompt using SAM3 video predictor (temporal memory, all frames at once)"""
    bl_idname = "rotoforge.track_video_text"
    bl_label = "Video Track (Text)"
    bl_options = {'REGISTER'}

    _thread: threading.Thread = None
    _timer = None
    _saved: int = 0
    _error: str = ""
    _status: str = ""
    _used_mask: str = ""
    _start_time: float = 0

    @classmethod
    def poll(cls, context):
        if _tracking_busy:
            return False
        if context.space_data.image is None:
            return False
        if context.space_data.image.source not in ['SEQUENCE', 'MOVIE']:
            return False
        space = context.space_data
        if not (space.mask and space.mask.layers.active):
            return False
        props = space.mask.rotoforge_maskgencontrols[space.mask.layers.active.name]
        return bool(props.text_prompt.strip())

    def invoke(self, context, event):
        global _tracking_busy, _tracking_status
        space = context.space_data
        mask = space.mask
        layer = mask.layers.active
        image = space.image
        props = mask.rotoforge_maskgencontrols[layer.name]

        try:
            client = _ensure_server_and_model(context)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        self._start_time = process_time()
        self._used_mask = f"{mask.name}/MaskLayers/{layer.name}"
        self._error = ""
        self._saved = 0

        _tracking_busy = True
        _tracking_status = "Exporting frames..."
        self._status = _tracking_status

        def _export_progress(current, total):
            global _tracking_status
            msg = f"Exporting frames {current}/{total}..."
            self._status = msg
            _tracking_status = msg
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

        local_frames_dir, frame_to_idx, original_size = generate_masks.export_frames_to_jpeg(
            image, mask.frame_start, mask.frame_end,
            progress_callback=_export_progress,
        )

        scene_frame_end = context.scene.frame_end
        prompt_frame = context.scene.frame_current
        text_prompt = props.text_prompt.strip()
        blur_radius = props.feather_radius
        confidence_threshold = props.text_confidence
        fill_hole_area = props.fill_hole_area
        frame_start = mask.frame_start
        frame_end = mask.frame_end
        used_mask = self._used_mask

        def _status_cb(s):
            global _tracking_status
            self._status = s
            _tracking_status = s

        def _bg():
            try:
                self._saved = generate_masks.server_track_video_text(
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
                    direction="both",
                    scene_frame_end=scene_frame_end,
                    fill_hole_area=fill_hole_area,
                    original_size=original_size,
                    status_callback=_status_cb,
                )
            except Exception as e:
                self._error = str(e)
                import traceback
                traceback.print_exc()

        self._thread = threading.Thread(target=_bg, daemon=True)
        self._thread.start()

        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        global _tracking_busy, _tracking_status
        context.window_manager.event_timer_remove(self._timer)
        if context.area:
            context.area.header_text_set(None)
        _tracking_busy = False
        _tracking_status = ""

    def modal(self, context, event):
        if event.type == 'ESC':
            self._finish(context)
            self.report({'WARNING'}, "Tracking continues in background")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._thread.is_alive():
            if context.area:
                context.area.header_text_set(f"RotoForge: {self._status}")
                context.area.tag_redraw()
            return {'PASS_THROUGH'}

        self._finish(context)

        if self._error:
            self.report({'ERROR'}, self._error)
            return {'CANCELLED'}

        if self._used_mask in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[self._used_mask], do_unlink=True)
        data_manager.update_maskseq(self._used_mask)
        overlay.invalidate_overlay_cache()
        context.scene.rotoforge_overlaycontrols.used_mask = self._used_mask

        self.report({'INFO'}, f'Video tracked {self._saved} frames with text prompt: {self._used_mask}')
        time_checkpoint(self._start_time, 'Video text tracking')
        return {'FINISHED'}


class TrackVideoPointsOperator(bpy.types.Operator):
    """Track with point/box prompt using SAM3 video predictor (temporal memory, all frames at once)"""
    bl_idname = "rotoforge.track_video_points"
    bl_label = "Video Track (Points)"
    bl_options = {'REGISTER'}

    _thread: threading.Thread = None
    _timer = None
    _saved: int = 0
    _error: str = ""
    _status: str = ""
    _used_mask: str = ""
    _start_time: float = 0

    @classmethod
    def poll(cls, context):
        if _tracking_busy:
            return False
        if context.space_data.image is None:
            return False
        return context.space_data.image.source in ['SEQUENCE', 'MOVIE']

    def invoke(self, context, event):
        global _tracking_busy, _tracking_status
        space = context.space_data
        mask = space.mask
        layer = mask.layers.active
        image = space.image
        props = mask.rotoforge_maskgencontrols[layer.name]

        if len(layer.splines) == 0:
            self.report({'ERROR'}, 'No mask splines found! Please draw a mask first.')
            return {'CANCELLED'}

        try:
            client = _ensure_server_and_model(context)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        self._start_time = process_time()
        resolution = tuple(image.size)
        prompt_points, prompt_labels = prompt_utils.extract_prompt_points(mask, resolution)

        if prompt_points is None:
            self.report({'ERROR'}, 'No valid prompt points found.')
            return {'CANCELLED'}

        self._used_mask = f"{mask.name}/MaskLayers/{layer.name}"
        self._error = ""
        self._saved = 0

        _tracking_busy = True
        _tracking_status = "Exporting frames..."
        self._status = _tracking_status

        def _export_progress(current, total):
            global _tracking_status
            msg = f"Exporting frames {current}/{total}..."
            self._status = msg
            _tracking_status = msg
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

        local_frames_dir, frame_to_idx, original_size = generate_masks.export_frames_to_jpeg(
            image, mask.frame_start, mask.frame_end,
            progress_callback=_export_progress,
        )

        scene_frame_end = context.scene.frame_end
        prompt_frame = context.scene.frame_current
        blur_radius = props.feather_radius
        fill_hole_area = props.fill_hole_area
        frame_start = mask.frame_start
        frame_end = mask.frame_end
        image_size = resolution
        used_mask = self._used_mask

        def _status_cb(s):
            global _tracking_status
            self._status = s
            _tracking_status = s

        def _bg():
            try:
                self._saved = generate_masks.server_track_video_points(
                    client=client,
                    local_frames_dir=local_frames_dir,
                    frame_to_idx=frame_to_idx,
                    used_mask=used_mask,
                    image_size=image_size,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    prompt_frame=prompt_frame,
                    input_points=prompt_points,
                    input_labels=prompt_labels,
                    blur_radius=blur_radius,
                    direction="both",
                    scene_frame_end=scene_frame_end,
                    fill_hole_area=fill_hole_area,
                    original_size=original_size,
                    status_callback=_status_cb,
                )
            except Exception as e:
                self._error = str(e)
                import traceback
                traceback.print_exc()

        self._thread = threading.Thread(target=_bg, daemon=True)
        self._thread.start()

        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        global _tracking_busy, _tracking_status
        context.window_manager.event_timer_remove(self._timer)
        if context.area:
            context.area.header_text_set(None)
        _tracking_busy = False
        _tracking_status = ""

    def modal(self, context, event):
        if event.type == 'ESC':
            self._finish(context)
            self.report({'WARNING'}, "Tracking continues in background")
            return {'CANCELLED'}

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        if self._thread.is_alive():
            if context.area:
                context.area.header_text_set(f"RotoForge: {self._status}")
                context.area.tag_redraw()
            return {'PASS_THROUGH'}

        self._finish(context)

        if self._error:
            self.report({'ERROR'}, self._error)
            return {'CANCELLED'}

        if self._used_mask in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[self._used_mask], do_unlink=True)
        data_manager.update_maskseq(self._used_mask)
        overlay.invalidate_overlay_cache()
        context.scene.rotoforge_overlaycontrols.used_mask = self._used_mask

        self.report({'INFO'}, f'Video tracked {self._saved} frames with point prompt: {self._used_mask}')
        time_checkpoint(self._start_time, 'Video point tracking')
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators — merge / import / misc
# ---------------------------------------------------------------------------

class MergeMaskOperator(bpy.types.Operator):
    """Rasterizes all masks down to image"""
    bl_idname = "rotoforge.merge_mask"
    bl_label = "Bake Mask to Texture"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _next_processed_frame = None
    _used_mask_dir = None
    _running = False

    @classmethod
    def poll(cls, context):
        return context.space_data.image is not None

    def modal(self, context, event):
        if event.type == 'TIMER':
            space = context.space_data
            mask = space.mask
            image = space.image

            context.scene.frame_current = self._next_processed_frame
            space.image_user.frame_current = self._next_processed_frame

            print(f'----Info---- Frame: {self._next_processed_frame}')

            img = mask_rasterize.rasterize_active_mask()
            overlay.rotoforge_overlay_shader.custom_img = img
            data_manager.save_sequential_mask(image, self._used_mask_dir, img, None)

            if self._next_processed_frame == mask.frame_end:
                self.cancel(context)
                return {'CANCELLED'}

            self._next_processed_frame += 1
            return {'PASS_THROUGH'}

        if event.type in ['ESC', 'RIGHTMOUSE']:
            self.cancel(context)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        if self._running:
            return {'CANCELLED'}

        _ensure_modules()

        space = context.space_data
        mask = space.mask

        self._used_mask_dir = f"{mask.name}/Combined"
        self._next_processed_frame = mask.frame_start
        self._running = True
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        self._running = False

        overlay.rotoforge_overlay_shader.custom_img = None

        if self._used_mask_dir in bpy.data.images:
            img = bpy.data.images[self._used_mask_dir]
            bpy.data.images.remove(img, do_unlink=True)

        data_manager.update_maskseq(self._used_mask_dir)
        overlay.invalidate_overlay_cache()

        overlaycontrols = context.scene.rotoforge_overlaycontrols
        overlaycontrols.used_mask = self._used_mask_dir

        context.scene.frame_current = self._next_processed_frame
        self.report({'INFO'}, f'Saved combined mask as image sequence: {self._used_mask_dir}')
        print("Quitting...")


class ImportMaskNodeOperator(bpy.types.Operator):
    """Imports a Mask as a texture node"""
    bl_idname = "rotoforge.import_mask_node"
    bl_label = "Import Mask"
    bl_options = {'REGISTER', 'UNDO'}

    mouse_pos = (0, 0)

    @classmethod
    def poll(cls, context):
        used_mask = context.scene.rotoforge_importcontrols.used_mask
        return context.space_data.node_tree is not None and used_mask not in ('', 'NONE')

    def invoke(self, context, event):
        self.mouse_pos = (event.mouse_region_x, event.mouse_region_y)
        return self.execute(context)

    def execute(self, context):
        nodetree = context.space_data.node_tree
        import_props = context.scene.rotoforge_importcontrols
        region = context.region
        view2d = region.view2d

        def get_selected(nt):
            return any(node.select for node in nt.nodes)

        bpy.ops.node.select_all(action='DESELECT')
        while get_selected(nodetree) and nodetree.nodes.active and nodetree.nodes.active.type == 'GROUP':
            nodetree = nodetree.nodes.active.node_tree

        used_mask = bpy.data.masks[import_props.used_mask]
        used_mask_img = bpy.data.images[f"{import_props.used_mask}/Combined"]

        nodetree_type = nodetree.type
        match nodetree_type:
            case 'COMPOSITING':
                bpy.ops.node.add_node(type='CompositorNodeImage')
                node = nodetree.nodes.active
                node.image = used_mask_img
                node.use_auto_refresh = True
                node.frame_duration = used_mask.frame_end
            case 'SHADER':
                bpy.ops.node.add_node(type='ShaderNodeTexImage')
                node = nodetree.nodes.active
                node.image = used_mask_img
                node.image_user.use_auto_refresh = True
                node.image_user.frame_duration = used_mask.frame_end
            case 'GEOMETRY':
                bpy.ops.node.add_node(type='GeometryNodeImageTexture')
                node = nodetree.nodes.active
                node.inputs['Image'].default_value = used_mask_img
            case _:
                raise Exception("Unknown nodetree type: ", nodetree_type)

        ui_scale = context.preferences.system.ui_scale
        x, y = view2d.region_to_view(self.mouse_pos[0], self.mouse_pos[1])
        node.location = x / ui_scale, y / ui_scale
        bpy.ops.node.translate_attach_remove_on_cancel('INVOKE_DEFAULT')

        self.report({'INFO'}, f'Created new image node linked to mask: {import_props.used_mask}')
        return {'FINISHED'}


class MaskRangeToSceneOperator(bpy.types.Operator):
    """Set the mask range to the scene range"""
    bl_idname = "rotoforge.set_mask_range_to_scene"
    bl_label = "Set Scene Frames"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        mask = context.space_data.mask
        mask.frame_start = scene.frame_start
        mask.frame_end = scene.frame_end
        return {'FINISHED'}


class FreePredictorOperator(bpy.types.Operator):
    """Frees the model from GPU memory (keeps server running)"""
    bl_idname = "rotoforge.free_predictor"
    bl_label = "Free Cache"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        free_server()
        return {'FINISHED'}


class StopServerOperator(bpy.types.Operator):
    """Stop the SAM3 inference server"""
    bl_idname = "rotoforge.stop_server"
    bl_label = "Stop Server"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        stop_server()
        self.report({'INFO'}, 'SAM3 server stopped')
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

class LayerPanel(bpy.types.Panel):
    """Mask Layers"""
    bl_label = "Mask Layers"
    bl_idname = "ROTOFORGE_PT_LayerPanel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "RotoForge"

    @classmethod
    def poll(cls, context):
        space_data = context.space_data
        return space_data.mask and space_data.mode == 'MASK'

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        space_data = context.space_data
        mask = space_data.mask
        active_layer = mask.layers.active

        row = layout.split(factor=0.4)
        row.operator("rotoforge.set_mask_range_to_scene")
        sub = row.row(align=True)
        sub.use_property_split = False
        sub.prop(mask, "frame_start", text="Start")
        sub.prop(mask, "frame_end", text="End")
        layout.operator("rotoforge.merge_mask", icon='RENDER_RESULT')

        rows = 4 if active_layer else 1
        row = layout.row()
        row.template_list("MASK_UL_layers", "", mask, "layers", mask, "active_layer_index", rows=rows)

        sub = row.column(align=True)
        sub.operator("mask.layer_new", icon='ADD', text="")
        sub.operator("mask.layer_remove", icon='REMOVE', text="")

        if active_layer:
            rotoforge_props = mask.rotoforge_maskgencontrols[active_layer.name]
            sub.separator()
            sub.operator("mask.layer_move", icon='TRIA_UP', text="").direction = 'UP'
            sub.operator("mask.layer_move", icon='TRIA_DOWN', text="").direction = 'DOWN'

            row = layout.row(align=True)
            row.prop(active_layer, "alpha")
            row.prop(active_layer, "invert", text="", icon='IMAGE_ALPHA')
            layout.prop(active_layer, "blend")

            layout.prop(rotoforge_props, "is_rflayer")
            layout.separator()
            if not rotoforge_props.is_rflayer:
                layout.prop(active_layer, "falloff")
                col = layout.column()
                col.prop(active_layer, "use_fill_overlap", text="Overlap")
                col.prop(active_layer, "use_fill_holes", text="Holes")


class RotoForgeMaskPanel(bpy.types.Panel):
    """RotoForge Mask Panel"""
    bl_label = "RotoForge"
    bl_idname = "ROTOFORGE_PT_RotoForgeMaskPanel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "RotoForge"

    @classmethod
    def poll(cls, context):
        space_data = context.space_data
        if space_data.mask and space_data.mask.layers.active is not None and space_data.mode == 'MASK':
            mask = space_data.mask
            active_layer = mask.layers.active
            return mask.rotoforge_maskgencontrols[active_layer.name].is_rflayer
        return False

    def draw(self, context):
        layout = self.layout
        space_data = context.space_data
        mask = space_data.mask
        active_layer = mask.layers.active
        props = mask.rotoforge_maskgencontrols[active_layer.name]

        # Mask Settings
        settings_box = layout.box()
        settings_box.label(text="Mask Settings")
        settings_box.prop(props, "feather_radius")
        settings_box.prop(props, "fill_hole_area")
        layout.separator()

        # Text Prompt
        text_box = layout.box()
        text_box.label(text="Text Prompt")
        text_box.prop(props, "text_prompt", text="")
        row = text_box.row(align=True)
        row.prop(props, "text_confidence")
        row.prop(props, "text_multi_mode", text="")

        row = text_box.row(align=True)
        row.label(text="Static:")
        row = row.row(align=True)
        row.alignment = 'RIGHT'
        row.operator("rotoforge.generate_text_mask", text="Generate", icon='SORTALPHA')

        row = text_box.row(align=True)
        row.label(text="Animated:")
        row.scale_x = 2.0
        op = row.operator("rotoforge.track_text_mask", text="", icon='TRACKING_BACKWARDS')
        op.backwards = True
        op = row.operator("rotoforge.track_text_mask", text="", icon='TRACKING_FORWARDS')
        op.backwards = False

        if _tracking_busy:
            row = text_box.row(align=True)
            row.alert = True
            row.label(text=_tracking_status or "Tracking...", icon='SORTTIME')
            row.enabled = False
        else:
            text_box.operator("rotoforge.track_video_text", text="Video Track (All Frames)", icon='SEQUENCE')
        layout.separator()

        # Point/Box Prompt
        box = layout.box()
        box.label(text="Point/Box Prompt")

        row = box.row(align=True)
        row.label(text="Static:")
        row = row.row(align=True)
        row.alignment = 'RIGHT'
        row.operator("rotoforge.generate_singular_mask", text="Generate", icon='IMAGE_PLANE')

        row = box.row(align=True)
        row.label(text="Animated:")
        row.scale_x = 2.0
        op = box.operator("rotoforge.track_mask", text="", icon='TRACKING_BACKWARDS')
        op.backwards = True
        op = box.operator("rotoforge.track_mask", text="", icon='TRACKING_FORWARDS')
        op.backwards = False

        if _tracking_busy:
            row = box.row(align=True)
            row.alert = True
            row.label(text=_tracking_status or "Tracking...", icon='SORTTIME')
            row.enabled = False
        else:
            box.operator("rotoforge.track_video_points", text="Video Track (All Frames)", icon='SEQUENCE')

        col = box.column(align=True)
        col.prop(props, "guide_strength")
        col.prop(props, "tracking")
        col.prop(props, "search_radius")
        layout.separator()

        # Active Spline Settings
        spline_settings = layout.box()
        spline_settings.label(text="Active Spline Settings")

        if hasattr(active_layer, 'splines'):
            active_mask_spline = context.edit_mask.layers.active.splines.active
        else:
            active_mask_spline = None

        if active_mask_spline is not None:
            spline_settings.prop(active_mask_spline, "use_cyclic", text="\u2611Boundary|\u2610Prompt points")
            if not active_mask_spline.use_cyclic:
                spline_settings.prop(active_mask_spline, "use_fill", text="\u2611Mask|\u2610Background")
        else:
            spline_settings.label(text="No active spline detected")
        layout.separator()

        # Server / cache controls
        layout.operator("rotoforge.resync_masksequence", icon='FILE_REFRESH')
        layout.operator("rotoforge.free_predictor", text="Free GPU Cache", icon='TRASH')
        layout.operator("rotoforge.stop_server", text="Stop Server", icon='CANCEL')
        layout.separator()


class NodeImportControls(bpy.types.PropertyGroup):
    def update_mask_options(self, context):
        possible_mask = []
        for mask in bpy.data.masks:
            image_name = f"{mask.name}/Combined"
            if image_name in bpy.data.images:
                possible_mask.append(mask.name)
        if len(possible_mask) < 1:
            return [('NONE', 'No baked masks available', 'Please bake a mask first using "Bake Mask to Texture"')]
        return [(element, element, f'Import the mask "{element}"') for element in possible_mask]

    used_mask: bpy.props.EnumProperty(name="Used Mask", items=update_mask_options)  # type: ignore

    @classmethod
    def register(cls):
        bpy.types.Scene.rotoforge_importcontrols = bpy.props.PointerProperty(type=cls)

    @classmethod
    def unregister(cls):
        if hasattr(bpy.types.Scene, 'rotoforge_importcontrols'):
            del bpy.types.Scene.rotoforge_importcontrols


class RotoForgeNodePanel(bpy.types.Panel):
    """RotoForge Node Panel"""
    bl_label = "RotoForge"
    bl_idname = "ROTOFORGE_PT_RotoForgeNodePanel"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "RotoForge"
    bl_context = "node_editor"

    def draw(self, context):
        layout = self.layout
        import_props = bpy.context.scene.rotoforge_importcontrols
        layout.prop(import_props, 'used_mask')
        layout.operator('rotoforge.import_mask_node')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = [
    NodeImportControls,
    GenerateSingularMaskOperator,
    GenerateTextMaskOperator,
    TrackMaskOperator,
    TrackTextMaskOperator,
    TrackVideoTextOperator,
    TrackVideoPointsOperator,
    MergeMaskOperator,
    ImportMaskNodeOperator,
    MaskRangeToSceneOperator,
    FreePredictorOperator,
    StopServerOperator,
    LayerPanel,
    RotoForgeMaskPanel,
    RotoForgeNodePanel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    return {'REGISTERED'}


def unregister():
    stop_server()
    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    return {'UNREGISTERED'}
