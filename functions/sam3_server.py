#!/usr/bin/env python3
"""
RotoForge AI - SAM3 Inference Server

Standalone HTTP server that runs SAM3 model inference in a separate Python
environment (3.12+). Blender communicates with this over localhost HTTP.

Supports both prompt types:
  - Point/box prompts via SAM3InteractiveImagePredictor (SAM1-style path)
  - Text prompts via Sam3Processor (DETR detector path)

Usage:
    python sam3_server.py --port 8799
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np

# MPS (Apple Silicon) workaround — must be set BEFORE importing torch.
# Falls back to CPU for ops that exceed Metal's texture size limits (16384).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

# ---------------------------------------------------------------------------
# CUDA compatibility shim — must run BEFORE any sam3 imports.
#
# SAM3 was written for NVIDIA GPUs and has hundreds of hardcoded .cuda()
# calls and device="cuda" defaults.  On macOS (MPS) or CPU-only systems
# we monkey-patch torch so these calls route to the best available device.
# ---------------------------------------------------------------------------

_BEST_DEVICE = "cuda"

if not torch.cuda.is_available():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _BEST_DEVICE = "mps"
    else:
        _BEST_DEVICE = "cpu"

    # Patch Tensor.cuda() → Tensor.to(best_device)
    _orig_tensor_cuda = torch.Tensor.cuda
    def _tensor_cuda_shim(self, device=None, *args, **kwargs):
        return self.to(_BEST_DEVICE)
    torch.Tensor.cuda = _tensor_cuda_shim

    # Patch torch.cuda.is_available so SAM3 doesn't skip GPU code paths
    # that are actually device-agnostic (position encoding precompute, etc.)
    # We keep the real value accessible for our own get_device() below.
    _real_cuda_available = torch.cuda.is_available

    # Patch Module.cuda() → Module.to(best_device)
    _orig_module_cuda = torch.nn.Module.cuda
    def _module_cuda_shim(self, device=None):
        return self.to(_BEST_DEVICE)
    torch.nn.Module.cuda = _module_cuda_shim

    print(f"[sam3_server] No CUDA — patched .cuda() calls to use '{_BEST_DEVICE}'")


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_model = None
_processor = None
_video_predictor = None
_device = None
_device_name = None
_lock = threading.Lock()
_checkpoint_path = None  # set via --checkpoint CLI arg


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        return "cuda", "CUDA acceleration"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "MPS acceleration (Apple Silicon)"
    return "cpu", "CPU"


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def load_model():
    """Load the SAM3 model with both text and interactive-point capabilities."""
    global _model, _processor, _device, _device_name

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    empty_cache()
    _device, _device_name = get_device()
    print(f"[sam3_server] PyTorch {torch.__version__}, device: {_device_name}")

    ckpt = _checkpoint_path
    use_hf = ckpt is None
    if ckpt:
        print(f"[sam3_server] Loading checkpoint from {ckpt}")
    else:
        print("[sam3_server] No local checkpoint — downloading from HuggingFace")

    _model = build_sam3_image_model(
        device=_device,
        eval_mode=True,
        checkpoint_path=ckpt,
        load_from_HF=use_hf,
        enable_segmentation=True,
        enable_inst_interactivity=True,
    )
    _processor = Sam3Processor(_model, device=_device)

    empty_cache()
    print("[sam3_server] Model loaded (text + interactive point prompting)")


def load_video_predictor():
    """Load the SAM3 video predictor for temporal tracking."""
    global _video_predictor, _device, _device_name

    from sam3.model_builder import build_sam3_video_predictor

    if _device is None:
        _device, _device_name = get_device()

    ckpt = _checkpoint_path
    use_hf = ckpt is None

    empty_cache()
    print(f"[sam3_server] Loading video predictor on {_device_name}...")

    _video_predictor = build_sam3_video_predictor(
        checkpoint_path=ckpt,
        apply_temporal_disambiguation=True,
    )

    empty_cache()
    print("[sam3_server] Video predictor loaded")


def free_model():
    global _model, _processor, _video_predictor
    _model = None
    _processor = None
    if _video_predictor is not None:
        # Close any open sessions
        _video_predictor = None
    empty_cache()
    print("[sam3_server] All models freed")


# ---------------------------------------------------------------------------
# Prediction — point/box prompts (SAM1-style interactive predictor)
# ---------------------------------------------------------------------------

def predict_points(image_rgb: np.ndarray,
                   input_points=None, input_labels=None,
                   input_box=None, mask_input=None,
                   multimask_output=True):
    """Run point/box prediction via SAM3InteractiveImagePredictor.

    Args:
        image_rgb: HxWx3 uint8 numpy array
        input_points: Nx2 array of (X,Y) pixel coords, or None
        input_labels: N array (1=fg, 0=bg), or None
        input_box: length-4 XYXY pixel coords, or None
        mask_input: 1xHxW low-res logits from prior prediction, or None
        multimask_output: whether to return multiple candidate masks

    Returns:
        (masks, scores, low_res_masks) numpy arrays, or (None, None, None)
    """
    predictor = _model.inst_interactive_predictor
    if predictor is None:
        raise RuntimeError("Interactive predictor not available (model built without enable_inst_interactivity)")

    predictor.set_image(image_rgb)

    masks, scores, low_res_masks = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=input_box,
        mask_input=mask_input,
        multimask_output=multimask_output,
        return_logits=False,
        normalize_coords=True,
    )

    empty_cache()
    return masks, scores, low_res_masks


# ---------------------------------------------------------------------------
# Prediction — text prompts (DETR detector path)
# ---------------------------------------------------------------------------

def predict_text(image_rgb: np.ndarray, prompt: str,
                 confidence_threshold=0.5):
    """Run text-prompted segmentation via Sam3Processor.

    Args:
        image_rgb: HxWx3 uint8 numpy array
        prompt: text description of the object to segment
        confidence_threshold: detection confidence threshold

    Returns:
        dict with 'masks', 'boxes', 'scores' as numpy arrays, or None values
    """
    from PIL import Image

    pil_image = Image.fromarray(image_rgb)
    _processor.set_confidence_threshold(confidence_threshold)

    state = _processor.set_image(pil_image)
    state = _processor.set_text_prompt(prompt=prompt, state=state)

    masks = state.get("masks")
    boxes = state.get("boxes")
    scores_t = state.get("scores")

    if masks is None or len(masks) == 0:
        empty_cache()
        return {"masks": None, "boxes": None, "scores": None}

    masks_np = masks.squeeze(1).cpu().numpy().astype(bool)
    boxes_np = boxes.cpu().numpy() if boxes is not None else None
    scores_np = scores_t.cpu().numpy() if scores_t is not None else None

    empty_cache()
    return {"masks": masks_np, "boxes": boxes_np, "scores": scores_np}


# ---------------------------------------------------------------------------
# Video session management
# ---------------------------------------------------------------------------

def video_start_session(frames_dir: str):
    """Start a video tracking session from a JPEG frame directory."""
    if _video_predictor is None:
        raise RuntimeError("Video predictor not loaded")

    response = _video_predictor.handle_request(dict(
        type="start_session",
        resource_path=frames_dir,
    ))
    session_id = response["session_id"]
    print(f"[sam3_server] Video session started: {session_id}")
    return session_id


def video_add_prompt(session_id: str, frame_index: int,
                     text=None, points=None, labels=None, obj_id=None):
    """Add a prompt on a specific frame in a video session.

    Args:
        session_id: session from start_session
        frame_index: which frame to prompt on
        text: text prompt string, or None
        points: list of [x, y] normalized coords, or None
        labels: list of 0/1 labels for points, or None
        obj_id: optional object ID to assign
    """
    request = dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_index,
    )
    if text is not None:
        request["text"] = text
    if points is not None:
        request["points"] = points
        request["labels"] = labels if labels is not None else [1] * len(points)
    if obj_id is not None:
        request["obj_id"] = obj_id

    response = _video_predictor.handle_request(request)
    outputs = response.get("outputs", {})

    result_masks = []
    result_obj_ids = []
    result_scores = []

    if "out_binary_masks" in outputs:
        for i, obj in enumerate(outputs.get("out_obj_ids", [])):
            mask = outputs["out_binary_masks"][i]
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            result_masks.append(np.asarray(mask, dtype=bool))
            result_obj_ids.append(int(obj))
            if "out_probs" in outputs:
                score = outputs["out_probs"][i]
                result_scores.append(float(score) if not hasattr(score, 'item') else score.item())
            else:
                result_scores.append(1.0)

    return {
        "masks": result_masks,
        "obj_ids": result_obj_ids,
        "scores": result_scores,
    }


def video_propagate(session_id: str, direction="both"):
    """Propagate tracking across all frames. Returns a dict of frame_index → masks.

    Args:
        session_id: active session
        direction: "forward", "backward", or "both"
    """
    results = {}

    for frame_result in _video_predictor.handle_stream_request(dict(
        type="propagate_in_video",
        session_id=session_id,
        direction=direction,
    )):
        frame_idx = frame_result.get("frame_index")
        outputs = frame_result.get("outputs", {})

        frame_masks = []
        frame_obj_ids = []

        if "out_binary_masks" in outputs:
            for i, obj in enumerate(outputs.get("out_obj_ids", [])):
                mask = outputs["out_binary_masks"][i]
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                frame_masks.append(np.asarray(mask, dtype=bool))
                frame_obj_ids.append(int(obj))

        results[frame_idx] = {
            "masks": frame_masks,
            "obj_ids": frame_obj_ids,
        }

    print(f"[sam3_server] Propagated {len(results)} frames")
    empty_cache()
    return results


def video_close_session(session_id: str):
    """Close a video session and free its resources."""
    _video_predictor.handle_request(dict(
        type="close_session",
        session_id=session_id,
    ))
    empty_cache()
    print(f"[sam3_server] Session closed: {session_id}")


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def ndarray_to_b64(arr: np.ndarray) -> dict:
    return {
        "data": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def b64_to_ndarray(obj: dict) -> np.ndarray:
    raw = base64.b64decode(obj["data"])
    return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class SAM3Handler(BaseHTTPRequestHandler):

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw)

    def log_message(self, format, *args):
        pass

    # ----- GET routes -----

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "image_model_loaded": _model is not None,
                "video_model_loaded": _video_predictor is not None,
                "device": _device_name,
                "torch_version": torch.__version__,
            })
        elif self.path == "/shutdown":
            self._send_json(200, {"status": "shutting_down"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self._send_json(404, {"error": "not_found"})

    # ----- POST routes -----

    def do_POST(self):
        try:
            if self.path == "/load_model":
                with _lock:
                    load_model()
                self._send_json(200, {"status": "loaded", "device": _device_name})

            elif self.path == "/free":
                with _lock:
                    free_model()
                self._send_json(200, {"status": "freed"})

            elif self.path == "/predict_points":
                self._handle_predict_points()

            elif self.path == "/predict_text":
                self._handle_predict_text()

            elif self.path == "/load_video_model":
                with _lock:
                    load_video_predictor()
                self._send_json(200, {"status": "loaded", "device": _device_name})

            elif self.path == "/video/upload_frames":
                self._handle_video_upload_frames()

            elif self.path == "/video/start_session":
                self._handle_video_start()

            elif self.path == "/video/add_prompt":
                self._handle_video_add_prompt()

            elif self.path == "/video/propagate":
                self._handle_video_propagate()

            elif self.path == "/video/close_session":
                self._handle_video_close()

            else:
                self._send_json(404, {"error": "not_found"})

        except Exception:
            tb = traceback.format_exc()
            print(f"[sam3_server] Error:\n{tb}", file=sys.stderr)
            self._send_json(500, {"error": tb})

    def _handle_predict_points(self):
        if _model is None:
            self._send_json(503, {"error": "model_not_loaded"})
            return

        data = self._read_json()
        image_rgb = b64_to_ndarray(data["image_rgb"])

        input_points = b64_to_ndarray(data["input_points"]) if data.get("input_points") else None
        input_labels = b64_to_ndarray(data["input_labels"]) if data.get("input_labels") else None
        input_box = b64_to_ndarray(data["input_box"]) if data.get("input_box") else None
        mask_input = b64_to_ndarray(data["mask_input"]) if data.get("mask_input") else None
        multimask = data.get("multimask_output", True)

        with _lock:
            masks, scores, low_res_masks = predict_points(
                image_rgb, input_points, input_labels,
                input_box, mask_input, multimask,
            )

        if masks is None:
            self._send_json(200, {"masks": None, "scores": None, "low_res_masks": None})
        else:
            self._send_json(200, {
                "masks": ndarray_to_b64(masks),
                "scores": ndarray_to_b64(scores),
                "low_res_masks": ndarray_to_b64(low_res_masks),
            })

    def _handle_predict_text(self):
        if _model is None:
            self._send_json(503, {"error": "model_not_loaded"})
            return

        data = self._read_json()
        image_rgb = b64_to_ndarray(data["image_rgb"])
        prompt = data["prompt"]
        confidence = data.get("confidence_threshold", 0.5)

        with _lock:
            result = predict_text(image_rgb, prompt, confidence)

        resp = {}
        for key in ("masks", "boxes", "scores"):
            val = result[key]
            resp[key] = ndarray_to_b64(val) if val is not None else None

        self._send_json(200, resp)

    # ----- video session routes -----

    def _handle_video_upload_frames(self):
        """Receive base64-encoded JPEG frames and write them to a temp dir."""
        data = self._read_json()
        frames = data.get("frames", {})
        if not frames:
            self._send_json(400, {"error": "no frames provided"})
            return

        tmp_dir = tempfile.mkdtemp(prefix="rotoforge_frames_")
        for fname, b64data in frames.items():
            fpath = os.path.join(tmp_dir, fname)
            with open(fpath, "wb") as f:
                f.write(base64.b64decode(b64data))

        print(f"[sam3_server] Received {len(frames)} frames → {tmp_dir}")
        self._send_json(200, {"frames_dir": tmp_dir})

    def _handle_video_start(self):
        if _video_predictor is None:
            self._send_json(503, {"error": "video_model_not_loaded"})
            return
        data = self._read_json()
        frames_dir = data["frames_dir"]
        is_dir = os.path.isdir(frames_dir)
        contents = os.listdir(frames_dir) if is_dir else []
        print(f"[sam3_server] video_start: frames_dir={frames_dir!r}  isdir={is_dir}  files={len(contents)}")
        with _lock:
            session_id = video_start_session(frames_dir)
        self._send_json(200, {"session_id": session_id})

    def _handle_video_add_prompt(self):
        if _video_predictor is None:
            self._send_json(503, {"error": "video_model_not_loaded"})
            return
        data = self._read_json()
        with _lock:
            result = video_add_prompt(
                session_id=data["session_id"],
                frame_index=data["frame_index"],
                text=data.get("text"),
                points=data.get("points"),
                labels=data.get("labels"),
                obj_id=data.get("obj_id"),
            )

        resp = {"obj_ids": result["obj_ids"], "scores": result["scores"]}
        if result["masks"]:
            resp["masks"] = [ndarray_to_b64(m) for m in result["masks"]]
        else:
            resp["masks"] = []
        self._send_json(200, resp)

    def _handle_video_propagate(self):
        if _video_predictor is None:
            self._send_json(503, {"error": "video_model_not_loaded"})
            return
        data = self._read_json()
        direction = data.get("direction", "both")

        with _lock:
            results = video_propagate(data["session_id"], direction)

        resp = {}
        for frame_idx, frame_data in results.items():
            encoded_masks = [ndarray_to_b64(m) for m in frame_data["masks"]]
            resp[str(frame_idx)] = {
                "masks": encoded_masks,
                "obj_ids": frame_data["obj_ids"],
            }

        self._send_json(200, resp)

    def _handle_video_close(self):
        if _video_predictor is None:
            self._send_json(503, {"error": "video_model_not_loaded"})
            return
        data = self._read_json()
        with _lock:
            video_close_session(data["session_id"])
        self._send_json(200, {"status": "closed"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _get_lan_ip():
    """Best-effort LAN IP detection for display purposes."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


def main():
    global _checkpoint_path

    parser = argparse.ArgumentParser(
        description="RotoForge SAM3 inference server. "
        "Run on a machine with a GPU, then point Blender to its IP.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0 = all interfaces)")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to local sam3.pt checkpoint file")
    args = parser.parse_args()

    if args.checkpoint and os.path.isfile(args.checkpoint):
        _checkpoint_path = args.checkpoint
        print(f"[sam3_server] Using local checkpoint: {_checkpoint_path}")
    elif args.checkpoint:
        print(f"[sam3_server] WARNING: checkpoint not found at {args.checkpoint}")

    server = HTTPServer((args.host, args.port), SAM3Handler)

    lan_ip = _get_lan_ip()
    print(f"[sam3_server] Listening on http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print(f"[sam3_server] LAN IP: {lan_ip}  — enter this in Blender's addon preferences")
    sys.stdout.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[sam3_server] Server stopped")


if __name__ == "__main__":
    main()
