"""
RotoForge AI - SAM3 Server Client

Lightweight HTTP client that talks to the sam3_server process.
Uses only Python stdlib + numpy — no torch, no sam3 needed on the Blender side.
"""

import base64
import io
import json
import subprocess
import sys
import os
import tarfile
import time
import urllib.request
import urllib.error
import zlib

import numpy as np


# ---------------------------------------------------------------------------
# Encoding helpers (mirrors server side)
# ---------------------------------------------------------------------------

def ndarray_to_b64(arr: np.ndarray) -> dict:
    raw = np.ascontiguousarray(arr).tobytes()
    compressed = zlib.compress(raw, level=1)
    return {
        "data": base64.b64encode(compressed).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "zlib": True,
    }


def b64_to_ndarray(obj: dict) -> np.ndarray:
    raw = base64.b64decode(obj["data"])
    if obj.get("zlib"):
        raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class SAM3Client:
    """Manages a sam3_server subprocess (local) or connects to a remote
    server, and provides prediction APIs for point/box and text prompts."""

    def __init__(self, host="127.0.0.1", port=8799):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._process = None
        self._is_remote = False

    # -- server lifecycle ---------------------------------------------------

    def start_server(self, python_exe: str, server_script: str,
                     timeout: float = 180.0, checkpoint: str | None = None):
        """Launch a local server subprocess and wait until it's healthy."""
        if self.is_alive():
            print("RotoForge AI: Server already running")
            return

        print(f"RotoForge AI: Starting SAM3 server on port {self.port}...")
        print(f"  Python:  {python_exe}")
        print(f"  Script:  {server_script}")

        cmd = [python_exe, server_script, "--port", str(self.port)]
        if checkpoint and os.path.isfile(checkpoint):
            cmd += ["--checkpoint", checkpoint]
            print(f"  Weights:  {checkpoint}")

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        self._is_remote = False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                out = self._process.stdout.read() if self._process.stdout else ""
                raise RuntimeError(
                    f"SAM3 server exited early (code {self._process.returncode}):\n{out}"
                )
            if self.is_alive():
                print("RotoForge AI: SAM3 server is ready")
                return
            time.sleep(0.5)

        self.stop_server()
        raise RuntimeError(f"SAM3 server did not respond within {timeout}s")

    def connect_remote(self, timeout: float = 10.0):
        """Connect to an already-running remote server.

        Raises RuntimeError if the server is unreachable.
        """
        self._is_remote = True
        self._process = None

        print(f"RotoForge AI: Connecting to remote SAM3 server at {self.base_url}...")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_alive():
                print(f"RotoForge AI: Connected to remote server at {self.base_url}")
                return
            time.sleep(0.5)

        raise RuntimeError(
            f"Cannot reach SAM3 server at {self.base_url}. "
            f"Make sure sam3_server.py is running on the remote machine."
        )

    def stop_server(self):
        """Gracefully shut down a local server subprocess.

        No-op for remote servers (we don't own them).
        """
        if self._is_remote:
            return
        if self._process is None:
            return
        try:
            self._request("GET", "/shutdown")
        except Exception:
            pass
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None
        print("RotoForge AI: SAM3 server stopped")

    def is_alive(self) -> bool:
        try:
            resp = self._request("GET", "/health", timeout=5)
            return resp.get("status") == "ok"
        except Exception:
            return False

    def is_model_loaded(self) -> bool:
        try:
            resp = self._request("GET", "/health", timeout=5)
            return resp.get("image_model_loaded", False)
        except Exception:
            return False

    def is_video_model_loaded(self) -> bool:
        try:
            resp = self._request("GET", "/health", timeout=5)
            return resp.get("video_model_loaded", False)
        except Exception:
            return False

    # -- model management ---------------------------------------------------

    def load_model(self):
        """Tell the server to load the SAM3 model (downloads from HF on first run)."""
        resp = self._request("POST", "/load_model", timeout=600)
        print(f"RotoForge AI: Model loaded on {resp.get('device', 'unknown')}")
        return resp

    def free_model(self):
        """Free GPU memory on the server."""
        return self._request("POST", "/free")

    def load_video_model(self):
        """Tell the server to load the SAM3 video predictor."""
        resp = self._request("POST", "/load_video_model", timeout=600)
        print(f"RotoForge AI: Video model loaded on {resp.get('device', 'unknown')}")
        return resp

    # -- point/box prediction (SAM1-style interactive predictor) ------------

    def predict_points(self, image_rgb: np.ndarray,
                       input_points=None, input_labels=None,
                       input_box=None, mask_input=None,
                       multimask_output=True):
        """Point/box prompt prediction.

        Args:
            image_rgb: HxWx3 uint8 numpy array.
            input_points: Nx2 array of (X,Y) pixel coords, or None.
            input_labels: N array (1=fg, 0=bg), or None.
            input_box: length-4 XYXY pixel coords, or None.
            mask_input: 1xHxW low-res logits from prior iteration, or None.
            multimask_output: return multiple candidates for ambiguous prompts.

        Returns:
            (masks, scores, low_res_masks) as numpy arrays, or (None, None, None).
        """
        payload = {
            "image_rgb": ndarray_to_b64(image_rgb),
            "multimask_output": multimask_output,
        }
        if input_points is not None:
            payload["input_points"] = ndarray_to_b64(np.asarray(input_points, dtype=np.float32))
        if input_labels is not None:
            payload["input_labels"] = ndarray_to_b64(np.asarray(input_labels, dtype=np.int32))
        if input_box is not None:
            payload["input_box"] = ndarray_to_b64(np.asarray(input_box, dtype=np.float32))
        if mask_input is not None:
            payload["mask_input"] = ndarray_to_b64(np.asarray(mask_input, dtype=np.float32))

        resp = self._request("POST", "/predict_points", payload)

        if resp.get("masks") is None:
            return None, None, None

        masks = b64_to_ndarray(resp["masks"])
        scores = b64_to_ndarray(resp["scores"])
        low_res = b64_to_ndarray(resp["low_res_masks"])
        return masks, scores, low_res

    # -- text prediction (DETR detector path) -------------------------------

    def predict_text(self, image_rgb: np.ndarray, prompt: str,
                     confidence_threshold=0.5):
        """Text-prompted segmentation.

        Args:
            image_rgb: HxWx3 uint8 numpy array.
            prompt: text description of the target object.
            confidence_threshold: detector confidence cutoff.

        Returns:
            (masks, boxes, scores) as numpy arrays, or (None, None, None).
        """
        payload = {
            "image_rgb": ndarray_to_b64(image_rgb),
            "prompt": prompt,
            "confidence_threshold": confidence_threshold,
        }

        resp = self._request("POST", "/predict_text", payload)

        if resp.get("masks") is None:
            return None, None, None

        masks = b64_to_ndarray(resp["masks"])
        boxes = b64_to_ndarray(resp["boxes"]) if resp.get("boxes") else None
        scores = b64_to_ndarray(resp["scores"]) if resp.get("scores") else None
        return masks, boxes, scores

    # -- video session API (temporal tracking) --------------------------------

    def video_upload_frames(self, local_frames_dir: str) -> str:
        """Upload local JPEG frames to the server as a tar archive.

        Used for remote servers where the local filesystem isn't shared.
        Packs all .jpg files into an uncompressed tar (JPEGs are already
        compressed) and sends the raw bytes. ~33% smaller than base64-in-JSON.

        Returns:
            The server-side directory path containing the uploaded frames.
        """
        buf = io.BytesIO()
        count = 0
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for fname in sorted(os.listdir(local_frames_dir)):
                if not fname.lower().endswith(('.jpg', '.jpeg')):
                    continue
                tar.add(os.path.join(local_frames_dir, fname), arcname=fname)
                count += 1

        print(f"RotoForge AI: Uploading {count} frames to server...")
        resp = self._request_binary("POST", "/video/upload_frames",
                                    buf.getvalue(),
                                    content_type="application/x-tar",
                                    timeout=300)
        print(f"RotoForge AI: Frames uploaded to server")
        return resp["frames_dir"]

    def video_start_session(self, frames_dir: str) -> str:
        """Start a video tracking session from a JPEG frame directory.

        Args:
            frames_dir: Path to a directory of JPEG frames on the server,
                        named like 00000.jpg, 00001.jpg, etc.

        Returns:
            session_id string.
        """
        resp = self._request("POST", "/video/start_session",
                             {"frames_dir": frames_dir}, timeout=120)
        return resp["session_id"]

    def video_add_prompt(self, session_id: str, frame_index: int,
                         text=None, points=None, labels=None, obj_id=None,
                         confidence_threshold=None):
        """Add a prompt on a specific frame in a video session.

        Args:
            session_id: from video_start_session.
            frame_index: 0-based frame index to prompt on.
            text: text prompt (e.g. "person"), or None.
            points: list of [x, y] normalized coords (0-1), or None.
            labels: list of 0/1 for points, or None.
            obj_id: optional explicit object ID.
            confidence_threshold: detection confidence for text prompts.

        Returns:
            dict with 'masks' (list of np arrays), 'obj_ids', 'scores'.
        """
        payload = {
            "session_id": session_id,
            "frame_index": frame_index,
        }
        if text is not None:
            payload["text"] = text
        if confidence_threshold is not None:
            payload["confidence_threshold"] = confidence_threshold
        if points is not None:
            payload["points"] = points
            payload["labels"] = labels if labels is not None else [1] * len(points)
        if obj_id is not None:
            payload["obj_id"] = obj_id

        resp = self._request("POST", "/video/add_prompt", payload, timeout=120)

        masks = [b64_to_ndarray(m) for m in resp["masks"]] if resp.get("masks") else []
        return {
            "masks": masks,
            "obj_ids": resp.get("obj_ids", []),
            "scores": resp.get("scores", []),
        }

    def video_propagate(self, session_id: str, direction="both",
                        fill_hole_area=16, progress_callback=None):
        """Propagate tracking across all video frames.

        Args:
            session_id: active session.
            direction: "forward", "backward", or "both".
            fill_hole_area: pixel area threshold for hole filling (0 = disabled).
            progress_callback: optional callable(frames_done: int) called as
                each frame result arrives from the server stream.

        Returns:
            dict mapping frame_index (int) → {"masks": [np arrays], "obj_ids": [ints]}.
        """
        url = self.base_url + "/video/propagate"
        payload = json.dumps({
            "session_id": session_id,
            "direction": direction,
            "fill_hole_area": fill_hole_area,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=payload, headers=headers,
                                     method="POST")

        results = {}
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                while True:
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.strip()
                    if not line:
                        continue
                    frame_data = json.loads(line)
                    frame_idx = int(frame_data["frame_idx"])
                    masks = ([b64_to_ndarray(m) for m in frame_data["masks"]]
                             if frame_data.get("masks") else [])
                    results[frame_idx] = {
                        "masks": masks,
                        "obj_ids": frame_data.get("obj_ids", []),
                    }
                    if progress_callback:
                        progress_callback(len(results))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SAM3 server error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach SAM3 server at {url}: {e}") from e

        return results

    def video_close_session(self, session_id: str):
        """Close a video session and free its resources."""
        return self._request("POST", "/video/close_session",
                             {"session_id": session_id})

    # -- HTTP plumbing ------------------------------------------------------

    def _request(self, method: str, path: str, body=None, timeout=300):
        url = self.base_url + path
        data = None
        headers = {}

        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SAM3 server error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot reach SAM3 server at {url}: {e}") from e

    def _request_binary(self, method: str, path: str, data: bytes,
                        content_type: str, timeout=300):
        """Send raw binary data, expect a JSON response."""
        url = self.base_url + path
        headers = {"Content-Type": content_type}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SAM3 server error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot reach SAM3 server at {url}: {e}") from e
