"""
ComfyUI Local Image Generation & Editing Service.

Integrates with a local ComfyUI instance (http://127.0.0.1:8188) using its
REST API and WebSocket/polling pipeline for txt2img and img2img workflows.
Handles offline states, model checks, and job lifecycle gracefully.
"""
import io
import json
import logging
import time
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import requests

import config

logger = logging.getLogger(__name__)


class ComfyUIError(RuntimeError):
    """Base exception for ComfyUI communication failures."""


class ComfyUIOfflineError(ComfyUIError):
    """Raised when the ComfyUI server is unreachable."""


class ComfyUIJobFailedError(ComfyUIError):
    """Raised when a ComfyUI generation/editing prompt execution fails."""


class ComfyUIService:
    """Client for local ComfyUI image engine."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = (base_url or config.COMFYUI_BASE_URL).rstrip("/")
        self.timeout = timeout or config.COMFYUI_TIMEOUT
        self.client_id = str(uuid.uuid4())

    def is_available(self) -> bool:
        """Check if ComfyUI local server is online and responding."""
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=2.5)
            return resp.status_code == 200
        except Exception:
            return False

    def get_system_stats(self) -> Dict[str, Any]:
        """Fetch system statistics, GPU info, and device memory from ComfyUI."""
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=4.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"ComfyUI system_stats check failed: {e}")
        return {"devices": [], "status": "offline"}

    def upload_image(self, image_bytes: bytes, filename: str = "input_image.png") -> str:
        """Upload an input image to ComfyUI for img2img / editing workflows."""
        url = f"{self.base_url}/upload/image"
        files = {"image": (filename, image_bytes, "image/png")}
        data = {"overwrite": "true"}
        try:
            resp = requests.post(url, files=files, data=data, timeout=15.0)
            if resp.status_code == 200:
                result = resp.json()
                return result.get("name", filename)
            raise ComfyUIError(f"Upload failed: HTTP {resp.status_code} - {resp.text}")
        except requests.RequestException as exc:
            raise ComfyUIOfflineError(f"Could not connect to ComfyUI upload endpoint: {exc}") from exc

    def _build_txt2img_workflow(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, deformed, bad anatomy, artifacts",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Standard Stable Diffusion txt2img API graph."""
        actual_seed = seed if seed is not None else int(time.time() * 1000) % 1000000000
        ckpt = model_name or config.COMFYUI_MODEL

        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": actual_seed,
                    "steps": steps,
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": ckpt},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": 1, "height": height, "width": width},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": prompt},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": negative_prompt},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "NexusAI_Txt2Img", "images": ["8", 0]},
            },
        }

    def _build_img2img_workflow(
        self,
        image_name: str,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, deformed",
        denoise: float = 0.65,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Standard Stable Diffusion img2img API graph for image editing."""
        actual_seed = seed if seed is not None else int(time.time() * 1000) % 1000000000
        ckpt = model_name or config.COMFYUI_MODEL

        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "2": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["1", 0], "vae": ["4", 2]},
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg,
                    "denoise": denoise,
                    "latent_image": ["2", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": actual_seed,
                    "steps": steps,
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": ckpt},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": prompt},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": negative_prompt},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "NexusAI_Edit", "images": ["8", 0]},
            },
        }

    def _queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Submit a workflow prompt graph to ComfyUI queue."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        url = f"{self.base_url}/prompt"
        try:
            resp = requests.post(url, json=payload, timeout=10.0)
            if resp.status_code != 200:
                raise ComfyUIError(f"Failed to queue prompt: {resp.status_code} - {resp.text}")
            data = resp.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise ComfyUIError(f"ComfyUI did not return prompt_id: {data}")
            return prompt_id
        except requests.RequestException as e:
            raise ComfyUIOfflineError(f"ComfyUI connection error: {e}") from e

    def _get_output_image(self, prompt_id: str, timeout_seconds: float = 60.0) -> bytes:
        """Poll ComfyUI history until job finishes, then download the resulting image."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                hist_resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=5.0)
                if hist_resp.status_code == 200:
                    history = hist_resp.json()
                    if prompt_id in history:
                        job_data = history[prompt_id]
                        status = job_data.get("status", {})
                        if status.get("status_str") == "error":
                            raise ComfyUIJobFailedError(f"ComfyUI job failed: {status.get('messages')}")

                        outputs = job_data.get("outputs", {})
                        for node_id, node_out in outputs.items():
                            images = node_out.get("images", [])
                            if images:
                                img_info = images[0]
                                filename = img_info.get("filename")
                                subfolder = img_info.get("subfolder", "")
                                img_type = img_info.get("type", "output")

                                params = {
                                    "filename": filename,
                                    "subfolder": subfolder,
                                    "type": img_type,
                                }
                                view_url = f"{self.base_url}/view?{urllib.parse.urlencode(params)}"
                                view_resp = requests.get(view_url, timeout=15.0)
                                if view_resp.status_code == 200:
                                    return view_resp.content
            except (ComfyUIJobFailedError, ComfyUIOfflineError):
                raise
            except Exception as poll_err:
                logger.debug(f"Waiting for ComfyUI job {prompt_id}: {poll_err}")

            time.sleep(1.0)

        raise ComfyUIError(f"ComfyUI generation timed out after {timeout_seconds}s")

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, deformed",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        """Generate an image using ComfyUI txt2img workflow."""
        if not self.is_available():
            raise ComfyUIOfflineError(
                f"ComfyUI is offline at {self.base_url}. Please launch ComfyUI to enable local image generation."
            )

        workflow = self._build_txt2img_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            model_name=model_name,
        )
        prompt_id = self._queue_prompt(workflow)
        img_bytes = self._get_output_image(prompt_id)
        return img_bytes, model_name or config.COMFYUI_MODEL

    def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, deformed",
        denoise: float = 0.65,
        steps: int = 20,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> Tuple[bytes, str]:
        """Edit an existing image using ComfyUI img2img workflow."""
        if not self.is_available():
            raise ComfyUIOfflineError(
                f"ComfyUI is offline at {self.base_url}. Please launch ComfyUI to enable local image editing."
            )

        uploaded_name = self.upload_image(image_bytes, f"edit_{uuid.uuid4().hex[:8]}.png")
        workflow = self._build_img2img_workflow(
            image_name=uploaded_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            denoise=denoise,
            steps=steps,
            cfg=cfg,
            seed=seed,
            model_name=model_name,
        )
        prompt_id = self._queue_prompt(workflow)
        img_bytes = self._get_output_image(prompt_id)
        return img_bytes, model_name or config.COMFYUI_MODEL

