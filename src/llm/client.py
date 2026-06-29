"""Qwen API 客户端 — LM Studio 原生接口，支持图片"""

import base64, time, json, urllib.request
from typing import Optional
import cv2, numpy as np
from logging import getLogger

logger = getLogger(__name__)


class QwenAPIClient:
    def __init__(
        self, base_url="http://127.0.0.1:1234/v1", api_key="",
        model_name="", crop_size=448, temperature=0.1,
        max_tokens=512, timeout_sec=60, max_retries=2,
    ):
        self.base_url = base_url.rstrip("/v1").rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.crop_size = crop_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout_sec
        self.max_retries = max_retries

    def chat(self, system_prompt="", user_text="",
             image: Optional[np.ndarray] = None) -> dict:
        """返回 {"content":str, "finish":str, "tokens":int, "elapsed":float}"""
        url = f"{self.base_url}/api/v1/chat"

        # 支持图片：data URL 格式拼在文本前面
        input_content = []
        if image is not None:
            img_b64 = self._encode_image(image)
            input_content.append({"type": "image", "data_url": f"data:image/jpeg;base64,{img_b64}"})
        if user_text:
            input_content.append({"type": "text", "content": user_text})
        if not input_content:
            input_content = ""

        body = {
            "model": self.model_name,
            "system_prompt": system_prompt,
            "input": input_content if input_content else user_text,
            "temperature": self.temperature,
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                req = urllib.request.Request(url, data=data, method="POST")
                req.add_header("Content-Type", "application/json")
                req.add_header("Authorization", f"Bearer {self.api_key}")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))

                dt = time.time() - t0
                content_text = ""
                for item in raw.get("output", []):
                    if item.get("type") == "message":
                        content_text = item.get("content", "").strip()
                        break
                tokens = raw.get("stats", {}).get("total_output_tokens", 0)
                return {"content": content_text, "finish": "stop",
                        "tokens": tokens, "elapsed": dt}
            except Exception as e:
                logger.warning(f"Qwen attempt {attempt+1}: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        return {"content": "", "finish": "error", "tokens": 0, "elapsed": 0}

    def text_only(self, system_prompt, user_text):
        return self.chat(system_prompt, user_text, image=None)

    def _encode_image(self, image):
        h, w = image.shape[:2]
        if w != self.crop_size or h != self.crop_size:
            image = cv2.resize(image, (self.crop_size, self.crop_size))
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")
