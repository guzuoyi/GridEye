"""Qwen API 客户端 —— LM Studio OpenAI 兼容接口"""

import base64
import time
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from openai import OpenAI
from logging import getLogger

logger = getLogger(__name__)


class QwenAPIClient:
    """LM Studio Qwen API 客户端。

    用法:
        client = QwenAPIClient(
            base_url="http://192.168.3.251:1234/v1",
            model_name="qwen3.5-9b-uncensored-hauhaucs-aggressive",
        )
        result = client.chat("你是交通专家", "这辆车违停了吗？", crop_image)
    """

    def __init__(
        self,
        base_url: str = "http://192.168.3.251:1234/v1",
        api_key: str = "lm-studio",
        model_name: str = "qwen3.5-9b-uncensored-hauhaucs-aggressive",
        crop_size: int = 448,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout_sec: float = 60,
        max_retries: int = 2,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.crop_size = crop_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout_sec
        self.max_retries = max_retries
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_sec)

    # ---- 核心调用 -----------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_text: str,
        image: Optional[np.ndarray] = None,
    ) -> dict:
        """发送 chat 请求，返回解析后的 dict。

        参数:
            system_prompt: 系统提示
            user_text: 用户文本
            image: 可选的目标裁剪图 (H, W, 3) BGR

        返回:
            {"content": str, "finish": str, "tokens": int, "elapsed": float}
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.append(self._build_user_message(user_text, image))

        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                r = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                dt = time.time() - t0
                msg = r.choices[0].message
                content = msg.content or ""
                finish = r.choices[0].finish_reason
                tokens = r.usage.completion_tokens

                logger.info(
                    f"Qwen {finish}: {tokens}t/{dt:.1f}s"
                )
                return {
                    "content": content,
                    "finish": finish,
                    "tokens": tokens,
                    "elapsed": dt,
                }

            except Exception as e:
                logger.warning(f"Qwen attempt {attempt+1}/{self.max_retries+1}: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return {"content": "", "finish": "error", "tokens": 0, "elapsed": 0}

    def _build_user_message(self, text: str, image: Optional[np.ndarray]) -> dict:
        """构建 OpenAI Vision 格式的 user message。"""
        content = []
        content.append({"type": "text", "text": text})

        if image is not None:
            img_b64 = self._encode_image(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": "auto",
                },
            })

        return {"role": "user", "content": content}

    def _encode_image(self, image: np.ndarray) -> str:
        """将图像缩放到 crop_size × crop_size 并 base64 编码。"""
        h, w = image.shape[:2]
        # 缩放到 crop_size
        if w != self.crop_size or h != self.crop_size:
            image = cv2.resize(image, (self.crop_size, self.crop_size))

        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    def text_only(self, system_prompt: str, user_text: str) -> dict:
        """纯文本推理（无图片）。"""
        return self.chat(system_prompt, user_text, image=None)
