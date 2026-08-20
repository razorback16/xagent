"""Local image attachments for multimodal provider requests.

The agent's providers use Anthropic's content-block shape, so an image is loaded
once at the boundary and then rendered as a normal ``image`` block in the opening
user message. Keeping the conversion here means the context store and provider do
not need to know about paths, MIME detection, or base64 encoding.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_IMAGE_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
SUPPORTED_MEDIA_TYPES = tuple(dict.fromkeys(SUPPORTED_IMAGE_TYPES.values()))

# A deliberately conservative placeholder for budgeting. Image bytes are not
# tokenized as the base64 string sent over the wire; vision models charge an
# image-dependent amount that is not available from the standard library.
IMAGE_TOKEN_ESTIMATE = 1_600


def _matches_signature(data: bytes, media_type: str) -> bool:
    if media_type == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if media_type == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if media_type == "image/gif":
        return data[:6] in (b"GIF87a", b"GIF89a")
    if media_type == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


@dataclass(frozen=True)
class ImageAttachment:
    """A validated local image ready for an Anthropic-compatible request."""

    path: Path
    media_type: str
    data: bytes

    @classmethod
    def from_path(cls, path: str | Path) -> "ImageAttachment":
        original = Path(path).expanduser()
        media_type = SUPPORTED_IMAGE_TYPES.get(original.suffix.lower())
        if media_type is None:
            supported = ", ".join(sorted(SUPPORTED_IMAGE_TYPES))
            raise ValueError(
                f"unsupported image type for {original}; use one of {supported}"
            )
        if not original.exists():
            raise FileNotFoundError(f"image file does not exist: {original}")
        if not original.is_file():
            raise ValueError(f"image path is not a file: {original}")
        data = original.read_bytes()
        if not data:
            raise ValueError(f"image file is empty: {original}")
        if not _matches_signature(data, media_type):
            raise ValueError(f"file does not contain a valid {media_type} image: {original}")
        return cls(path=original, media_type=media_type, data=data)

    def content_block(self) -> dict:
        """Return the wire-format image content block."""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": base64.b64encode(self.data).decode("ascii"),
            },
        }


ImageInput = str | Path | ImageAttachment


def normalize_images(
    images: ImageInput | Sequence[ImageInput] | None,
) -> list[ImageAttachment]:
    """Load and validate one image or a sequence of image inputs."""
    if images is None:
        return []
    if isinstance(images, (str, Path, ImageAttachment)):
        images = [images]
    return [
        image if isinstance(image, ImageAttachment) else ImageAttachment.from_path(image)
        for image in images
    ]


def _without_image_data(value):
    """Replace base64 payloads before estimating a request's text size."""
    if isinstance(value, dict):
        result = {key: _without_image_data(item) for key, item in value.items()}
        if value.get("type") == "image":
            source = dict(result.get("source") or {})
            if "data" in source:
                source["data"] = "<image bytes omitted>"
            result["source"] = source
        return result
    if isinstance(value, list):
        return [_without_image_data(item) for item in value]
    return value


def _iter_blocks(value):
    if isinstance(value, dict):
        if value.get("type") == "image":
            yield value
        for item in value.values():
            yield from _iter_blocks(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_blocks(item)


def estimate_message_tokens(messages: list[dict]) -> int:
    """Estimate message tokens without treating base64 bytes as text tokens."""
    sanitized = _without_image_data(messages)
    image_count = sum(
        1
        for message in messages
        for block in _iter_blocks(message)
        if isinstance(block, dict) and block.get("type") == "image"
    )
    return len(json.dumps(sanitized)) // 4 + image_count * IMAGE_TOKEN_ESTIMATE
