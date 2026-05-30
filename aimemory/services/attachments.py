import base64
import binascii
import hashlib
from dataclasses import dataclass
from typing import Any


ALLOWED_IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_ATTACHMENTS_PER_MEMORY = 5
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
MAX_TOTAL_ATTACHMENT_BYTES = 15 * 1024 * 1024


class AttachmentValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DecodedAttachment:
    filename: str
    mime_type: str
    image_bytes: bytes
    size_bytes: int
    sha256: str
    description: str | None
    ocr_text: str | None
    metadata: dict[str, Any]

    @property
    def search_text(self) -> str:
        parts = [
            self.filename,
            self.description or "",
            self.ocr_text or "",
            _metadata_search_text(self.metadata),
        ]
        return "\n".join(part for part in parts if part)


def decode_attachment_inputs(inputs: list[Any] | None) -> list[DecodedAttachment]:
    if inputs is None:
        return []
    if len(inputs) > MAX_ATTACHMENTS_PER_MEMORY:
        raise AttachmentValidationError(f"最多允许 {MAX_ATTACHMENTS_PER_MEMORY} 个图片附件。")

    decoded: list[DecodedAttachment] = []
    total_size = 0
    for item in inputs:
        filename = item.filename.strip()
        mime_type = item.mime_type.strip().lower()
        if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise AttachmentValidationError("不支持的图片 MIME 类型。")

        raw_base64 = _strip_data_uri(item.data_base64, mime_type)
        try:
            image_bytes = base64.b64decode(raw_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AttachmentValidationError("图片 base64 编码无效。") from exc

        size_bytes = len(image_bytes)
        if size_bytes <= 0:
            raise AttachmentValidationError("图片附件不能为空。")
        if size_bytes > MAX_ATTACHMENT_BYTES:
            raise AttachmentValidationError("单个图片附件超过 5MB。")
        if not _bytes_match_mime(image_bytes, mime_type):
            raise AttachmentValidationError("图片内容与 MIME 类型不匹配。")

        total_size += size_bytes
        if total_size > MAX_TOTAL_ATTACHMENT_BYTES:
            raise AttachmentValidationError("单条记忆的图片附件总大小超过 15MB。")

        decoded.append(
            DecodedAttachment(
                filename=filename,
                mime_type=mime_type,
                image_bytes=image_bytes,
                size_bytes=size_bytes,
                sha256=hashlib.sha256(image_bytes).hexdigest(),
                description=item.description.strip() if item.description else None,
                ocr_text=item.ocr_text.strip() if item.ocr_text else None,
                metadata=item.metadata or {},
            )
        )
    return decoded


def attachment_search_text(attachments: list[Any]) -> str:
    parts: list[str] = []
    for attachment in attachments:
        parts.extend(
            [
                getattr(attachment, "filename", "") or "",
                getattr(attachment, "description", "") or "",
                getattr(attachment, "ocr_text", "") or "",
                _metadata_search_text(getattr(attachment, "metadata_json", None) or getattr(attachment, "metadata", None) or {}),
            ]
        )
    return "\n".join(part for part in parts if part)


def _strip_data_uri(value: str, expected_mime: str) -> str:
    raw = value.strip()
    if not raw.startswith("data:"):
        return raw
    prefix, separator, body = raw.partition(",")
    if not separator:
        raise AttachmentValidationError("图片 data URI 缺少 base64 内容。")
    if ";base64" not in prefix:
        raise AttachmentValidationError("图片 data URI 必须使用 base64。")
    actual_mime = prefix.removeprefix("data:").split(";", 1)[0].lower()
    if actual_mime and actual_mime != expected_mime:
        raise AttachmentValidationError("图片 data URI MIME 与 mime_type 不一致。")
    return body.strip()


def _bytes_match_mime(value: bytes, mime_type: str) -> bool:
    if mime_type == "image/png":
        return value.startswith(b"\x89PNG\r\n\x1a\n")
    if mime_type == "image/jpeg":
        return value.startswith(b"\xff\xd8\xff")
    if mime_type == "image/gif":
        return value.startswith((b"GIF87a", b"GIF89a"))
    if mime_type == "image/webp":
        return len(value) >= 12 and value[:4] == b"RIFF" and value[8:12] == b"WEBP"
    return False


def _metadata_search_text(value: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in value.values():
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, list):
            parts.extend(str(entry) for entry in item if isinstance(entry, (str, int, float)))
        elif isinstance(item, (int, float, bool)):
            parts.append(str(item))
    return " ".join(parts)
