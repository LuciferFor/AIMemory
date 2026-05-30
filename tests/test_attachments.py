import base64

import pytest

from aimemory.schemas.memory import MemoryAttachmentInput
from aimemory.services.attachments import AttachmentValidationError, decode_attachment_inputs


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _png_attachment(**overrides) -> MemoryAttachmentInput:
    values = {
        "filename": "note.png",
        "mime_type": "image/png",
        "data_base64": base64.b64encode(PNG_BYTES).decode("ascii"),
        "description": "一张测试图片",
        "metadata": {"tags": ["测试", "图片"]},
    }
    values.update(overrides)
    return MemoryAttachmentInput(**values)


def test_decode_attachment_accepts_valid_png() -> None:
    decoded = decode_attachment_inputs([_png_attachment()])

    assert decoded[0].filename == "note.png"
    assert decoded[0].mime_type == "image/png"
    assert decoded[0].size_bytes == len(PNG_BYTES)
    assert decoded[0].sha256
    assert "测试图片" in decoded[0].search_text


def test_decode_attachment_rejects_invalid_base64() -> None:
    with pytest.raises(AttachmentValidationError):
        decode_attachment_inputs([_png_attachment(data_base64="not-base64")])


def test_decode_attachment_rejects_mime_mismatch() -> None:
    with pytest.raises(AttachmentValidationError):
        decode_attachment_inputs([_png_attachment(mime_type="image/jpeg")])


def test_decode_attachment_rejects_too_many_files() -> None:
    with pytest.raises(AttachmentValidationError):
        decode_attachment_inputs([_png_attachment(filename=f"{index}.png") for index in range(6)])
