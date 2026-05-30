import base64
import hashlib
import hmac
import os


class AiConfigEncryptionError(ValueError):
    pass


def _derive_keys(secret: str, salt: bytes) -> tuple[bytes, bytes]:
    value = str(secret or "").strip()
    if not value:
        raise AiConfigEncryptionError("AI_CONFIG_ENCRYPTION_SECRET 未配置。")
    material = hashlib.pbkdf2_hmac("sha256", value.encode("utf-8"), salt, 200_000, dklen=64)
    return material[:32], material[32:]


def _keystream(key: bytes, nonce: bytes, size: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < size:
        block = hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        output.extend(block)
        counter += 1
    return bytes(output[:size])


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(left, right, strict=True))


def encrypt_secret(value: str, secret: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    salt = os.urandom(16)
    nonce = os.urandom(16)
    enc_key, mac_key = _derive_keys(secret, salt)
    plaintext = text.encode("utf-8")
    ciphertext = _xor_bytes(plaintext, _keystream(enc_key, nonce, len(plaintext)))
    payload = b"v1" + salt + nonce + ciphertext
    tag = hmac.new(mac_key, payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload + tag).decode("ascii")


def decrypt_secret(value: str | None, secret: str) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
    except Exception as exc:
        raise AiConfigEncryptionError("AI 配置密钥格式无效。") from exc
    if len(raw) < 2 + 16 + 16 + 32 or raw[:2] != b"v1":
        raise AiConfigEncryptionError("AI 配置密钥格式无效。")

    salt = raw[2:18]
    nonce = raw[18:34]
    ciphertext = raw[34:-32]
    tag = raw[-32:]
    enc_key, mac_key = _derive_keys(secret, salt)
    expected = hmac.new(mac_key, raw[:-32], hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise AiConfigEncryptionError("AI 配置密钥无法解密，请检查 AI_CONFIG_ENCRYPTION_SECRET。")
    try:
        return _xor_bytes(ciphertext, _keystream(enc_key, nonce, len(ciphertext))).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AiConfigEncryptionError("AI 配置密钥无法解密，请检查 AI_CONFIG_ENCRYPTION_SECRET。") from exc


def mask_secret(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= 10:
        return "***"
    return f"{text[:4]}...{text[-4:]}"
