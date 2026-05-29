import hashlib
import hmac
import secrets


def generate_api_key(prefix: str = "aim_") -> str:
    return f"{prefix}{secrets.token_urlsafe(32)}"


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def api_key_prefix(raw_key: str, length: int = 16) -> str:
    return raw_key[:length]


def verify_api_key(raw_key: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(raw_key), stored_hash)
