from aimemory.core.security import api_key_prefix, generate_api_key, hash_api_key, verify_api_key


def test_api_key_hash_and_verify() -> None:
    key = generate_api_key("aim_")
    digest = hash_api_key(key)

    assert key.startswith("aim_")
    assert len(digest) == 64
    assert verify_api_key(key, digest)
    assert not verify_api_key(f"{key}x", digest)


def test_api_key_prefix_is_stable() -> None:
    key = "aim_abcdefghijklmnopqrstuvwxyz"

    assert api_key_prefix(key) == "aim_abcdefghijkl"
