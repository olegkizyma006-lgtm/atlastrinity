import os
import sys
from datetime import UTC, datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_datetime_fix():
    print("Testing datetime fix logic...")
    # Simulation of the fix
    last_consolidation = datetime.now(UTC)
    # This should NOT crash if 'now' is also UTC
    now_utc = datetime.now(UTC)
    diff = now_utc - last_consolidation
    print(f"UTC Difference: {diff}")

    # This IS what would crash (naive - aware)
    now_naive = datetime.now()
    try:
        _ = now_naive - last_consolidation
        print("ERROR: Naive - Aware did not crash (unexpected)")
    except TypeError as e:
        print(f"Confirmed expected crash with naive datetime: {e}")


def sanitize_metadata(metadata: dict) -> dict:
    """Sanitize metadata for ChromaDB (no lists allowed)"""
    sanitized = {}
    for k, v in metadata.items():
        if isinstance(v, list):
            sanitized[k] = ", ".join(map(str, v))
        else:
            sanitized[k] = v
    return sanitized


def test_metadata_sanitization():
    print("\nTesting metadata sanitization...")
    bad_metadata = {
        "entities": ["ТзОВ Кардинал-Клінінг", "м. Київ", "вул. Голосіївська"],
        "session_id": "test_session",
        "count": 5,
    }

    sanitized = sanitize_metadata(bad_metadata)
    print(f"Original: {bad_metadata}")
    print(f"Sanitized: {sanitized}")

    assert isinstance(sanitized["entities"], str)
    assert sanitized["entities"] == "ТзОВ Кардинал-Клінінг, м. Київ, вул. Голосіївська"
    print("Metadata sanitization SUCCESS")


if __name__ == "__main__":
    test_datetime_fix()
    test_metadata_sanitization()
