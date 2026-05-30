"""
Fixtures shared by recognition integration tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import pytest

# Bright synthetic BGR frame (well above default brightness threshold).
BRIGHT_FRAME = np.full((48, 64, 3), 200, dtype=np.uint8)

# Dark synthetic frame (below default brightness threshold).
DARK_FRAME = np.full((48, 64, 3), 10, dtype=np.uint8)

# Small deterministic "embedding" vectors (not real face vectors).
EMBEDDING_ALICE = [1.0, 0.0, 0.0]
EMBEDDING_STRANGER = [0.0, 1.0, 0.0]


@pytest.fixture
def bright_frame() -> np.ndarray:
    return BRIGHT_FRAME.copy()


@pytest.fixture
def dark_frame() -> np.ndarray:
    return DARK_FRAME.copy()


@pytest.fixture
def alice_record() -> dict[str, Any]:
    return {
        "user_name": "alice",
        "model_name": "test-model",
        "embeddings": [EMBEDDING_ALICE],
    }


@pytest.fixture
def trusted_records_alice(alice_record: dict[str, Any]) -> list[dict[str, Any]]:
    return [alice_record]


@pytest.fixture(autouse=True)
def pad_bypass_for_identity_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Legacy integration tests target identity matching; disable PAD by default.
    PAD-specific tests opt in via the pad_mock_live fixture.
    """
    from dataclasses import replace

    import src.config as config_module
    import src.recognize as recognize_module

    disabled = replace(config_module.PAD_SETTINGS, enabled=False)
    monkeypatch.setattr(config_module, "PAD_SETTINGS", disabled)
    monkeypatch.setattr(recognize_module, "PAD_SETTINGS", disabled)
