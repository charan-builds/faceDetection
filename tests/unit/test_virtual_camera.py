"""
Unit tests for virtual camera detection and enforcement in src.capture.

No real webcam required. cv2.VideoCapture is mocked throughout.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.capture import (
    VIRTUAL_CAMERA_SIGNATURES,
    VirtualCameraError,
    check_virtual_camera_policy,
    is_virtual_camera,
    open_webcam,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cap(backend_name: str = "", opened: bool = True) -> MagicMock:
    """Build a mock VideoCapture with a controllable backend name."""
    cap = MagicMock()
    cap.isOpened.return_value = opened
    cap.getBackendName.return_value = backend_name
    return cap


# ---------------------------------------------------------------------------
# Tests: is_virtual_camera
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsVirtualCamera:
    def test_returns_false_for_empty_backend(self) -> None:
        cap = _make_cap(backend_name="")
        assert is_virtual_camera(cap) is False

    def test_returns_false_for_physical_camera(self) -> None:
        cap = _make_cap(backend_name="DSHOW")  # DirectShow — physical device
        # "dshow" doesn't match any virtual signature
        assert is_virtual_camera(cap) is False

    def test_detects_obs_virtual_camera(self) -> None:
        cap = _make_cap(backend_name="OBS Virtual Camera")
        assert is_virtual_camera(cap) is True

    def test_detects_droidcam(self) -> None:
        cap = _make_cap(backend_name="DroidCam Source 3")
        assert is_virtual_camera(cap) is True

    def test_detects_manycam(self) -> None:
        cap = _make_cap(backend_name="ManyCam Virtual Webcam")
        assert is_virtual_camera(cap) is True

    def test_detects_snap_camera(self) -> None:
        cap = _make_cap(backend_name="Snap Camera")
        assert is_virtual_camera(cap) is True

    def test_detects_generic_virtual(self) -> None:
        cap = _make_cap(backend_name="Some Virtual Device")
        assert is_virtual_camera(cap) is True

    def test_case_insensitive(self) -> None:
        cap = _make_cap(backend_name="OBS VIRTUAL CAMERA")
        assert is_virtual_camera(cap) is True

    def test_getbackendname_not_available(self) -> None:
        """Old OpenCV without getBackendName — should not crash, returns False."""
        cap = MagicMock(spec=["isOpened", "read", "release"])  # no getBackendName
        assert is_virtual_camera(cap) is False

    @pytest.mark.parametrize("sig", VIRTUAL_CAMERA_SIGNATURES)
    def test_all_signatures_detected(self, sig: str) -> None:
        cap = _make_cap(backend_name=f"prefix {sig} suffix")
        assert is_virtual_camera(cap) is True


# ---------------------------------------------------------------------------
# Tests: check_virtual_camera_policy
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckVirtualCameraPolicy:
    def test_allow_virtual_true_does_nothing(self) -> None:
        cap = _make_cap(backend_name="OBS Virtual Camera")
        # Must not raise when allowed
        check_virtual_camera_policy(cap, camera_index=0, allow_virtual_camera=True)

    def test_physical_camera_always_passes(self) -> None:
        cap = _make_cap(backend_name="DSHOW")
        check_virtual_camera_policy(cap, camera_index=0, allow_virtual_camera=False)
        cap.release.assert_not_called()

    def test_virtual_camera_rejected_when_disallowed(self) -> None:
        cap = _make_cap(backend_name="OBS Virtual Camera")
        with pytest.raises(VirtualCameraError, match="Virtual camera detected"):
            check_virtual_camera_policy(cap, camera_index=0, allow_virtual_camera=False)

    def test_virtual_camera_rejection_releases_cap(self) -> None:
        cap = _make_cap(backend_name="ManyCam Virtual Webcam")
        try:
            check_virtual_camera_policy(cap, camera_index=0, allow_virtual_camera=False)
        except VirtualCameraError:
            pass
        cap.release.assert_called_once()

    def test_error_message_includes_camera_index(self) -> None:
        cap = _make_cap(backend_name="DroidCam Source")
        with pytest.raises(VirtualCameraError) as exc_info:
            check_virtual_camera_policy(cap, camera_index=2, allow_virtual_camera=False)
        assert "index 2" in str(exc_info.value)

    def test_error_message_includes_backend_name(self) -> None:
        cap = _make_cap(backend_name="SomeCoolVirtualCam")
        with pytest.raises(VirtualCameraError) as exc_info:
            check_virtual_camera_policy(cap, camera_index=0, allow_virtual_camera=False)
        assert "somecoolvirtualcam" in str(exc_info.value).lower()

    def test_error_is_runtime_error_subclass(self) -> None:
        assert issubclass(VirtualCameraError, RuntimeError)


# ---------------------------------------------------------------------------
# Tests: open_webcam integration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOpenWebcamVirtualPolicy:
    def _mock_capture(self, backend_name: str = "DSHOW") -> MagicMock:
        cap = _make_cap(backend_name=backend_name, opened=True)
        return cap

    def test_physical_camera_opens_when_virtual_disallowed(self) -> None:
        cap = self._mock_capture(backend_name="DSHOW")
        with patch("src.capture.cv2.VideoCapture", return_value=cap):
            result = open_webcam(0, allow_virtual_camera=False)
        assert result is cap

    def test_virtual_camera_raises_when_disallowed(self) -> None:
        cap = self._mock_capture(backend_name="OBS Virtual Camera")
        with patch("src.capture.cv2.VideoCapture", return_value=cap):
            with pytest.raises(VirtualCameraError):
                open_webcam(0, allow_virtual_camera=False)

    def test_virtual_camera_opens_when_allowed(self) -> None:
        cap = self._mock_capture(backend_name="OBS Virtual Camera")
        with patch("src.capture.cv2.VideoCapture", return_value=cap):
            result = open_webcam(0, allow_virtual_camera=True)
        assert result is cap

    def test_camera_not_opened_raises_runtime_error(self) -> None:
        cap = _make_cap(backend_name="DSHOW", opened=False)
        with patch("src.capture.cv2.VideoCapture", return_value=cap):
            with pytest.raises(RuntimeError, match="Could not open webcam"):
                open_webcam(0, allow_virtual_camera=False)

    def test_default_policy_from_pad_settings(self) -> None:
        """Without explicit override, allow_virtual_camera reads from PAD_SETTINGS."""
        from dataclasses import replace
        import src.config as config_module
        import src.capture as capture_module

        cap = self._mock_capture(backend_name="OBS Virtual Camera")

        # Patch PAD_SETTINGS to disallow virtual cameras
        strict = replace(config_module.PAD_SETTINGS, allow_virtual_camera=False)
        with (
            patch("src.capture.cv2.VideoCapture", return_value=cap),
            patch("src.capture.PAD_SETTINGS", strict),
        ):
            with pytest.raises(VirtualCameraError):
                open_webcam(0)  # no explicit override → uses PAD_SETTINGS

    def test_explicit_allow_overrides_pad_settings(self) -> None:
        """allow_virtual_camera=True overrides a strict PAD_SETTINGS."""
        from dataclasses import replace
        import src.config as config_module

        cap = self._mock_capture(backend_name="ManyCam Virtual Webcam")
        strict = replace(config_module.PAD_SETTINGS, allow_virtual_camera=False)

        with (
            patch("src.capture.cv2.VideoCapture", return_value=cap),
            patch("src.capture.PAD_SETTINGS", strict),
        ):
            # Explicit True should override the strict config
            result = open_webcam(0, allow_virtual_camera=True)
        assert result is cap
