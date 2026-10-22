"""PAD evaluation harness utilities."""

from eval.harness.clip_loader import (
    ClipLoadError,
    LoadedPresentation,
    discover_frame_paths,
    frames_from_arrays,
    load_clip_frames,
    load_presentation,
    load_still_frame,
)
from eval.harness.outcome_mapper import (
    POLICY_FAIL_CLOSED_V1,
    VALID_OUTCOMES,
    classify_for_apcer,
    classify_for_bpcer,
    is_classified_as_live,
    map_outcome_fail_closed_v1,
    normalize_outcome,
    validate_outcome,
)
from eval.harness.settings_snapshot import (
    EnvironmentMetadata,
    build_settings_snapshot,
    capture_environment,
    merge_pad_settings,
    pad_settings_to_dict,
    read_settings_snapshot,
    write_settings_snapshot,
)

_LAZY_RUN_PAD_EVAL_EXPORTS = {
    "DatasetManifest",
    "EvalRunResult",
    "PadEvalConfig",
    "PresentationResult",
    "ProtocolSpec",
    "aggregate_metrics",
    "execute_presentation",
    "load_manifest_json",
    "load_protocol_yaml",
    "run_iso_lab",
    "run_live_sim_session",
    "run_pad_evaluation",
    "run_registration",
}


def __getattr__(name: str):
    """Lazy import run_pad_eval (requires PyYAML) when execution API is used."""
    if name in _LAZY_RUN_PAD_EVAL_EXPORTS:
        from eval.harness import run_pad_eval

        return getattr(run_pad_eval, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClipLoadError",
    "LoadedPresentation",
    "POLICY_FAIL_CLOSED_V1",
    "VALID_OUTCOMES",
    "EnvironmentMetadata",
    "build_settings_snapshot",
    "capture_environment",
    "classify_for_apcer",
    "classify_for_bpcer",
    "discover_frame_paths",
    "frames_from_arrays",
    "is_classified_as_live",
    "load_clip_frames",
    "load_presentation",
    "load_still_frame",
    "map_outcome_fail_closed_v1",
    "merge_pad_settings",
    "normalize_outcome",
    "pad_settings_to_dict",
    "read_settings_snapshot",
    "validate_outcome",
    "write_settings_snapshot",
    *_LAZY_RUN_PAD_EVAL_EXPORTS,
]
