"""
PAD evaluation harness — protocol-driven batch evaluation.

Loads protocol YAML and dataset manifests, runs pad_engine entrypoints,
aggregates E2 metrics, and writes run artifacts (no markdown reports).
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from eval.harness.clip_loader import ClipLoadError, LoadedPresentation, load_presentation
from eval.harness.outcome_mapper import (
    POLICY_FAIL_CLOSED_V1,
    normalize_outcome,
    validate_outcome,
)
from eval.harness.settings_snapshot import (
    build_settings_snapshot,
    merge_pad_settings,
    write_settings_snapshot,
)
from eval.metrics.apcer_bpcer import (
    InsufficientSamplesError,
    compute_apcer,
    compute_bpcer,
    to_percentage,
)
from eval.metrics.confusion import build_confusion_counts
from eval.metrics.latency_stats import LatencyStats, compute_latency_stats
from src.config import PADSettings, resolve_project_root
from src.pad_engine import (
    MSG_CHALLENGE_PROMPT,
    PadOutcome,
    PadResult,
    build_pad_frame,
    check_single_image_pad,
    create_pad_session,
    run_pad_gate,
)

GroundTruth = Literal["attack", "bona_fide"]
EntrypointName = Literal[
    "PadSession.evaluate_final",
    "check_single_image_pad",
    "live_sim_session",
]


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    """Parsed evaluation protocol."""

    protocol_id: str
    version: str
    evaluation_policy: str
    entrypoint: EntrypointName
    pad_settings: dict[str, Any]
    tier_gates: dict[str, Any]
    raw: dict[str, Any]

    @property
    def assurance_tier(self) -> str:
        return str(self.pad_settings.get("assurance_tier", "standard"))


@dataclass(frozen=True, slots=True)
class ManifestSample:
    """One presentation row from a dataset manifest."""

    sample_id: str
    pai_id: str
    ground_truth: GroundTruth
    subject_id: str
    path: str
    presentation_type: Literal["clip", "still"]
    frame_count: int | None = None


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Parsed dataset manifest."""

    manifest_version: str
    collection_id: str
    samples: tuple[ManifestSample, ...]
    description: str | None = None


@dataclass(frozen=True, slots=True)
class PresentationResult:
    """Outcome for one evaluated presentation."""

    sample_id: str
    pai_id: str
    ground_truth: GroundTruth
    subject_id: str
    predicted_outcome: str
    normalized_outcome: str
    reason_code: str
    model_id: str
    model_version: str
    latency_ms: float
    frames_analyzed: int
    entrypoint: str
    source_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TierGateResult:
    apcer: str
    bpcer: str
    latency: str
    overall: str


@dataclass(frozen=True, slots=True)
class EvalMetricsBundle:
    """Aggregated E2 metrics for a run."""

    apcer: float | None
    bpcer: float | None
    apcer_pct: float | None
    bpcer_pct: float | None
    apcer_by_pai: dict[str, float]
    confusion: dict[str, int]
    latency: LatencyStats | None
    tier_gate: TierGateResult | None
    n_attack: int
    n_bona_fide: int


@dataclass(frozen=True, slots=True)
class EvalRunResult:
    """Complete harness run output."""

    run_id: str
    protocol: ProtocolSpec
    manifest: DatasetManifest
    pad_settings: PADSettings
    presentations: tuple[PresentationResult, ...]
    metrics: EvalMetricsBundle
    output_dir: Path
    settings_snapshot_path: Path
    results_jsonl_path: Path
    summary_json_path: Path


@dataclass
class PadEvalConfig:
    """Configuration for a single harness execution."""

    protocol_path: Path
    manifest_path: Path
    collection_root: Path
    output_dir: Path
    run_id: str | None = None
    max_frames_per_clip: int | None = None
    skip_missing_media: bool = False


def load_protocol_yaml(protocol_path: Path) -> ProtocolSpec:
    """Load and parse a protocol YAML file."""
    raw = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid protocol YAML: {protocol_path}")

    protocol_id = str(raw.get("protocol_id", ""))
    if not protocol_id:
        raise ValueError(f"protocol_id missing in {protocol_path}")

    sut = raw.get("system_under_test", {})
    entrypoint = _resolve_entrypoint(protocol_id, sut)

    return ProtocolSpec(
        protocol_id=protocol_id,
        version=str(raw.get("version", "0.0.0")),
        evaluation_policy=str(raw.get("evaluation_policy", POLICY_FAIL_CLOSED_V1)),
        entrypoint=entrypoint,
        pad_settings=dict(raw.get("pad_settings") or {}),
        tier_gates=dict(raw.get("tier_gates") or {}),
        raw=raw,
    )


def _resolve_entrypoint(protocol_id: str, sut: Any) -> EntrypointName:
    if isinstance(sut, dict) and sut.get("entrypoint"):
        name = str(sut["entrypoint"])
        if name in (
            "PadSession.evaluate_final",
            "check_single_image_pad",
            "live_sim_session",
        ):
            return name  # type: ignore[return-value]
        raise ValueError(f"Unsupported protocol entrypoint: {name}")

    if protocol_id == "PAD-REG-v1":
        return "check_single_image_pad"
    if protocol_id == "PAD-LIVE-SIM-v1":
        return "live_sim_session"
    return "PadSession.evaluate_final"


def load_manifest_json(manifest_path: Path) -> DatasetManifest:
    """Load and validate a dataset manifest JSON file."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest JSON: {manifest_path}")

    manifest_version = str(data.get("manifest_version", ""))
    collection_id = str(data.get("collection_id", ""))
    if not manifest_version or not collection_id:
        raise ValueError(
            f"manifest_version and collection_id required in {manifest_path}"
        )

    raw_samples = data.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"manifest must include non-empty samples: {manifest_path}")

    samples: list[ManifestSample] = []
    for index, item in enumerate(raw_samples):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid sample at index {index} in {manifest_path}")
        sample = _parse_manifest_sample(item)
        _validate_sample_consistency(sample)
        samples.append(sample)

    return DatasetManifest(
        manifest_version=manifest_version,
        collection_id=collection_id,
        samples=tuple(samples),
        description=data.get("description"),
    )


def _parse_manifest_sample(item: dict[str, Any]) -> ManifestSample:
    required = ("sample_id", "pai_id", "ground_truth", "subject_id", "path", "presentation_type")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"Manifest sample missing fields: {missing}")

    ground_truth = str(item["ground_truth"])
    if ground_truth not in ("attack", "bona_fide"):
        raise ValueError(f"Invalid ground_truth: {ground_truth!r}")

    presentation_type = str(item["presentation_type"])
    if presentation_type not in ("clip", "still"):
        raise ValueError(f"Invalid presentation_type: {presentation_type!r}")

    frame_count = item.get("frame_count")
    return ManifestSample(
        sample_id=str(item["sample_id"]),
        pai_id=str(item["pai_id"]),
        ground_truth=ground_truth,  # type: ignore[arg-type]
        subject_id=str(item["subject_id"]),
        path=str(item["path"]),
        presentation_type=presentation_type,  # type: ignore[arg-type]
        frame_count=int(frame_count) if frame_count is not None else None,
    )


def _validate_sample_consistency(sample: ManifestSample) -> None:
    if sample.pai_id == "BF" and sample.ground_truth != "bona_fide":
        raise ValueError(
            f"Sample {sample.sample_id}: pai_id BF requires ground_truth bona_fide"
        )
    if sample.pai_id.startswith("PA-") and sample.ground_truth != "attack":
        raise ValueError(
            f"Sample {sample.sample_id}: {sample.pai_id} requires ground_truth attack"
        )


def run_iso_lab(
    presentation: LoadedPresentation,
    pad_settings: PADSettings,
) -> PadResult:
    """Batch clip evaluation via run_pad_gate (ISO lab path)."""
    frames = [
        build_pad_frame(image=image, frame_index=index)
        for index, image in enumerate(presentation.frames, start=1)
    ]
    return run_pad_gate(frames, settings=pad_settings)


def run_live_sim_session(
    presentation: LoadedPresentation,
    pad_settings: PADSettings,
    *,
    max_evaluate_calls: int = 10,
) -> PadResult:
    """
    Feed frames through PadSession like recognize_live().

    Re-calls evaluate_final when an active challenge is pending.
    """
    session = create_pad_session(settings=pad_settings)
    frame_index = 1
    last_result: PadResult | None = None

    for _ in range(max_evaluate_calls):
        for image in presentation.frames:
            session.add_frame(
                build_pad_frame(image=image, frame_index=frame_index)
            )
            frame_index += 1

        last_result = session.evaluate_final()
        if last_result.outcome != PadOutcome.INCONCLUSIVE:
            break
        if last_result.user_message_key != MSG_CHALLENGE_PROMPT:
            break

    assert last_result is not None
    return last_result


def run_registration(
    presentation: LoadedPresentation,
    pad_settings: PADSettings,
) -> PadResult:
    """Registration path — single still via check_single_image_pad."""
    if not presentation.frames:
        raise ClipLoadError("Registration presentation has no frames.")
    return check_single_image_pad(presentation.frames[0], settings=pad_settings)


def execute_presentation(
    sample: ManifestSample,
    presentation: LoadedPresentation,
    protocol: ProtocolSpec,
    pad_settings: PADSettings,
) -> PresentationResult:
    """Run pad_engine for one loaded presentation."""
    if protocol.entrypoint == "check_single_image_pad":
        pad_result = run_registration(presentation, pad_settings)
        entrypoint_label = "check_single_image_pad"
    elif protocol.entrypoint == "live_sim_session":
        max_calls = int(
            protocol.raw.get("active_challenge", {}).get(
                "max_evaluate_calls_per_presentation", 10
            )
        )
        pad_result = run_live_sim_session(
            presentation,
            pad_settings,
            max_evaluate_calls=max_calls,
        )
        entrypoint_label = "live_sim_session"
    else:
        pad_result = run_iso_lab(presentation, pad_settings)
        entrypoint_label = "PadSession.evaluate_final"

    predicted = pad_result.outcome.value
    validate_outcome(predicted)
    policy = protocol.evaluation_policy
    if policy != POLICY_FAIL_CLOSED_V1:
        raise ValueError(f"Unsupported evaluation policy: {policy}")

    return PresentationResult(
        sample_id=sample.sample_id,
        pai_id=sample.pai_id,
        ground_truth=sample.ground_truth,
        subject_id=sample.subject_id,
        predicted_outcome=predicted,
        normalized_outcome=normalize_outcome(predicted),
        reason_code=pad_result.reason_code,
        model_id=pad_result.model_id,
        model_version=pad_result.model_version,
        latency_ms=pad_result.latency_ms,
        frames_analyzed=pad_result.frames_analyzed,
        entrypoint=entrypoint_label,
        source_paths=tuple(str(path) for path in presentation.source_paths),
    )


def aggregate_metrics(
    presentations: Sequence[PresentationResult],
    protocol: ProtocolSpec,
) -> EvalMetricsBundle:
    """Compute E2 metrics from presentation results."""
    attack_outcomes = [
        row.predicted_outcome
        for row in presentations
        if row.ground_truth == "attack"
    ]
    bona_fide_outcomes = [
        row.predicted_outcome
        for row in presentations
        if row.ground_truth == "bona_fide"
    ]

    apcer: float | None
    bpcer: float | None
    try:
        apcer = compute_apcer(attack_outcomes) if attack_outcomes else None
    except InsufficientSamplesError:
        apcer = None
    try:
        bpcer = compute_bpcer(bona_fide_outcomes) if bona_fide_outcomes else None
    except InsufficientSamplesError:
        bpcer = None

    apcer_by_pai: dict[str, float] = {}
    pai_ids = sorted({row.pai_id for row in presentations if row.ground_truth == "attack"})
    for pai_id in pai_ids:
        subset = [
            row.predicted_outcome
            for row in presentations
            if row.ground_truth == "attack" and row.pai_id == pai_id
        ]
        if subset:
            apcer_by_pai[pai_id] = compute_apcer(subset)

    labeled = [(row.ground_truth, row.predicted_outcome) for row in presentations]
    confusion_counts = build_confusion_counts(labeled)
    confusion = {
        "tp": confusion_counts.tp,
        "tn": confusion_counts.tn,
        "fp": confusion_counts.fp,
        "fn": confusion_counts.fn,
    }

    latencies = [row.latency_ms for row in presentations]
    latency_stats = compute_latency_stats(latencies) if latencies else None

    tier_gate = _evaluate_tier_gates(
        protocol=protocol,
        apcer=apcer,
        bpcer=bpcer,
        latency_stats=latency_stats,
    )

    return EvalMetricsBundle(
        apcer=apcer,
        bpcer=bpcer,
        apcer_pct=to_percentage(apcer) if apcer is not None else None,
        bpcer_pct=to_percentage(bpcer) if bpcer is not None else None,
        apcer_by_pai=apcer_by_pai,
        confusion=confusion,
        latency=latency_stats,
        tier_gate=tier_gate,
        n_attack=len(attack_outcomes),
        n_bona_fide=len(bona_fide_outcomes),
    )


def _evaluate_tier_gates(
    *,
    protocol: ProtocolSpec,
    apcer: float | None,
    bpcer: float | None,
    latency_stats: LatencyStats | None,
) -> TierGateResult | None:
    tier_name = protocol.assurance_tier
    gates = protocol.tier_gates.get(tier_name)
    if not gates or apcer is None or bpcer is None:
        return None

    apcer_pass = apcer <= float(gates.get("apcer_max_pooled", 1.0))
    bpcer_pass = bpcer <= float(gates.get("bpcer_max", 1.0))

    latency_pass = "skipped"
    latency_limit = gates.get("latency_ms_p95_max")
    if latency_stats is not None and latency_limit is not None:
        latency_pass = (
            "pass" if latency_stats.p95 <= float(latency_limit) else "fail"
        )

    overall = "pass" if apcer_pass and bpcer_pass and latency_pass != "fail" else "fail"
    return TierGateResult(
        apcer="pass" if apcer_pass else "fail",
        bpcer="pass" if bpcer_pass else "fail",
        latency=latency_pass,
        overall=overall,
    )


def _write_results_jsonl(path: Path, presentations: Sequence[PresentationResult]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in presentations:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    return path


def _write_summary_json(
    path: Path,
    *,
    run_id: str,
    protocol: ProtocolSpec,
    manifest: DatasetManifest,
    pad_settings: PADSettings,
    metrics: EvalMetricsBundle,
) -> Path:
    summary: dict[str, Any] = {
        "run_id": run_id,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.version,
        "collection_id": manifest.collection_id,
        "manifest_version": manifest.manifest_version,
        "model_id": pad_settings.passive_model_id,
        "evaluation_policy": protocol.evaluation_policy,
        "assurance_tier": protocol.assurance_tier,
        "n_attack": metrics.n_attack,
        "n_bona_fide": metrics.n_bona_fide,
        "apcer_overall": metrics.apcer,
        "bpcer_overall": metrics.bpcer,
        "apcer_overall_pct": metrics.apcer_pct,
        "bpcer_overall_pct": metrics.bpcer_pct,
        "apcer_by_pai": metrics.apcer_by_pai,
        "confusion": metrics.confusion,
    }
    if metrics.latency is not None:
        summary["latency_ms"] = asdict(metrics.latency)
    if metrics.tier_gate is not None:
        summary["tier_gate"] = asdict(metrics.tier_gate)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_pad_evaluation(config: PadEvalConfig) -> EvalRunResult:
    """Execute a full PAD evaluation run."""
    protocol = load_protocol_yaml(config.protocol_path)
    manifest = load_manifest_json(config.manifest_path)
    pad_settings = merge_pad_settings(protocol.pad_settings)

    run_id = config.run_id or _default_run_id()
    output_dir = config.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    presentation_results: list[PresentationResult] = []
    for sample in manifest.samples:
        try:
            loaded = load_presentation(
                config.collection_root,
                sample.path,
                sample.presentation_type,
                max_frames=config.max_frames_per_clip,
            )
        except ClipLoadError as error:
            if config.skip_missing_media:
                continue
            raise

        if sample.frame_count is not None and loaded.frame_count < sample.frame_count:
            if not config.skip_missing_media:
                raise ClipLoadError(
                    f"Sample {sample.sample_id}: expected {sample.frame_count} "
                    f"frames, found {loaded.frame_count}"
                )

        result = execute_presentation(sample, loaded, protocol, pad_settings)
        presentation_results.append(result)

    if not presentation_results:
        raise ValueError(
            "No presentations evaluated. Check collection_root and skip_missing_media."
        )

    metrics = aggregate_metrics(presentation_results, protocol)

    snapshot = build_settings_snapshot(
        run_id=run_id,
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.version,
        evaluation_policy=protocol.evaluation_policy,
        pad_settings=pad_settings,
        collection_id=manifest.collection_id,
        manifest_version=manifest.manifest_version,
        extra={
            "protocol_path": str(config.protocol_path),
            "manifest_path": str(config.manifest_path),
            "collection_root": str(config.collection_root),
            "entrypoint": protocol.entrypoint,
        },
    )
    snapshot_path = write_settings_snapshot(
        output_dir / "settings_snapshot.json",
        snapshot,
    )
    results_path = _write_results_jsonl(
        output_dir / "results.jsonl",
        presentation_results,
    )
    summary_path = _write_summary_json(
        output_dir / "summary.json",
        run_id=run_id,
        protocol=protocol,
        manifest=manifest,
        pad_settings=pad_settings,
        metrics=metrics,
    )

    return EvalRunResult(
        run_id=run_id,
        protocol=protocol,
        manifest=manifest,
        pad_settings=pad_settings,
        presentations=tuple(presentation_results),
        metrics=metrics,
        output_dir=output_dir,
        settings_snapshot_path=snapshot_path,
        results_jsonl_path=results_path,
        summary_json_path=summary_path,
    )


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"pad_eval_{timestamp}_{uuid.uuid4().hex[:8]}"


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI argument parser for harness execution."""
    parser = argparse.ArgumentParser(
        description="Run Face Lock PAD evaluation harness (Phase E3).",
    )
    root = resolve_project_root()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "eval" / "protocols" / "pad_iso_lab_v1.yaml",
        help="Path to protocol YAML file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to dataset manifest JSON.",
    )
    parser.add_argument(
        "--collection-root",
        type=Path,
        required=True,
        help="Root directory containing manifest-relative media paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "eval" / "reports" / "runs",
        help="Directory for run artifacts.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional run identifier.")
    parser.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=None,
        help="Optional cap on frames loaded per clip.",
    )
    parser.add_argument(
        "--skip-missing-media",
        action="store_true",
        help="Skip samples whose media cannot be loaded.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    config = PadEvalConfig(
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        collection_root=args.collection_root,
        output_dir=args.output_dir,
        run_id=args.run_id,
        max_frames_per_clip=args.max_frames_per_clip,
        skip_missing_media=args.skip_missing_media,
    )
    result = run_pad_evaluation(config)
    print(
        f"PAD evaluation complete: run_id={result.run_id} "
        f"presentations={len(result.presentations)} "
        f"apcer={result.metrics.apcer} bpcer={result.metrics.bpcer} "
        f"output={result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
