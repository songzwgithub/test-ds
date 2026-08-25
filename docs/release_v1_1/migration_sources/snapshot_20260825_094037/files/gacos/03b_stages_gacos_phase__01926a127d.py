from pathlib import Path

p = Path("pystamps/pipeline/stages.py")
text = p.read_text(encoding="utf-8")


def once(old, new, label):
    global text
    n = text.count(old)
    if n != 1:
        raise RuntimeError(
            f"{label}: expected 1 match, found {n}"
        )
    text = text.replace(old, new, 1)


# ============================================================
# 1. json import
# ============================================================

once(
    '''from pathlib import Path
import os
import shutil
''',
    '''from pathlib import Path
import json
import os
import shutil
''',
    "json import",
)


# ============================================================
# 2. phase-input helpers
# ============================================================

anchor = '''def _normalize_backend(name: str) -> str:
'''

helpers = r'''def _stage_phase_marker_path(
    dataset_root: Path,
    stage_id: int,
) -> Path:
    return (
        dataset_root
        / f"_pystamps_stage{stage_id}_phase_input.json"
    )


def _phase_input_signature(
    dataset_root: Path,
    phase_file: str,
) -> dict[str, object]:

    path = (
        dataset_root
        / phase_file
    )

    if not path.is_file():
        raise StageExecutionError(
            f"Stage phase input does not exist: {path}"
        )

    stat = path.stat()

    return {
        "phase_file": phase_file,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _phase_input_is_current(
    dataset_root: Path,
    stage_id: int,
    phase_file: str,
) -> bool:

    marker = _stage_phase_marker_path(
        dataset_root,
        stage_id,
    )

    # Backward compatibility:
    #
    # v1.0.0 Stage 7/8 outputs have no marker and were
    # produced from ordinary phuw2.mat. Treat those as
    # current while GACOS remains disabled.
    if not marker.is_file():
        return (
            phase_file
            == "phuw2.mat"
        )

    try:
        saved = json.loads(
            marker.read_text(
                encoding="utf-8"
            )
        )

        current = _phase_input_signature(
            dataset_root,
            phase_file,
        )

        return (
            saved.get("phase_file")
            == current["phase_file"]
            and int(
                saved.get("size", -1)
            )
            == current["size"]
            and int(
                saved.get(
                    "mtime_ns",
                    -1,
                )
            )
            == current["mtime_ns"]
        )

    except Exception:
        return False


def _write_phase_input_marker(
    dataset_root: Path,
    stage_id: int,
    phase_file: str,
) -> None:

    payload = _phase_input_signature(
        dataset_root,
        phase_file,
    )

    marker = _stage_phase_marker_path(
        dataset_root,
        stage_id,
    )

    tmp = marker.with_suffix(
        marker.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    tmp.replace(marker)


def _resolve_stage78_phase_file(
    dataset_root: Path,
    context: PipelineContext,
) -> str:

    gacos = context.run_config.gacos

    if not bool(gacos.enabled):
        return "phuw2.mat"

    # Dry-run must not create atmospheric products.
    if context.dry_run:
        return "phuw2_gacos.mat"

    # Lazy import: ordinary non-GACOS runs do not enter
    # the atmospheric correction module at all.
    from pystamps.pipeline.gacos_correction import (
        ensure_gacos_corrected_phuw,
    )

    corrected = ensure_gacos_corrected_phuw(
        dataset_root,
        gacos,
    )

    if corrected.parent != dataset_root.resolve():
        raise StageExecutionError(
            "GACOS corrected phase was created outside "
            "the dataset root"
        )

    return corrected.name


def _normalize_backend(name: str) -> str:
'''

once(
    anchor,
    helpers,
    "phase helpers",
)


# ============================================================
# 3. Resolve Stage7/8 phase BEFORE existing-output skip
# ============================================================

old = '''    artifact = dataset_root / expected
    bundle = MERGED_STAGE_BUNDLES.get(stage.stage_id, [expected])
    if not force_run and all((dataset_root / filename).exists() for filename in bundle):
        return StageResult(stage.stage_id, "merged", dataset_root.name, "skipped_existing", f"{expected} present")
'''

new = '''    phase_file = "phuw2.mat"

    if stage.stage_id in {7, 8}:
        phase_file = _resolve_stage78_phase_file(
            dataset_root,
            context,
        )

        if (
            not context.dry_run
            and not _phase_input_is_current(
                dataset_root,
                stage.stage_id,
                phase_file,
            )
        ):
            force_run = True

        # Direct Stage-8 execution must not use an SCLA
        # generated from a different phase input.
        if (
            stage.stage_id == 8
            and not context.dry_run
            and not _phase_input_is_current(
                dataset_root,
                7,
                phase_file,
            )
        ):
            force_run = True

    artifact = dataset_root / expected
    bundle = MERGED_STAGE_BUNDLES.get(stage.stage_id, [expected])

    if (
        not force_run
        and all(
            (dataset_root / filename).exists()
            for filename in bundle
        )
    ):
        return StageResult(
            stage.stage_id,
            "merged",
            dataset_root.name,
            "skipped_existing",
            f"{expected} present",
        )
'''

once(
    old,
    new,
    "phase-aware skip logic",
)


# ============================================================
# 4. Stage 7 passes selected phase_file
# ============================================================

once(
    '''                triangle_path=context.run_config.tools.triangle,
            )
        elif stage.stage_id == 8:
''',
    '''                triangle_path=context.run_config.tools.triangle,
                phase_file=phase_file,
            )

            _write_phase_input_marker(
                dataset_root,
                7,
                phase_file,
            )

        elif stage.stage_id == 8:
''',
    "stage7 phase_file",
)


# ============================================================
# 5. Stage 8 direct-run safeguard + phase_file
# ============================================================

old = '''        elif stage.stage_id == 8:
            details = stage8_filter_scn(
                dataset_root,
'''

new = '''        elif stage.stage_id == 8:

            # If Stage 8 is launched directly and Stage 7
            # belongs to another phase input, rebuild SCLA
            # first using the selected phase.
            if not _phase_input_is_current(
                dataset_root,
                7,
                phase_file,
            ):
                stage7_calc_scla(
                    dataset_root,
                    backend=_kernel_backend_for_name(
                        context,
                        "stage7_scla",
                        context.run_config.runtime.backend,
                    ),
                    chunk_ps=context.run_config.runtime.stage7_chunk_ps,
                    enable_mat_cache=context.run_config.runtime.enable_mat_stage_cache,
                    io_workers=context.run_config.runtime.io_workers,
                    triangle_path=context.run_config.tools.triangle,
                    phase_file=phase_file,
                )

                _write_phase_input_marker(
                    dataset_root,
                    7,
                    phase_file,
                )

            details = stage8_filter_scn(
                dataset_root,
'''

once(
    old,
    new,
    "stage8 safeguard",
)

once(
    '''                snaphu_path=context.run_config.tools.snaphu,
            )
        else:
''',
    '''                snaphu_path=context.run_config.tools.snaphu,
                phase_file=phase_file,
            )

            _write_phase_input_marker(
                dataset_root,
                8,
                phase_file,
            )

        else:
''',
    "stage8 phase_file",
)


p.write_text(
    text,
    encoding="utf-8",
)

print("03b STAGES GACOS ROUTING: PASS")
