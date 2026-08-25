from pathlib import Path


# ============================================================
# 1. PipelineContext gets one internal per-run cache
# ============================================================

p = Path("pystamps/pipeline/types.py")
text = p.read_text(encoding="utf-8")

old = '''    dry_run: bool = False
    workflow_profile: WorkflowProfile = "default"
'''

new = '''    dry_run: bool = False
    workflow_profile: WorkflowProfile = "default"

    # Internal per-run cache. Stage 7 and Stage 8 must use
    # exactly the same selected phase artifact, especially
    # when gacos.rebuild=true.
    stage78_phase_file: str | None = field(
        default=None,
        init=False,
        repr=False,
    )
'''

n = text.count(old)

if n != 1:
    raise RuntimeError(
        f"PipelineContext insertion: expected 1 match, found {n}"
    )

text = text.replace(old, new, 1)

p.write_text(
    text.rstrip() + "\n",
    encoding="utf-8",
)

print("PipelineContext cache field: PASS")


# ============================================================
# 2. Cache _resolve_stage78_phase_file result
# ============================================================

p = Path("pystamps/pipeline/stages.py")
text = p.read_text(encoding="utf-8")

old = '''def _resolve_stage78_phase_file(
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
'''

new = '''def _resolve_stage78_phase_file(
    dataset_root: Path,
    context: PipelineContext,
) -> str:

    # Stage 7 and Stage 8 in one pipeline invocation must use
    # the exact same materialized phase product.
    if context.stage78_phase_file is not None:
        return context.stage78_phase_file

    gacos = context.run_config.gacos

    if not bool(gacos.enabled):
        selected = "phuw2.mat"

    elif context.dry_run:
        # Dry-run must not create atmospheric products.
        selected = "phuw2_gacos.mat"

    else:
        # Lazy import: ordinary non-GACOS runs never enter
        # the atmospheric correction module.
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

        selected = corrected.name

    context.stage78_phase_file = selected
    return selected
'''

n = text.count(old)

if n != 1:
    raise RuntimeError(
        f"phase resolver replacement: expected 1 match, found {n}"
    )

text = text.replace(old, new, 1)


# Reset cache at beginning of every run_pipeline() invocation.
old = '''def run_pipeline(context: PipelineContext) -> PipelineReport:
    dataset: DatasetLayout = discover_dataset(context.dataset_root)
'''

new = '''def run_pipeline(context: PipelineContext) -> PipelineReport:
    # The selected Stage7/8 phase product is valid only for
    # this invocation.
    context.stage78_phase_file = None

    dataset: DatasetLayout = discover_dataset(context.dataset_root)
'''

n = text.count(old)

if n != 1:
    raise RuntimeError(
        f"run_pipeline reset: expected 1 match, found {n}"
    )

text = text.replace(old, new, 1)

p.write_text(
    text.rstrip() + "\n",
    encoding="utf-8",
)

print("Stage7/8 phase cache: PASS")
print("03d SINGLE MATERIALIZATION: PASS")
