#!/usr/bin/env python3
from __future__ import annotations

import ast
import datetime as dt
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PYSTAMPS_ROOT", "/home/ubuntu/software/pystamps-main")).expanduser().resolve()
DATASET = Path(os.environ.get(
    "PYSTAMPS_DATASET",
    "/mnt/vol-gdc28n1r/insar/cangzhou_P69/pystamps_sbas_ps_optimized",
)).expanduser().resolve()
HERE = Path(__file__).resolve().parent


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if old not in source:
        raise RuntimeError(f"Cannot find patch target: {label}")
    return source.replace(old, new, 1)


def backup_sources() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = ROOT / "_archive" / f"production_stage78_v6_5_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for path in (
        ROOT / "pystamps/pipeline/stage7_sbas.py",
        ROOT / "pystamps/pipeline/stage8_sbas.py",
        ROOT / "pystamps/prep/gamma_stage1.py",
        ROOT / "run_gamma_sbas_ps_optimized.py",
    ):
        if path.exists():
            shutil.copy2(path, backup / path.name)
    print("source backup:", backup)
    return backup


def install_helper() -> None:
    target = ROOT / "pystamps/pipeline/sbas_production.py"
    shutil.copy2(HERE / "sbas_production.py", target)
    print("installed:", target)


def patch_gamma_stage1() -> None:
    path = ROOT / "pystamps/prep/gamma_stage1.py"
    source = path.read_text(encoding="utf-8")

    if "sbas_deramp_cell_m:" not in source:
        anchor = "    reference_radius_m: float | None = None\n"
        fields = (
            "\n"
            "    # Production SBAS Stage7/8 settings.\n"
            "    sbas_deramp_mode: str = \"robust_huber_balanced\"\n"
            "    sbas_deramp_cell_m: float = 2000.0\n"
            "    sbas_deramp_anchors_per_cell: int = 8\n"
            "    sbas_deramp_huber_delta: float = 1.345\n"
            "    sbas_deramp_huber_iterations: int = 5\n"
            "    sbas_stage8_use_scla: bool = False\n"
            "    scn_time_win: float = 365.0\n"
            "    scn_wavelength: float = 100.0\n"
        )
        source = replace_once(source, anchor, anchor + fields, "GammaStage1Config production fields")

    if 'parameters["sbas_deramp_mode"]' not in source:
        anchor = '        parameters["ref_radius_m"] = float(config.reference_radius_m)\n'
        addition = (
            "\n"
            '    parameters["sbas_deramp_mode"] = str(config.sbas_deramp_mode)\n'
            '    parameters["sbas_deramp_cell_m"] = float(config.sbas_deramp_cell_m)\n'
            '    parameters["sbas_deramp_anchors_per_cell"] = int(config.sbas_deramp_anchors_per_cell)\n'
            '    parameters["sbas_deramp_huber_delta"] = float(config.sbas_deramp_huber_delta)\n'
            '    parameters["sbas_deramp_huber_iterations"] = int(config.sbas_deramp_huber_iterations)\n'
            '    parameters["sbas_stage8_use_scla"] = ("y" if config.sbas_stage8_use_scla else "n")\n'
            '    parameters["scn_time_win"] = float(config.scn_time_win)\n'
            '    parameters["scn_wavelength"] = float(config.scn_wavelength)\n'
        )
        source = replace_once(source, anchor, anchor + addition, "parms.mat production fields")

    ast.parse(source, filename=str(path))
    path.write_text(source, encoding="utf-8")
    print("patched:", path)


def patch_runner() -> None:
    path = ROOT / "run_gamma_sbas_ps_optimized.py"
    source = path.read_text(encoding="utf-8")

    if "sbas_deramp_cell_m=" not in source:
        anchor = "    reference_radius_m=500.0,\n"
        addition = (
            "\n"
            "    # Production SBAS Stage7/8 settings.\n"
            '    sbas_deramp_mode="robust_huber_balanced",\n'
            "    sbas_deramp_cell_m=2000.0,\n"
            "    sbas_deramp_anchors_per_cell=8,\n"
            "    sbas_deramp_huber_delta=1.345,\n"
            "    sbas_deramp_huber_iterations=5,\n"
            "    sbas_stage8_use_scla=False,\n"
            "    scn_time_win=365.0,\n"
            "    scn_wavelength=100.0,\n"
        )
        source = replace_once(source, anchor, anchor + addition, "runner production settings")

    ast.parse(source, filename=str(path))
    path.write_text(source, encoding="utf-8")
    print("patched:", path)


def patch_stage7() -> None:
    path = ROOT / "pystamps/pipeline/stage7_sbas.py"
    source = path.read_text(encoding="utf-8")

    source = source.replace(
        "raw_ref = np.nanmean(ph_ifg_float[ref_ps, :], axis=0)",
        "raw_ref = np.nanmedian(ph_ifg_float[ref_ps, :], axis=0)",
    )
    source = source.replace(
        "proc_ref = np.nanmean(ph_ifg_deramped[ref_ps, :], axis=0)",
        "proc_ref = np.nanmedian(ph_ifg_deramped[ref_ps, :], axis=0)",
    )

    if "PRODUCTION_ROBUST_DERAMP_V65" not in source:
        old = '''    ph_ifg_float = np.asarray(ph_ifg_raw, dtype=np.float64)
    if ported._mat_text(parms.get("scla_deramp", "y"), "y").lower() == "y":
        ph_ifg_deramped, ph_ramp_ifg = ported._deramp_unwrapped_phase(ps2, ph_ifg_float)
    else:
        ph_ifg_deramped = ph_ifg_float
        ph_ramp_ifg = np.zeros_like(ph_ifg_float, dtype=np.float64)
'''
        new = '''    # === PRODUCTION_ROBUST_DERAMP_V65 ===
    ph_ifg_float = np.asarray(ph_ifg_raw, dtype=np.float64)
    deramp_debug: dict[str, Any] = {"mode": "disabled"}

    if ported._mat_text(parms.get("scla_deramp", "y"), "y").lower() == "y":
        deramp_mode = ported._mat_text(
            parms.get("sbas_deramp_mode", "robust_huber_balanced"),
            "robust_huber_balanced",
        ).strip().lower()

        if deramp_mode in {"robust", "robust_huber_balanced"}:
            from pystamps.pipeline.sbas_production import robust_deramp_unwrapped_phase
            ph_ifg_deramped, ph_ramp_ifg, deramp_debug = robust_deramp_unwrapped_phase(
                root, ps2, ph_ifg_float, parms
            )
        elif deramp_mode in {"legacy", "ols", "all_ps_ols"}:
            ph_ifg_deramped, ph_ramp_ifg = ported._deramp_unwrapped_phase(
                ps2, ph_ifg_float
            )
            deramp_debug = {"mode": "legacy_all_ps_ols"}
        else:
            raise Stage7SbasError(f"Unsupported sbas_deramp_mode={deramp_mode!r}")
    else:
        ph_ifg_deramped = ph_ifg_float
        ph_ramp_ifg = np.zeros_like(ph_ifg_float, dtype=np.float64)
'''
        source = replace_once(source, old, new, "Stage7 deramp block")

    if '"deramp": deramp_debug' not in source:
        anchor = '            "reference_ps": int(ref_ps.size),\n'
        if anchor in source:
            source = source.replace(anchor, anchor + '            "deramp": deramp_debug,\n', 1)

    ast.parse(source, filename=str(path))
    path.write_text(source, encoding="utf-8")
    print("patched:", path)


def patch_stage8() -> None:
    path = ROOT / "pystamps/pipeline/stage8_sbas.py"
    source = path.read_text(encoding="utf-8")

    if "PRODUCTION_DIRECT_SCN_V65" not in source:
        old = '''def _corrected_chunk(
    ph_sm: np.ndarray,
    ph_scla: np.ndarray,
    c_ps: np.ndarray,
    ph_ramp: np.ndarray | None,
    start: int,
    stop: int,
    reference_phase: np.ndarray,
) -> np.ndarray:
    y = np.asarray(ph_sm[start:stop, :], dtype=np.float64)
    y -= np.asarray(ph_scla[start:stop, :], dtype=np.float64)
    y -= np.asarray(c_ps[start:stop], dtype=np.float64)[:, None]
    if ph_ramp is not None:
        y -= np.asarray(ph_ramp[start:stop, :], dtype=np.float64)
    y -= reference_phase[None, :]
    return y
'''
        new = '''# === PRODUCTION_DIRECT_SCN_V65 ===
def _corrected_chunk(
    ph_sm: np.ndarray,
    ph_scla: np.ndarray,
    c_ps: np.ndarray,
    ph_ramp: np.ndarray | None,
    start: int,
    stop: int,
    reference_phase: np.ndarray,
    use_scla: bool,
) -> np.ndarray:
    y = np.asarray(ph_sm[start:stop, :], dtype=np.float64)

    if use_scla:
        y -= np.asarray(ph_scla[start:stop, :], dtype=np.float64)
        y -= np.asarray(c_ps[start:stop], dtype=np.float64)[:, None]

    # ph_ramp = raw acquisition phase - deramped acquisition phase.
    if ph_ramp is not None:
        y -= np.asarray(ph_ramp[start:stop, :], dtype=np.float64)

    y -= reference_phase[None, :]
    return y
'''
        source = replace_once(source, old, new, "Stage8 corrected chunk")

    if "use_scla = ported._mat_text(" not in source:
        anchor = '''    if ported._mat_text(parms.get("small_baseline_flag", "n"), "n").lower() != "y":
        raise Stage8SbasError("Stage 8 SBAS requires small_baseline_flag='y'")
'''
        addition = '''
    use_scla = ported._mat_text(
        parms.get("sbas_stage8_use_scla", "n"),
        "n",
    ).strip().lower() in {"1", "y", "yes", "true", "on"}
'''
        source = replace_once(source, anchor, anchor + addition, "Stage8 SCLA switch")

    old_ref = '''    if ref_ps.size:
        ref_values = np.asarray(ph_sm[ref_ps, :], dtype=np.float64)
        ref_values -= np.asarray(ph_scla[ref_ps, :], dtype=np.float64)
        ref_values -= np.asarray(c_ps[ref_ps], dtype=np.float64)[:, None]
        if ph_ramp is not None:
            ref_values -= np.asarray(ph_ramp[ref_ps, :], dtype=np.float64)
        reference_phase = np.nanmedian(ref_values, axis=0)
        reference_phase[~np.isfinite(reference_phase)] = 0.0
    else:
        reference_phase = np.zeros(n_image, dtype=np.float64)
'''
    new_ref = '''    if ref_ps.size:
        ref_values = np.asarray(ph_sm[ref_ps, :], dtype=np.float64)
        if use_scla:
            ref_values -= np.asarray(ph_scla[ref_ps, :], dtype=np.float64)
            ref_values -= np.asarray(c_ps[ref_ps], dtype=np.float64)[:, None]
        if ph_ramp is not None:
            ref_values -= np.asarray(ph_ramp[ref_ps, :], dtype=np.float64)
        reference_phase = np.nanmedian(ref_values, axis=0)
        reference_phase[~np.isfinite(reference_phase)] = 0.0
    else:
        reference_phase = np.zeros(n_image, dtype=np.float64)
'''
    if old_ref in source:
        source = source.replace(old_ref, new_ref, 1)

    old_call = '''            ph_sm, ph_scla, c_ps, ph_ramp, start, stop, reference_phase
        )'''
    new_call = '''            ph_sm,
            ph_scla,
            c_ps,
            ph_ramp,
            start,
            stop,
            reference_phase,
            use_scla,
        )'''
    source = source.replace(old_call, new_call)

    source = source.replace(
        "ref_scn = np.nanmean(np.asarray(ph_scn[ref_ps, :], dtype=np.float64), axis=0)",
        "ref_scn = np.nanmedian(np.asarray(ph_scn[ref_ps, :], dtype=np.float64), axis=0)",
    )

    if '"scla_applied_to_final"' not in source:
        anchor = '            "reference_ps": int(ref_ps.size),\n'
        if anchor in source:
            source = source.replace(
                anchor,
                anchor
                + '            "scla_applied_to_final": bool(use_scla),\n'
                + '            "scn_reference_statistic": "median",\n',
                1,
            )

    ast.parse(source, filename=str(path))
    path.write_text(source, encoding="utf-8")
    print("patched:", path)


def update_current_parms() -> None:
    sys.path.insert(0, str(ROOT))
    from pystamps.io.mat import read_mat, write_mat

    path = DATASET / "parms.mat"
    parms = read_mat(path)
    parms["sbas_deramp_mode"] = "robust_huber_balanced"
    parms["sbas_deramp_cell_m"] = 2000.0
    parms["sbas_deramp_anchors_per_cell"] = 8
    parms["sbas_deramp_huber_delta"] = 1.345
    parms["sbas_deramp_huber_iterations"] = 5
    parms["sbas_stage8_use_scla"] = "n"
    parms["scn_time_win"] = 365.0
    parms["scn_wavelength"] = 100.0
    write_mat(path, parms)
    print("updated:", path)


def compile_check() -> None:
    import py_compile
    for path in (
        ROOT / "pystamps/pipeline/sbas_production.py",
        ROOT / "pystamps/pipeline/stage7_sbas.py",
        ROOT / "pystamps/pipeline/stage8_sbas.py",
        ROOT / "pystamps/prep/gamma_stage1.py",
        ROOT / "run_gamma_sbas_ps_optimized.py",
    ):
        py_compile.compile(str(path), doraise=True)
    print("compile check: PASS")


def main() -> None:
    backup_sources()
    install_helper()
    patch_gamma_stage1()
    patch_runner()
    patch_stage7()
    patch_stage8()
    update_current_parms()
    compile_check()
    print()
    print("Production Stage7/8 V6.5 patch installed.")
    print("Robust ramp: R2_C2000_P8")
    print("SCLA applied to final deformation: OFF")
    print("SCN mode: DIRECT")
    print("SCN reference: median")


if __name__ == "__main__":
    main()
