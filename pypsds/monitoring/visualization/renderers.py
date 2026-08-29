from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np

from pypsds.monitoring.visualization.quality import (
    render_override as render_quality,
)

VISUALIZATION_PROFILE = "scientific_final_v1"


def _residual_ramp_final(ax, output: Path, stack, base):
    d = output / "processing" / "residual_ramp"

    manifest_path = d / "residual_ramp_manifest.json"
    direct_path = d / "ifg_ramp_direct_coefficients_rad_per_km.npy"
    projected_path = d / "ifg_ramp_projected_coefficients_rad_per_km.npy"
    acq_path = d / "acquisition_ramp_coefficients_rad_per_km.npy"

    if not (
        manifest_path.is_file()
        and direct_path.is_file()
        and projected_path.is_file()
        and acq_path.is_file()
    ):
        return render_quality(
            "residual_ramp",
            ax,
            None,
            output,
            stack,
            300000,
            base,
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.get("status") == "PASS_DISABLED":
        for a in ax:
            a.set_axis_off()
        ax[0].text(
            0.03,
            0.97,
            "Residual ramp is disabled.\n"
            "No IFG spatial ramp was removed.",
            va="top",
            transform=ax[0].transAxes,
            family="monospace",
        )
        return "DISABLED", [
            f"profile: {VISUALIZATION_PROFILE}",
            "residual_ramp: disabled",
        ]

    direct = np.asarray(
        np.load(direct_path, mmap_mode="r"),
        dtype=np.float64,
    )
    projected = np.asarray(
        np.load(projected_path, mmap_mode="r"),
        dtype=np.float64,
    )
    acq = np.asarray(
        np.load(acq_path, mmap_mode="r"),
        dtype=np.float64,
    )

    direct_mag = np.hypot(direct[:, 0], direct[:, 1])
    projected_mag = np.hypot(projected[:, 0], projected[:, 1])

    ax[0].scatter(
        direct_mag,
        projected_mag,
        s=18,
        alpha=0.75,
    )
    lo = float(np.nanmin(np.r_[direct_mag, projected_mag]))
    hi = float(np.nanmax(np.r_[direct_mag, projected_mag]))
    if np.isfinite(lo) and np.isfinite(hi):
        ax[0].plot([lo, hi], [lo, hi], "--", linewidth=0.9)
    ax[0].set_xlabel("Direct IFG |slope| [rad/km]")
    ax[0].set_ylabel("Projected IFG |slope| [rad/km]")
    ax[0].set_title(
        "Direct vs network-projected IFG ramp",
        fontsize=10,
    )

    idx = np.arange(direct.shape[0])
    ax[1].plot(idx, direct_mag, ".-", label="direct IFG fit")
    ax[1].plot(
        idx,
        projected_mag,
        ".-",
        label="network projected",
    )
    ax[1].set_xlabel("IFG index")
    ax[1].set_ylabel("|slope| [rad/km]")
    ax[1].set_title("IFG ramp magnitude", fontsize=10)
    ax[1].legend(fontsize=8)

    dates = base["_date_objects"](stack)
    n = min(len(dates), acq.shape[0])

    ax[2].plot(dates[:n], acq[:n, 0], ".-", label="ax")
    ax[2].plot(dates[:n], acq[:n, 1], ".-", label="by")
    ax[2].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[2].xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            ax[2].xaxis.get_major_locator()
        )
    )
    ax[2].set_ylabel("Acquisition slope [rad/km]")
    ax[2].set_title(
        "Network-integrable acquisition ramp",
        fontsize=10,
    )
    ax[2].legend(fontsize=8)

    projection = manifest.get("network_projection", {})

    visual_status = (
        "REVIEW"
        if str(manifest.get("status", "")).startswith("REVIEW")
        else "PASS"
    )

    return visual_status, [
        f"profile: {VISUALIZATION_PROFILE}",
        f"method: {manifest.get('method')}",
        f"anchors: {manifest.get('anchors')}",
        f"occupied cells: {manifest.get('occupied_cells')}",
        (
            "direct/projected xy corr: "
            f"{projection.get('combined_xy_correlation')}"
        ),
        (
            "projection ratio p50/p95/p99: "
            f"{projection.get('spatial_diff_to_direct_ratio_p50_p95_p99')}"
        ),
        (
            "projection recommendation: "
            f"{projection.get('recommendation')}"
        ),
        "intercept removed: false",
    ]


def render_override(
    stage,
    ax,
    cfg,
    output,
    stack,
    max_points,
    base,
):
    if stage == "residual_ramp":
        return _residual_ramp_final(
            ax,
            output,
            stack,
            base,
        )

    return render_quality(
        stage,
        ax,
        cfg,
        output,
        stack,
        max_points,
        base,
    )
