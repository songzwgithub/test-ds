from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _out(fig_dir: Path, name: str, dpi: int):
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = fig_dir / name
    plt.tight_layout()
    plt.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close()
    return p


def plot_step01(fig_dir: Path, dates, mean_amp, sample, dpi=140):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(len(mean_amp)), mean_amp, marker="o", markersize=3)
    ax.set_title("Step 01 - Mean amplitude by acquisition")
    ax.set_xlabel("Acquisition index")
    ax.set_ylabel("Mean amplitude")
    ax.grid(True, alpha=0.25)
    p1 = _out(fig_dir, "01_mean_amplitude.png", dpi)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.abs(sample[0]), origin="upper")
    ax.set_title("Step 01 - First acquisition amplitude sample")
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    fig.colorbar(im, ax=ax, label="Amplitude")
    p2 = _out(fig_dir, "01_amplitude_sample.png", dpi)
    return [p1, p2]


def plot_step02(fig_dir: Path, counts, window_level=None, candidates=None, dpi=140):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(counts, origin="upper")
    ax.set_title("Step 02 - SHP count")
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    fig.colorbar(im, ax=ax, label="SHP count")
    out=[_out(fig_dir, "02_shp_count_map.png", dpi)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts.ravel(), bins=40)
    ax.set_title("Step 02 - SHP count distribution")
    ax.set_xlabel("SHP count")
    ax.set_ylabel("Pixels")
    out.append(_out(fig_dir, "02_shp_count_hist.png", dpi))
    if window_level is not None and np.size(window_level):
        fig, ax = plt.subplots(figsize=(7, 5))
        im=ax.imshow(window_level, origin="upper")
        ax.set_title("Step 02 - Adaptive window level")
        ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row")
        fig.colorbar(im, ax=ax, label="Window level")
        out.append(_out(fig_dir, "02_window_level_map.png", dpi))
    return out


def _scatter_map(ax, centers, values, title, label):
    sc=ax.scatter(centers[:,1], centers[:,0], c=values, s=8)
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_title(title); ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row")
    plt.colorbar(sc, ax=ax, label=label)


def plot_step03(fig_dir: Path, centers, coherence, box_coherence, dpi=140):
    iu=np.triu_indices(coherence.shape[1], 1)
    shp_med=np.median(np.abs(coherence[:,iu[0],iu[1]]), axis=1)
    box_med=np.median(np.abs(box_coherence[:,iu[0],iu[1]]), axis=1)
    fig, ax=plt.subplots(figsize=(7,5)); _scatter_map(ax, centers, shp_med, "Step 03 - Median pair coherence", "Median |Cij|")
    out=[_out(fig_dir,"03_coherence_map.png",dpi)]
    fig, ax=plt.subplots(figsize=(6,6)); ax.scatter(box_med, shp_med, s=8, alpha=.6); lo=min(box_med.min(),shp_med.min()); hi=max(box_med.max(),shp_med.max()); ax.plot([lo,hi],[lo,hi],linestyle="--"); ax.set_xlabel("Box median |Cij|"); ax.set_ylabel("SHP median |Cij|"); ax.set_title("Step 03 - SHP vs box coherence")
    out.append(_out(fig_dir,"03_shp_vs_box_coherence.png",dpi)); return out


def plot_step04(fig_dir: Path, centers, temporal_coherence, phase_rad, dpi=140):
    fig, ax=plt.subplots(figsize=(7,5)); _scatter_map(ax, centers, temporal_coherence, "Step 04 - Temporal coherence", "Temporal coherence")
    out=[_out(fig_dir,"04_temporal_coherence_map.png",dpi)]
    fig, ax=plt.subplots(figsize=(7,4)); ax.hist(temporal_coherence,bins=40); ax.set_xlabel("Temporal coherence"); ax.set_ylabel("Points"); ax.set_title("Step 04 - Temporal coherence distribution")
    out.append(_out(fig_dir,"04_temporal_coherence_hist.png",dpi)); return out


def plot_step05(fig_dir: Path, evd_tc, emi_tc, box_tc, dpi=140):
    fig, ax=plt.subplots(figsize=(6,6)); ax.scatter(box_tc, evd_tc, s=8, alpha=.6); lo=min(box_tc.min(),evd_tc.min()); hi=max(box_tc.max(),evd_tc.max()); ax.plot([lo,hi],[lo,hi],linestyle="--"); ax.set_xlabel("Box-EVD temporal coherence"); ax.set_ylabel("SHP-EVD temporal coherence"); ax.set_title("Step 05 - SHP vs box")
    out=[_out(fig_dir,"05_shp_vs_box_tc.png",dpi)]
    fig, ax=plt.subplots(figsize=(7,4)); ax.hist(evd_tc-box_tc,bins=50); ax.axvline(0,linestyle="--"); ax.set_xlabel("SHP-EVD TC - Box-EVD TC"); ax.set_ylabel("Points"); ax.set_title("Step 05 - Paired TC difference")
    out.append(_out(fig_dir,"05_tc_difference_hist.png",dpi)); return out


def plot_step06(fig_dir: Path, dispersion, mask, dpi=140):
    fig, ax=plt.subplots(figsize=(7,5)); im=ax.imshow(dispersion,origin="upper"); ax.set_title("Step 06 - Amplitude dispersion index"); ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row"); fig.colorbar(im,ax=ax,label="ADI")
    out=[_out(fig_dir,"06_ps_amplitude_dispersion.png",dpi)]
    fig, ax=plt.subplots(figsize=(7,5)); ax.imshow(mask,origin="upper"); ax.set_title("Step 06 - PS candidate mask"); ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row")
    out.append(_out(fig_dir,"06_ps_candidate_mask.png",dpi)); return out


def plot_step07(fig_dir: Path, row, col, point_type, quality, dpi=140):
    fig, ax=plt.subplots(figsize=(7,5))
    for t, label in [(1,"PS"),(2,"DS")]:
        m=point_type==t
        if np.any(m): ax.scatter(col[m],row[m],s=7,label=label)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row"); ax.set_title("Step 07 - Fused PS/DS points"); ax.legend()
    out=[_out(fig_dir,"07_fused_point_types.png",dpi)]
    fig, ax=plt.subplots(figsize=(7,4)); ax.hist(quality[np.isfinite(quality)],bins=40); ax.set_xlabel("Quality"); ax.set_ylabel("Points"); ax.set_title("Step 07 - Fused point quality")
    out.append(_out(fig_dir,"07_fused_quality_hist.png",dpi)); return out


def plot_step08(fig_dir: Path, result, *, dpi=140):
    out=[]

    # SHP structure must be inspected before phase-linking quality.
    fig,ax=plt.subplots(figsize=(8,5))
    im=ax.imshow(result.shp_count_map,origin="upper")
    ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row")
    ax.set_title("Step 08 - SHP count (candidate-first workflow)")
    fig.colorbar(im,ax=ax,label="SHP count")
    out.append(_out(fig_dir,"08_shp_count_map.png",dpi))

    fig,ax=plt.subplots(figsize=(8,5))
    im=ax.imshow(result.ds_candidate_mask,origin="upper",vmin=0,vmax=1)
    ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row")
    ax.set_title(f"Step 08 - DS candidate mask (min SHP={result.ds_min_shp})")
    fig.colorbar(im,ax=ax,label="Candidate")
    out.append(_out(fig_dir,"08_ds_candidate_mask.png",dpi))

    # 0=invalid, 1=PS, 2=DS candidate only, 3=phase-linked DS.
    type_map=np.zeros(result.ps_mask.shape,dtype=np.uint8)
    type_map[result.ds_candidate_mask]=2
    type_map[result.ds_evaluated_mask]=3
    type_map[result.ps_mask]=1
    fig,ax=plt.subplots(figsize=(8,5))
    im=ax.imshow(type_map,origin="upper",vmin=0,vmax=3)
    ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row")
    ax.set_title("Step 08 - PS / DS candidate / linked-DS classes")
    fig.colorbar(im,ax=ax,label="0 invalid, 1 PS, 2 candidate, 3 linked DS")
    out.append(_out(fig_dir,"08_phase_link_validity.png",dpi))

    for arr,name,title,label in [
        (result.ds_tc_map,"08_ds_tc_map.png","Step 08 - Candidate-only DS temporal coherence","Temporal coherence"),
        (result.ds_pair_coherence_map,"08_ds_pair_coherence_map.png","Step 08 - Candidate-only median pair coherence","Median |Cij|"),
    ]:
        fig,ax=plt.subplots(figsize=(8,5))
        show=np.where(result.ds_evaluated_mask,arr,np.nan)
        im=ax.imshow(show,origin="upper")
        ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row"); ax.set_title(title)
        fig.colorbar(im,ax=ax,label=label)
        out.append(_out(fig_dir,name,dpi))

    fig,ax=plt.subplots(figsize=(8,5))
    show=np.where(result.ds_evaluated_mask,result.ds_estimator_map.astype(float),np.nan)
    im=ax.imshow(show,origin="upper",vmin=0,vmax=1)
    ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row")
    ax.set_title("Step 08 - Actual PL estimator (0 EVD, 1 EMI)")
    fig.colorbar(im,ax=ax,label="Estimator")
    out.append(_out(fig_dir,"08_pl_estimator_map.png",dpi))

    fig,ax=plt.subplots(figsize=(8,5))
    show=np.where(result.ds_evaluated_mask,result.ds_pl_status_map.astype(float),np.nan)
    im=ax.imshow(show,origin="upper")
    ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row")
    ax.set_title("Step 08 - Phase-link status code")
    fig.colorbar(im,ax=ax,label="PL status")
    out.append(_out(fig_dir,"08_pl_status_map.png",dpi))

    fig,ax=plt.subplots(figsize=(8,5))
    im=ax.imshow(result.geometry_valid_mask,origin="upper",vmin=0,vmax=1)
    ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row")
    ax.set_title("Step 08 - Geometry correction valid mask")
    fig.colorbar(im,ax=ax,label="Valid")
    out.append(_out(fig_dir,"08_geometry_valid_mask.png",dpi))

    tc=result.ds_tc_map[result.ds_evaluated_mask]
    if tc.size:
        fig,ax=plt.subplots(figsize=(7,4)); ax.hist(tc,bins=50)
        ax.set_xlabel("DS temporal coherence"); ax.set_ylabel("Points")
        ax.set_title("Step 08 - Linked-DS temporal coherence distribution")
        out.append(_out(fig_dir,"08_ds_tc_hist.png",dpi))
    pc=result.ds_pair_coherence_map[result.ds_evaluated_mask]
    pc=pc[np.isfinite(pc)]
    if pc.size:
        fig,ax=plt.subplots(figsize=(7,4)); ax.hist(pc,bins=50)
        ax.set_xlabel("Median pair coherence"); ax.set_ylabel("Points")
        ax.set_title("Step 08 - Linked-DS median pair coherence distribution")
        out.append(_out(fig_dir,"08_ds_pair_coherence_hist.png",dpi))

    m=result.ds_evaluated_mask & np.isfinite(result.ds_tc_map) & np.isfinite(result.ds_pair_coherence_map)
    if np.any(m):
        fig,ax=plt.subplots(figsize=(6,6))
        ax.hexbin(result.ds_pair_coherence_map[m],result.ds_tc_map[m],gridsize=55,mincnt=1)
        ax.set_xlabel("Median pair coherence"); ax.set_ylabel("Temporal coherence")
        ax.set_title("Step 08 - Linked-DS quality space")
        out.append(_out(fig_dir,"08_tc_vs_pair_coherence.png",dpi))
    return out


def plot_step09_quality_map(fig_dir: Path, ps_mask, final_ds, tc, pair, *, dpi=140):
    out=[]
    fig,ax=plt.subplots(figsize=(8,5))
    rr,cc=np.where(final_ds)
    if rr.size:
        ax.scatter(cc,rr,s=2,alpha=.55,label="Final DS")
    pr,pc=np.where(ps_mask)
    if pr.size:
        ax.scatter(pc,pr,s=7,alpha=.85,label="PS")
    ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xlabel("Range column"); ax.set_ylabel("Azimuth row")
    ax.set_title("Step 09 - Final PS/DS point distribution")
    ax.legend()
    out.append(_out(fig_dir,"09_final_psds_points.png",dpi))

    m=final_ds & np.isfinite(tc) & np.isfinite(pair)
    if np.any(m):
        fig,ax=plt.subplots(figsize=(6,6))
        ax.scatter(pair[m],tc[m],s=4,alpha=.35)
        ax.set_xlabel("Median pair coherence"); ax.set_ylabel("Temporal coherence")
        ax.set_title("Step 09 - Final DS quality space")
        out.append(_out(fig_dir,"09_final_ds_quality_space.png",dpi))
    return out


def plot_step09_sensitivity(fig_dir: Path, tc_thresholds, curves, *, dpi=140):
    fig,ax=plt.subplots(figsize=(8,5))
    for label,density in curves:
        ax.plot(tc_thresholds,density,marker="o",label=label)
    ax.set_xlabel("Temporal coherence threshold")
    ax.set_ylabel("Final PS + DS density")
    ax.set_ylim(0,1.02); ax.grid(True,alpha=.25); ax.legend()
    ax.set_title("Step 09 - Candidate-only DS selection sensitivity")
    return [_out(fig_dir,"09_ds_selection_sensitivity.png",dpi)]
