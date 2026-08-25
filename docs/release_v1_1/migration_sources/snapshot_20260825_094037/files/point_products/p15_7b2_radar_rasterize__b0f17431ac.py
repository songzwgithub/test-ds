from pathlib import Path
import json
import time
import numpy as np


PSDS = Path(
    "/home/ubuntu/Downloads/psds"
)

PROC = (
    PSDS
    / "output/processing"
)

OUT = (
    PROC
    / "final_los_geocoding"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


ROWS = (
    PROC
    / "point_phase_stack"
    / "rows.npy"
)

COLS = (
    PROC
    / "point_phase_stack"
    / "cols.npy"
)

STRICT = (
    PROC
    / "network_inversion"
    / "strict_point_ids.npy"
)

VEL = (
    PROC
    / "final_los_products"
    / "los_velocity_toward_satellite_mm_per_year.npy"
)

CUM = (
    PROC
    / "final_los_products"
    / "los_cumulative_toward_satellite_mm.npy"
)


# GAMMA 4:1 radar geometry
LINES = 600
WIDTH = 500


def rasterize_median(
        row,
        col,
        value,
        shape
):

    flat = (
        row.astype(np.int64)
        *
        shape[1]
        +
        col.astype(np.int64)
    )

    order = np.argsort(flat)

    flat_s = flat[order]
    val_s = value[order]


    unique, start, count = np.unique(
        flat_s,
        return_index=True,
        return_counts=True
    )


    out = np.full(
        shape[0]*shape[1],
        np.nan,
        dtype=np.float32
    )

    cnt = np.zeros(
        shape[0]*shape[1],
        dtype=np.uint16
    )


    for u,s,c in zip(
        unique,
        start,
        count
    ):

        out[u] = np.median(
            val_s[s:s+c]
        )

        cnt[u] = c


    return (
        out.reshape(shape),
        cnt.reshape(shape)
    )


print("="*80)
print("P15-7B2 RADAR RASTERIZATION")
print("="*80)


t0=time.time()


strict_ids = np.load(
    STRICT
)


rows_all = np.load(
    ROWS,
    mmap_mode="r"
)

cols_all = np.load(
    COLS,
    mmap_mode="r"
)


rows = np.asarray(
    rows_all[strict_ids],
    dtype=np.int32
)

cols = np.asarray(
    cols_all[strict_ids],
    dtype=np.int32
)


velocity = np.load(
    VEL
).astype(
    np.float32
)


cumulative = np.load(
    CUM
).astype(
    np.float32
)


print(
    "points:",
    len(rows)
)

print(
    "radar grid:",
    LINES,
    WIDTH
)


t1=time.time()


vel_raster, vel_count = rasterize_median(
    rows,
    cols,
    velocity,
    (LINES,WIDTH)
)


cum_raster, cum_count = rasterize_median(
    rows,
    cols,
    cumulative,
    (LINES,WIDTH)
)


print(
    "raster seconds:",
    time.time()-t1
)


vel_file = (
    OUT
    /
    "radar_velocity_mm_yr.flt"
)

cum_file = (
    OUT
    /
    "radar_cumulative_mm.flt"
)

cnt_file = (
    OUT
    /
    "radar_point_count.uint16"
)


vel_raster.astype(
    ">f4"
).tofile(
    vel_file
)


cum_raster.astype(
    ">f4"
).tofile(
    cum_file
)


vel_count.astype(
    ">u2"
).tofile(
    cnt_file
)


manifest = {

    "stage":
        "P15-7B2",

    "points":
        int(len(rows)),

    "radar_shape":
        [
            LINES,
            WIDTH
        ],

    "aggregation":
        "median",

    "velocity_file":
        str(vel_file),

    "cumulative_file":
        str(cum_file),

    "count_file":
        str(cnt_file),

    "elapsed_seconds":
        time.time()-t0
}


with open(
    OUT/"p15_7b2_manifest.json",
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


print()
print("valid velocity pixels:",
      np.count_nonzero(np.isfinite(vel_raster)))

print("valid cumulative pixels:",
      np.count_nonzero(np.isfinite(cum_raster)))

print("max points/pixel:",
      vel_count.max())


print()
print("OUTPUT:")
print(OUT)

print()
print("P15-7B2 FINAL RESULT: PASS_RADAR_RASTERIZATION")

