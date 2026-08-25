from pathlib import Path
import json
import re
import numpy as np

ROOT = Path("/home/ubuntu/Downloads")
PSDS = ROOT / "psds"
PROC = PSDS / "output/processing"

# ----------------------------------------------------------------------
# 1. Current 108 production pairs
# ----------------------------------------------------------------------

LOGDIR = PROC / "batch_unwrap_validation/logs"

pat_log = re.compile(
    r"pair(\d+)_([0-9]{8})_([0-9]{8})_single_ifg\.log$"
)

pairs = []

for p in LOGDIR.glob("pair*_single_ifg.log"):
    m = pat_log.match(p.name)
    if not m:
        continue

    pairs.append(
        (
            int(m.group(1)),
            m.group(2),
            m.group(3),
        )
    )

pairs.sort()

if len(pairs) != 108:
    raise RuntimeError(
        f"expected 108 production IFGs, found {len(pairs)}"
    )

pair_keys = {
    tuple(sorted((d1, d2)))
    for _, d1, d2 in pairs
}

# ----------------------------------------------------------------------
# 2. Find ORIGINAL .base files
#
# Deliberately exclude prototype/preproduction/output generated bridges.
# ----------------------------------------------------------------------

exclude_parts = {
    "prototype_outputs",
    "output_preproduction_20260824_091037",
    "pystamps_bridge",
    "generated_bases",
}

date_pair_re = re.compile(
    r"(?<!\d)(20\d{6})[^0-9]+(20\d{6})(?!\d)"
)

candidates = {}

for p in ROOT.rglob("*.base"):

    parts = set(p.parts)

    if parts & exclude_parts:
        continue

    s = str(p)

    # Avoid treating current pyPSDS derived output as original GAMMA input.
    if "/psds/output/" in s:
        continue

    m = date_pair_re.search(p.name)

    if not m:
        # try full path
        m = date_pair_re.search(s)

    if not m:
        continue

    d1, d2 = m.group(1), m.group(2)
    key = tuple(sorted((d1, d2)))

    if key not in pair_keys:
        continue

    candidates.setdefault(
        key,
        [],
    ).append(p)


# ----------------------------------------------------------------------
# 3. Parse GAMMA baseline-model fields
# ----------------------------------------------------------------------

def parse_base(path: Path):
    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    def values_for(label):
        for line in text.splitlines():
            if label not in line:
                continue

            rhs = (
                line.split(":", 1)[1]
                if ":" in line
                else line
            )

            nums = re.findall(
                r"[-+]?"
                r"(?:\d+(?:\.\d*)?|\.\d+)"
                r"(?:[Ee][-+]?\d+)?",
                rhs,
            )

            if len(nums) >= 3:
                return np.asarray(
                    [float(x) for x in nums[:3]],
                    dtype=np.float64,
                )

        return None

    B = (
        values_for("initial_baseline(TCN)")
        if "initial_baseline(TCN)" in text
        else values_for("initial_baseline")
    )

    Br = values_for(
        "initial_baseline_rate"
    )

    return B, Br


valid_sources = {}
invalid_sources = {}
missing = []

for _, d1, d2 in pairs:

    key = tuple(sorted((d1, d2)))

    options = candidates.get(
        key,
        [],
    )

    good = []

    for p in options:

        B, Br = parse_base(p)

        if (
            B is not None
            and Br is not None
            and np.all(np.isfinite(B))
            and np.all(np.isfinite(Br))
        ):
            good.append(
                {
                    "path": str(p),
                    "baseline_tcn": B.tolist(),
                    "baseline_rate_tcn": Br.tolist(),
                }
            )

    if good:
        valid_sources[
            f"{d1}_{d2}"
        ] = good

    elif options:
        invalid_sources[
            f"{d1}_{d2}"
        ] = [
            str(x)
            for x in options
        ]

    else:
        missing.append(
            f"{d1}_{d2}"
        )


# ----------------------------------------------------------------------
# 4. Existing current-point geometry contract
# ----------------------------------------------------------------------

geom = (
    PROC
    / "gacos_geometry"
)

check_files = [
    geom / "strict_points.plist",
    geom / "longitude_deg.npy",
    geom / "latitude_deg.npy",
    geom / "incidence_gamma_compatible_fast_rad.npy",
]

geometry_status = {}

for p in check_files:

    item = {
        "exists": p.exists(),
    }

    if p.exists():

        item["bytes"] = p.stat().st_size

        if p.suffix == ".npy":
            x = np.load(
                p,
                mmap_mode="r",
                allow_pickle=False,
            )
            item["shape"] = list(x.shape)
            item["dtype"] = str(x.dtype)

    geometry_status[
        p.name
    ] = item


# ----------------------------------------------------------------------
# 5. Current production acquisition Bperp: diagnostic only
# ----------------------------------------------------------------------

b_acq = np.load(
    PROC
    / "network/acquisition_bperp_m.npy"
).astype(np.float64)

# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

print("=" * 88)
print("P15-5B3 STAMPS FINAL-PASS BASELINE SOURCE CONTRACT")
print("=" * 88)

print(f"production IFGs           : {len(pairs)}")
print(f"valid original base pairs : {len(valid_sources)}")
print(f"invalid base pairs        : {len(invalid_sources)}")
print(f"missing base pairs        : {len(missing)}")

print()
print("acquisition Bperp:")
print(
    f"  shape/span m            : "
    f"{b_acq.shape} / {np.ptp(b_acq):.6f}"
)
print("  role                    : DIAGNOSTIC_ONLY")

print()
print("current-point geometry:")

for name, x in geometry_status.items():
    print(
        f"  {name:<42} "
        f"{x}"
    )

if missing:
    print()
    print("MISSING ORIGINAL BASE PAIRS:")
    for x in missing:
        print(" ", x)

if invalid_sources:
    print()
    print("BASE FILES WITHOUT REQUIRED TCN/RATE:")
    for pair, paths in invalid_sources.items():
        print(" ", pair)
        for p in paths:
            print("    ", p)

duplicate = {
    k: v
    for k, v in valid_sources.items()
    if len(v) > 1
}

if duplicate:
    print()
    print("MULTIPLE VALID SOURCES:")
    for pair, vals in duplicate.items():
        print(" ", pair)
        for v in vals:
            print("    ", v["path"])

report = {
    "status": (
        "PASS_BASE_SOURCE_COMPLETE"
        if len(valid_sources) == 108
        else "INCOMPLETE_BASE_SOURCE"
    ),
    "production_ifgs": len(pairs),
    "valid_base_pairs": len(valid_sources),
    "invalid_base_pairs": len(invalid_sources),
    "missing_base_pairs": len(missing),
    "missing": missing,
    "invalid": invalid_sources,
    "sources": valid_sources,
    "geometry": geometry_status,
    "acquisition_bperp_role": "DIAGNOSTIC_ONLY",
}

out = (
    PROC
    / "scla_residual_dem_estimation"
    / "p15_5b3_baseline_source_contract.json"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.write_text(
    json.dumps(
        report,
        indent=2,
    ),
    encoding="utf-8",
)

print()
print("report                   :", out)

print("=" * 88)

if len(valid_sources) == 108:
    print(
        "P15-5B3 FINAL RESULT: "
        "PASS_BASE_SOURCE_COMPLETE"
    )
else:
    print(
        "P15-5B3 FINAL RESULT: "
        "INCOMPLETE_BASE_SOURCE"
    )

print("=" * 88)
print("AUDIT ONLY -- NO PHASE MODIFICATION")
