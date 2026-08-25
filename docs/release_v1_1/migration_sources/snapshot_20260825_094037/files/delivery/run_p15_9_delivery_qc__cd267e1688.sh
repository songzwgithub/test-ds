#!/bin/bash
set -euo pipefail


echo "================================================================================"
echo "P15-9 FINAL DELIVERY QC"
echo "================================================================================"


cat > /tmp/p15_9_delivery_qc.py <<'PY'


from pathlib import Path
import json
import time

import numpy as np
import pandas as pd



BASE = Path(
"/home/ubuntu/Downloads/psds/output/processing"
)


INPUT = (
BASE /
"final_point_products" /
"psds_final_points.parquet"
)


OUT = (
BASE /
"final_delivery"
)

OUT.mkdir(
parents=True,
exist_ok=True
)


t0=time.time()


print("Loading final point product")


df=pd.read_parquet(
INPUT
)


n=len(df)


print(
"points:",
n
)



# =====================================================
# integrity
# =====================================================


required=[

"longitude_deg",
"latitude_deg",
"los_velocity_mm_yr",
"velocity_se_mm_yr",
"linear_residual_rms_mm",
"phase_std_rad"

]


missing=[

x for x in required
if x not in df.columns

]


if missing:

    raise RuntimeError(
        f"missing fields {missing}"
    )



# =====================================================
# quality fields
# =====================================================


df["velocity_abs_mm_yr"] = (
np.abs(
df["los_velocity_mm_yr"]
)
)


df["velocity_snr"] = (
df["velocity_abs_mm_yr"]
/
(df["velocity_se_mm_yr"]+1e-6)
)



def phase_grade(x):

    if x < 0.5:
        return "A"

    if x < 1.0:
        return "B"

    if x < 2.0:
        return "C"

    return "D"



df["phase_quality"] = (
df["phase_std_rad"]
.apply(
phase_grade
)
)



df["quality_flag"] = (
(df["velocity_snr"]>=2)
&
(df["linear_residual_rms_mm"]<5)
)



# =====================================================
# statistics
# =====================================================


stats={

"points":
int(n),

"longitude":

[
float(df.longitude_deg.min()),
float(df.longitude_deg.max())
],


"latitude":

[
float(df.latitude_deg.min()),
float(df.latitude_deg.max())
],


"velocity_mm_yr":

{

"p01":
float(np.percentile(df.los_velocity_mm_yr,1)),

"p50":
float(np.percentile(df.los_velocity_mm_yr,50)),

"p99":
float(np.percentile(df.los_velocity_mm_yr,99))

},


"quality_pass_fraction":

float(
df.quality_flag.mean()
)

}



with open(
OUT/
"velocity_statistics.json",
"w"
) as f:

    json.dump(
        stats,
        f,
        indent=2
    )



# =====================================================
# save final
# =====================================================


df.to_parquet(
OUT/
"PSDS_final_points.parquet",
index=False,
compression="zstd"
)


df.to_csv(
OUT/
"PSDS_final_points.csv",
index=False
)



# =====================================================
# report
# =====================================================


report={

"stage":
"P15-9_FINAL_DELIVERY_QC",

"points":
int(n),

"input":
str(INPUT),

"columns":
list(df.columns),

"quality_pass_fraction":
float(df.quality_flag.mean()),

"elapsed_seconds":
time.time()-t0

}



with open(
OUT/
"p15_9_delivery_manifest.json",
"w"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print()
print("="*80)
print(
"P15-9 FINAL RESULT: PASS_DELIVERY_QC"
)
print("="*80)


PY


python -m py_compile \
/tmp/p15_9_delivery_qc.py

echo "SYNTAX PASS"


python \
/tmp/p15_9_delivery_qc.py


