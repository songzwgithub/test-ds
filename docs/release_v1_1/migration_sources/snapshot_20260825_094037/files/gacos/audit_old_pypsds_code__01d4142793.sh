#!/bin/bash
set -euo pipefail


ROOT=$(pwd)

OUT="docs/legacy_code_mapping.md"


echo "# Legacy code mapping" > ${OUT}
echo "" >> ${OUT}
echo "Generated: $(date)" >> ${OUT}
echo "" >> ${OUT}


echo "## Python files" >> ${OUT}
echo "" >> ${OUT}


find . \
-name "*.py" \
-not -path "./.git/*" \
| sort \
>> ${OUT}



echo "" >> ${OUT}
echo "## Shell scripts" >> ${OUT}
echo "" >> ${OUT}


find . \
-name "*.sh" \
-not -path "./.git/*" \
| sort \
>> ${OUT}



echo "" >> ${OUT}
echo "## Suggested migration" >> ${OUT}
echo "" >> ${OUT}


cat >> ${OUT} <<EOF

|Old component|New module|
|-|-|
|geometry scripts|preparation/geometry.py|
|baseline scripts|correction/baseline.py|
|GACOS scripts|correction/gacos.py|
|SCLA scripts|correction/scla.py|
|SCN scripts|inversion/scn.py|
|time series scripts|inversion/timeseries.py|
|velocity export|export/products.py|
|QC scripts|quality/qc.py|

EOF



echo "Generated:"
echo ${OUT}

