#!/bin/bash
set -euo pipefail


OUT="docs/legacy_code_inventory.md"


echo "# pyPSDS-GAMMA v1.1 legacy code inventory" > ${OUT}
echo "" >> ${OUT}

echo "Generated: $(date)" >> ${OUT}
echo "" >> ${OUT}


echo "## Python files" >> ${OUT}
echo "" >> ${OUT}

printf "| File | Size | Suggested module |\n" >> ${OUT}
printf "|---|---|---|\n" >> ${OUT}



find . \
-name "*.py" \
-not -path "./.git/*" \
| sort \
| while read f
do

    size=$(du -h "$f" | awk '{print $1}')

    name=$(basename "$f" | tr '[:upper:]' '[:lower:]')


    module="review"

    case "$name" in

        *gamma*|*geometry*|*geo*|*lookup*|*incidence*)
            module="preparation/geometry.py"
            ;;


        *base*|*baseline*)
            module="correction/baseline.py"
            ;;


        *gacos*|*atmo*)
            module="correction/gacos.py"
            ;;


        *scla*|*dem*error*)
            module="correction/scla.py"
            ;;


        *scn*|*phase*link*|*unwrap*)
            module="inversion/scn.py"
            ;;


        *timeseries*|*velocity*|*los*)
            module="inversion/timeseries.py"
            ;;


        *export*|*product*|*point*)
            module="export/products.py"
            ;;


        *quality*|*audit*|*check*)
            module="quality/qc.py"
            ;;

    esac


    echo "| $f | $size | $module |" >> ${OUT}

done



echo "" >> ${OUT}

echo "## Shell scripts" >> ${OUT}
echo "" >> ${OUT}


printf "| Script | Suggested role |\n" >> ${OUT}
printf "|---|---|\n" >> ${OUT}



find . \
-name "*.sh" \
-not -path "./.git/*" \
| sort \
| while read f
do

    echo "| $f | workflow entry |" >> ${OUT}

done



echo
echo "Generated:"
echo ${OUT}


