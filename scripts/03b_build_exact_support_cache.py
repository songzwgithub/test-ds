#!/usr/bin/env python3

from pathlib import Path
import runpy


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


runpy.run_path(
    str(
        ROOT
        /
        "tools"
        /
        "build_exact_support_cache.py"
    ),
    run_name="__main__",
)
