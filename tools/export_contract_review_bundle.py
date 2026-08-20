#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INV = (
    ROOT
    / "docs"
    / "release"
    / "stage_contract_inventory.json"
)

OUT = (
    ROOT
    / "docs"
    / "release"
    / "stage_contract_review_bundle.txt"
)

if not INV.is_file():
    raise FileNotFoundError(INV)

inv = json.loads(
    INV.read_text(encoding="utf-8")
)

blocks = []

blocks.append(
    "pyPSDS-GAMMA StageContract review bundle\n"
)

blocks.append(
    f"TOTAL      : {inv['summary']['total']}\n"
    f"AUTO_READY : {inv['summary']['auto_ready']}\n"
    f"REVIEW     : {inv['summary']['review']}\n"
)

for name, info in inv["stages"].items():

    if info.get("status") != "REVIEW":
        continue

    script_name = info["script"]
    script = ROOT / "scripts" / script_name

    if not script.is_file():
        continue

    lines = script.read_text(
        encoding="utf-8"
    ).splitlines()

    blocks.append(
        "\n"
        + "=" * 110
        + "\n"
        + f"STAGE: {name}\n"
        + f"SCRIPT: {script_name}\n"
        + "=" * 110
        + "\n"
    )

    blocks.append("\nRESOLVED INPUTS:\n")

    for x in info.get(
        "exact_output_inputs",
        []
    ):
        blocks.append(
            f"  IN  {x}\n"
        )

    blocks.append("\nRESOLVED OUTPUTS:\n")

    for x in info.get(
        "exact_output_outputs",
        []
    ):
        blocks.append(
            f"  OUT {x}\n"
        )

    issues = []

    for x in info.get(
        "dynamic_refs",
        []
    ):
        issues.append(
            (
                "DYNAMIC",
                x["line"],
                x.get(
                    "symbolic_path",
                    ""
                ),
                x.get(
                    "api",
                    ""
                ),
            )
        )

    for x in info.get(
        "unresolved_io",
        []
    ):
        issues.append(
            (
                "UNRESOLVED",
                x["line"],
                "",
                x.get(
                    "api",
                    ""
                ),
            )
        )

    for x in info.get(
        "missing_exact",
        []
    ):
        issues.append(
            (
                "MISSING",
                x["line"],
                x.get(
                    "symbolic_path",
                    ""
                ),
                x.get(
                    "api",
                    ""
                ),
            )
        )

    issues.sort(
        key=lambda x: x[1]
    )

    blocks.append(
        "\nISSUE SOURCE CONTEXT:\n"
    )

    for kind, lineno, path, api in issues:

        blocks.append(
            "\n"
            + "-" * 100
            + "\n"
        )

        blocks.append(
            f"{kind} line={lineno} "
            f"api={api} "
            f"path={path}\n"
        )

        lo = max(
            1,
            lineno - 18,
        )

        hi = min(
            len(lines),
            lineno + 18,
        )

        for n in range(
            lo,
            hi + 1,
        ):

            marker = (
                ">>>"
                if n == lineno
                else "   "
            )

            blocks.append(
                f"{marker} {n:5d}: "
                f"{lines[n-1]}\n"
            )

    # Also include all Path / np.load / np.save / memmap references
    # in the file. This is useful for dependencies missed because
    # the path is assembled inside a helper function.
    blocks.append(
        "\nPOTENTIAL I/O SOURCE LINES:\n"
    )

    needles = (
        "np.load",
        "np.save",
        "open_memmap",
        ".read_text",
        ".write_text",
        ".open(",
        "copy2",
        "copyfile",
        ".glob(",
        ".rglob(",
        "tofile",
    )

    for n, line in enumerate(
        lines,
        start=1,
    ):

        if any(
            x in line
            for x in needles
        ):

            blocks.append(
                f"  {n:5d}: {line}\n"
            )

OUT.write_text(
    "".join(blocks),
    encoding="utf-8",
)

print("review bundle :", OUT)
print(
    "size          :",
    f"{OUT.stat().st_size:,} bytes",
)
print(
    "review stages :",
    inv["summary"]["review"],
)

