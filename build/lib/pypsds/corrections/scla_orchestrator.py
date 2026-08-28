from __future__ import annotations

from pathlib import Path
import json
import os
import re
import shutil
from typing import Any, Iterable


_DATE_RE = re.compile(r"^20\d{6}$")
_PAIR_RE = re.compile(
    r"(20\d{6})[^0-9]+(20\d{6})"
)


def _is_date(value: Any) -> bool:
    return bool(
        _DATE_RE.match(
            str(value)
        )
    )


def _pair_from_record(
    record: Any,
) -> tuple[str, str] | None:

    if isinstance(record, dict):

        key_pairs = (
            ("date_i", "date_j"),
            ("date1", "date2"),
            ("master", "slave"),
            ("reference", "secondary"),
            ("reference_date", "secondary_date"),
            ("ref_date", "sec_date"),
            ("primary_date", "secondary_date"),
        )

        for a, b in key_pairs:
            if (
                a in record
                and
                b in record
                and
                _is_date(record[a])
                and
                _is_date(record[b])
            ):
                return (
                    str(record[a]),
                    str(record[b]),
                )

        dates = [
            str(v)
            for v in record.values()
            if _is_date(v)
        ]

        if len(dates) == 2:
            return (
                dates[0],
                dates[1],
            )

    elif isinstance(
        record,
        (list, tuple),
    ):

        dates = [
            str(v)
            for v in record
            if _is_date(v)
        ]

        if len(dates) == 2:
            return (
                dates[0],
                dates[1],
            )

    return None


def _candidate_edge_lists(
    obj: Any,
) -> list[list[tuple[str, str]]]:

    out: list[
        list[
            tuple[str, str]
        ]
    ] = []

    if isinstance(obj, list):

        pairs = []

        for item in obj:
            pair = _pair_from_record(
                item
            )

            if pair is None:
                pairs = []
                break

            pairs.append(pair)

        if pairs:
            out.append(pairs)

        for item in obj:
            out.extend(
                _candidate_edge_lists(
                    item
                )
            )

    elif isinstance(obj, dict):

        for value in obj.values():
            out.extend(
                _candidate_edge_lists(
                    value
                )
            )

    return out


def load_finalized_network(
    manifest_path: str | Path,
) -> list[dict[str, Any]]:

    path = Path(
        manifest_path
    )

    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------------
    # Public network ownership contract:
    #
    #   network_manifest.json -> acquisition date ordering
    #   network.itab          -> finalized IFG edge ordering
    #
    # Do not depend on the internal JSON schema of obj["network"].
    # The finalized pipeline already persists network.itab as its
    # canonical edge-index product.
    # --------------------------------------------------------------

    dates = [
        str(x)
        for x in obj.get(
            "dates",
            []
        )
    ]

    if (
        not dates
        or
        len(dates)
        !=
        len(set(dates))
        or
        not all(
            _is_date(x)
            for x in dates
        )
    ):
        raise ValueError(
            "Invalid or missing acquisition dates in "
            f"{path}"
        )

    itab = (
        path.parent
        / "network.itab"
    )

    if itab.is_file():

        pairs = []

        for line_no, raw in enumerate(
            itab.read_text(
                encoding="utf-8"
            ).splitlines(),
            start=1,
        ):

            raw = raw.strip()

            if (
                not raw
                or
                raw.startswith("#")
            ):
                continue

            fields = raw.split()

            if len(fields) < 2:
                raise ValueError(
                    f"{itab}:{line_no}: "
                    "expected at least two columns"
                )

            try:
                i = int(
                    fields[0]
                ) - 1

                j = int(
                    fields[1]
                ) - 1

            except ValueError as exc:
                raise ValueError(
                    f"{itab}:{line_no}: "
                    "invalid acquisition indices"
                ) from exc

            if (
                i < 0
                or
                j < 0
                or
                i >= len(dates)
                or
                j >= len(dates)
                or
                i == j
            ):
                raise ValueError(
                    f"{itab}:{line_no}: "
                    f"edge {(i + 1, j + 1)} "
                    f"outside acquisition domain "
                    f"1..{len(dates)}"
                )

            # Preserve the finalized ITAB orientation and order exactly.
            pairs.append(
                (
                    dates[i],
                    dates[j],
                )
            )

        if not pairs:
            raise ValueError(
                f"No finalized edges found in {itab}"
            )

        if len(
            pairs
        ) != len(
            set(pairs)
        ):
            raise ValueError(
                f"Duplicate finalized edges in {itab}"
            )

        return [
            {
                "edge":
                    edge,

                "date_i":
                    d1,

                "date_j":
                    d2,
            }
            for edge, (d1, d2)
            in enumerate(
                pairs,
                start=1,
            )
        ]

    # --------------------------------------------------------------
    # Fallback for small synthetic tests / compatible external
    # manifests that directly store an edge list.
    # --------------------------------------------------------------

    candidates = (
        _candidate_edge_lists(
            obj
        )
    )

    if not candidates:
        raise ValueError(
            "Could not locate finalized temporal network. "
            f"Expected {itab} or an embedded edge list in {path}"
        )

    pairs = max(
        candidates,
        key=len,
    )

    return [
        {
            "edge":
                i,

            "date_i":
                d1,

            "date_j":
                d2,
        }
        for i, (d1, d2)
        in enumerate(
            pairs,
            start=1,
        )
    ]


def write_compat_network_logs(
    network: list[dict[str, Any]],
    directory: str | Path,
) -> Path:

    out = Path(directory)

    if out.exists():
        shutil.rmtree(out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    for i, item in enumerate(
        network,
        start=1,
    ):

        d1 = str(
            item["date_i"]
        )

        d2 = str(
            item["date_j"]
        )

        name = (
            f"pair{i:04d}_"
            f"{d1}_{d2}_"
            "single_ifg.log"
        )

        (
            out
            / name
        ).write_text(
            "",
            encoding="utf-8",
        )

    return out


def pair_from_base_filename(
    path: str | Path,
) -> tuple[str, str] | None:

    m = _PAIR_RE.search(
        Path(path).name
    )

    if not m:
        return None

    return (
        m.group(1),
        m.group(2),
    )


def _is_under(
    path: Path,
    root: Path,
) -> bool:

    try:
        path.resolve().relative_to(
            root.resolve()
        )

        return True

    except ValueError:
        return False


def discover_baseline_sources(
    network: list[dict[str, Any]],
    search_roots: Iterable[str | Path],
    exclude_roots: Iterable[str | Path] = (),
) -> dict[str, list[str]]:

    wanted = {
        frozenset(
            (
                str(item["date_i"]),
                str(item["date_j"]),
            )
        ):
            f"{item['date_i']}_{item['date_j']}"
        for item in network
    }

    result = {
        pair: []
        for pair in wanted.values()
    }

    excludes = [
        Path(x).resolve()
        for x in exclude_roots
    ]

    seen: set[Path] = set()

    for root_value in search_roots:

        root = Path(
            root_value
        )

        if not root.is_dir():
            continue

        for current, dirs, files in os.walk(
            root
        ):

            current_path = Path(
                current
            ).resolve()

            dirs[:] = [
                d
                for d in dirs
                if (
                    d
                    not in {
                        ".git",
                        "__pycache__",
                        ".pytest_cache",
                        ".cache",
                    }
                    and
                    not any(
                        _is_under(
                            current_path / d,
                            ex,
                        )
                        for ex in excludes
                    )
                    and
                    "generated_missing"
                    not in d
                    and
                    "current_network_missing"
                    not in d
                    and
                    "scla_dynamic"
                    not in d
                )
            ]

            for filename in files:

                if not filename.endswith(
                    ".base"
                ):
                    continue

                path = (
                    current_path
                    / filename
                ).resolve()

                if path in seen:
                    continue

                seen.add(path)

                if any(
                    _is_under(
                        path,
                        ex,
                    )
                    for ex in excludes
                ):
                    continue

                pair = pair_from_base_filename(
                    path
                )

                if pair is None:
                    continue

                key = frozenset(
                    pair
                )

                canonical = wanted.get(
                    key
                )

                if canonical is None:
                    continue

                result[
                    canonical
                ].append(
                    str(path)
                )

    for key in result:
        result[key] = sorted(
            set(
                result[key]
            )
        )

    return result


def build_baseline_contract(
    network: list[dict[str, Any]],
    search_roots: Iterable[str | Path],
    output_path: str | Path,
    exclude_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:

    sources = discover_baseline_sources(
        network,
        search_roots,
        exclude_roots,
    )

    missing = [
        f"{item['date_i']}_{item['date_j']}"
        for item in network
        if not sources[
            f"{item['date_i']}_{item['date_j']}"
        ]
    ]

    payload = {
        "status":
            "PORTABLE_BASELINE_SOURCE_DISCOVERY",

        "network_ifgs":
            len(network),

        "original_pairs":
            len(network)
            -
            len(missing),

        "missing":
            missing,

        "sources":
            sources,
    }

    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        +
        "\n",
        encoding="utf-8",
    )

    return payload
