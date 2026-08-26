from __future__ import annotations

from pathlib import Path
import json

from pypsds.corrections import scla_orchestrator as orch


def _dates_edges(proc):
    p = Path(proc)

    manifest_path = (
        p
        / "network"
        / "network_manifest.json"
    )

    network = orch.load_finalized_network(
        manifest_path
    )

    man = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    dates = [
        str(x)
        for x in man["dates"]
    ]

    date_to_index = {
        d: i
        for i, d in enumerate(dates)
    }

    edges = [
        (
            date_to_index[str(item["date_i"])],
            date_to_index[str(item["date_j"])],
        )
        for item in network
    ]

    return (
        dates,
        edges,
        network,
    )


def prepare(
    project,
    data_root,
    proc,
    support,
):
    project = Path(
        project
    ).resolve()

    data_root = Path(
        data_root
    ).resolve()

    proc = Path(
        proc
    ).resolve()

    support = Path(
        support
    ).resolve()

    support.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        dates,
        edges,
        network,
    ) = _dates_edges(
        proc
    )

    network_log_dir = (
        support
        / "network_logs"
    )

    orch.write_compat_network_logs(
        network,
        network_log_dir,
    )

    contract_path = (
        support
        / "baseline_source_contract.json"
    )

    search_roots = []

    for root in (
        project,
        data_root,
    ):
        if (
            root.is_dir()
            and
            root not in search_roots
        ):
            search_roots.append(
                root
            )

    if not search_roots:
        raise RuntimeError(
            "No valid SCLA baseline search roots."
        )

    payload = orch.build_baseline_contract(
        network=network,
        search_roots=search_roots,
        output_path=contract_path,
        exclude_roots=(
            support,
        ),
    )

    if not contract_path.is_file():
        raise RuntimeError(
            "build_baseline_contract did not create "
            f"{contract_path}"
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "build_baseline_contract did not return a dict."
        )

    network_ifgs = int(
        payload.get(
            "network_ifgs",
            -1,
        )
    )

    if network_ifgs != len(
        network
    ):
        raise RuntimeError(
            "SCLA baseline contract network count mismatch: "
            f"{network_ifgs} != {len(network)}"
        )

    original_pairs = int(
        payload.get(
            "original_pairs",
            -1,
        )
    )

    missing = list(
        payload.get(
            "missing",
            [],
        )
    )

    if (
        original_pairs
        +
        len(missing)
        !=
        len(network)
    ):
        raise RuntimeError(
            "SCLA baseline source accounting mismatch: "
            f"original={original_pairs}, "
            f"missing={len(missing)}, "
            f"network={len(network)}"
        )

    return {
        "dates":
            dates,

        "edges":
            edges,

        "network":
            network,

        "network_log_dir":
            network_log_dir,

        "contract_path":
            contract_path,

        "baseline_contract":
            payload,

        "original_pairs":
            original_pairs,

        "missing_pairs":
            len(missing),
    }
