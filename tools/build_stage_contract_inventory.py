#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from pypsds.config import load_config
from pypsds.pipeline import STAGES
from pypsds.project import resolve_project_paths


# ======================================================================
# Symbolic path roots
# ======================================================================

SPECIAL_ATTRS = {
    "paths.output_dir": "$OUTPUT",
    "paths.rslc_dir": "$RSLC",
    "paths.rslc_tab": "$RSLC_TAB",
    "paths.data_dir": "$DATA",
    "config_path": "$CONFIG",
}


FILE_SUFFIXES = (
    ".npy",
    ".npz",
    ".dat",
    ".bin",
    ".csv",
    ".json",
    ".txt",
    ".itab",
    ".mat",
    ".par",
    ".hgt",
    ".gd",
    ".grd",
)


@dataclass
class FileRef:
    kind: str
    symbolic_path: str
    output_relative: Optional[str]
    api: str
    line: int
    exists: Optional[bool]
    match_count: Optional[int]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


def dotted_name(node: ast.AST) -> Optional[str]:

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        left = dotted_name(node.value)

        if left is None:
            return None

        return f"{left}.{node.attr}"

    return None


def string_value(node: ast.AST) -> Optional[str]:

    if isinstance(node, ast.Constant):

        if isinstance(node.value, str):
            return node.value

    return None


class PathResolver:

    def __init__(self):
        self.env: dict[str, str] = {}

    def resolve(
        self,
        node: ast.AST,
    ) -> Optional[str]:

        # --------------------------------------------------------------
        # Direct string
        # --------------------------------------------------------------

        if isinstance(node, ast.Constant):

            if isinstance(node.value, str):
                return node.value

            return None

        # --------------------------------------------------------------
        # Known variable
        # --------------------------------------------------------------

        if isinstance(node, ast.Name):

            if node.id in self.env:
                return self.env[node.id]

            if node.id in SPECIAL_ATTRS:
                return SPECIAL_ATTRS[node.id]

            return None

        # --------------------------------------------------------------
        # paths.output_dir etc.
        # --------------------------------------------------------------

        if isinstance(node, ast.Attribute):

            d = dotted_name(node)

            if d in SPECIAL_ATTRS:
                return SPECIAL_ATTRS[d]

            return None

        # --------------------------------------------------------------
        # Path construction with /
        # --------------------------------------------------------------

        if (
            isinstance(node, ast.BinOp)
            and
            isinstance(node.op, ast.Div)
        ):

            left = self.resolve(node.left)
            right = self.resolve(node.right)

            if (
                left is not None
                and
                right is not None
            ):

                return (
                    left.rstrip("/")
                    +
                    "/"
                    +
                    right.lstrip("/")
                )

            return None

        # --------------------------------------------------------------
        # f"..."
        #
        # Unknown fields become "*", which is useful for dynamic IFG
        # products.
        # --------------------------------------------------------------

        if isinstance(node, ast.JoinedStr):

            pieces = []

            for x in node.values:

                if isinstance(x, ast.Constant):

                    pieces.append(
                        str(x.value)
                    )

                elif isinstance(
                    x,
                    ast.FormattedValue,
                ):

                    v = self.resolve(
                        x.value
                    )

                    if v is None:
                        pieces.append("*")
                    else:
                        pieces.append(v)

                else:
                    pieces.append("*")

            return "".join(pieces)

        # --------------------------------------------------------------
        # Path(...)
        # --------------------------------------------------------------

        if isinstance(node, ast.Call):

            func = dotted_name(
                node.func
            )

            if func in (
                "Path",
                "pathlib.Path",
            ):

                if not node.args:
                    return None

                return self.resolve(
                    node.args[0]
                )

            # path.resolve()
            if (
                isinstance(node.func, ast.Attribute)
                and
                node.func.attr
                in (
                    "resolve",
                    "absolute",
                    "expanduser",
                )
            ):

                return self.resolve(
                    node.func.value
                )

            # with_suffix()
            if (
                isinstance(node.func, ast.Attribute)
                and
                node.func.attr
                ==
                "with_suffix"
                and
                node.args
            ):

                base = self.resolve(
                    node.func.value
                )

                suffix = self.resolve(
                    node.args[0]
                )

                if (
                    base is not None
                    and
                    suffix is not None
                ):

                    return str(
                        Path(base).with_suffix(
                            suffix
                        )
                    )

        return None

    def populate(
        self,
        tree: ast.AST,
    ) -> None:
        """
        Iteratively solve simple path assignments.

        Multiple passes handle:
            root = ...
            outdir = root / ...
            file = outdir / ...
        """

        assignments = []

        for node in ast.walk(tree):

            if isinstance(
                node,
                (
                    ast.Assign,
                    ast.AnnAssign,
                ),
            ):

                assignments.append(
                    node
                )

        for _ in range(12):

            changed = False

            for node in assignments:

                if isinstance(
                    node,
                    ast.Assign,
                ):

                    if len(
                        node.targets
                    ) != 1:
                        continue

                    target = (
                        node.targets[0]
                    )

                    value = node.value

                else:

                    target = node.target
                    value = node.value

                    if value is None:
                        continue

                if not isinstance(
                    target,
                    ast.Name,
                ):
                    continue

                result = self.resolve(
                    value
                )

                if result is None:
                    continue

                if (
                    self.env.get(
                        target.id
                    )
                    != result
                ):

                    self.env[
                        target.id
                    ] = result

                    changed = True

            if not changed:
                break


def keyword_value(
    node: ast.Call,
    name: str,
) -> Optional[str]:

    for kw in node.keywords:

        if kw.arg == name:
            return string_value(
                kw.value
            )

    return None


def call_mode(
    node: ast.Call,
    *,
    default: str = "r",
) -> str:

    m = keyword_value(
        node,
        "mode",
    )

    if m is not None:
        return m

    if len(node.args) >= 2:

        x = string_value(
            node.args[1]
        )

        if x is not None:
            return x

    return default


def path_method_mode(
    node: ast.Call,
    *,
    default: str = "r",
) -> str:

    m = keyword_value(
        node,
        "mode",
    )

    if m is not None:
        return m

    if len(node.args) >= 1:

        x = string_value(
            node.args[0]
        )

        if x is not None:
            return x

    return default


def classify_mode(
    mode: str,
) -> str:

    if any(
        x in mode
        for x in (
            "w",
            "a",
            "x",
            "+",
        )
    ):
        return "output"

    return "input"


def output_relative(
    symbolic: str,
) -> Optional[str]:

    prefix = "$OUTPUT/"

    if symbolic.startswith(
        prefix
    ):
        return symbolic[
            len(prefix):
        ]

    return None


def path_status(
    output_root: Path,
    rel: Optional[str],
):

    if rel is None:
        return None, None

    if "*" in rel or "?" in rel:

        matches = list(
            output_root.glob(
                rel
            )
        )

        return (
            len(matches) > 0,
            len(matches),
        )

    p = (
        output_root
        / rel
    )

    return (
        p.exists(),
        1 if p.exists() else 0,
    )


def looks_like_file(
    p: str,
) -> bool:

    low = p.lower()

    return (
        any(
            low.endswith(x)
            for x in FILE_SUFFIXES
        )
        or
        "*" in low
    )


def resolve_first_arg(
    resolver: PathResolver,
    node: ast.Call,
) -> Optional[str]:

    if not node.args:
        return None

    return resolver.resolve(
        node.args[0]
    )


def infer_helper_io(
    tree: ast.AST,
):
    """
    Infer path roles of simple helper-function parameters.

    Examples:

        optional_load(path)
            -> path is INPUT

        write_itab(path, ...)
            -> path is OUTPUT

        open_or_create_npy(path, ...)
            -> path is INPUT + OUTPUT

    The actual path is subsequently recovered at each call site.
    """

    helpers = {}

    # Calls inside helper definitions that are successfully
    # represented by a formal path parameter. They should not
    # later be reported as unresolved raw I/O.
    covered_calls = set()

    read_funcs = {
        "np.load",
        "numpy.load",
        "np.fromfile",
        "numpy.fromfile",
    }

    write_funcs = {
        "np.save",
        "numpy.save",
        "np.savez",
        "numpy.savez",
        "np.savez_compressed",
        "numpy.savez_compressed",
        "np.savetxt",
        "numpy.savetxt",
    }

    for fn in ast.walk(tree):

        if not isinstance(
            fn,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            continue

        params = [
            x.arg
            for x in fn.args.args
        ]

        roles = {
            x: set()
            for x in params
        }

        def formal(expr):

            if (
                isinstance(expr, ast.Name)
                and
                expr.id in roles
            ):
                return expr.id

            return None

        for node in ast.walk(fn):

            if not isinstance(
                node,
                ast.Call,
            ):
                continue

            func = dotted_name(
                node.func
            )

            # ------------------------------------------------
            # np.load / np.fromfile
            # ------------------------------------------------

            if (
                func in read_funcs
                and
                node.args
            ):

                q = formal(
                    node.args[0]
                )

                if q is not None:

                    roles[q].add(
                        "input"
                    )

                    covered_calls.add(
                        id(node)
                    )

                continue

            # ------------------------------------------------
            # np.save / np.savez ...
            # ------------------------------------------------

            if (
                func in write_funcs
                and
                node.args
            ):

                q = formal(
                    node.args[0]
                )

                if q is not None:

                    roles[q].add(
                        "output"
                    )

                    covered_calls.add(
                        id(node)
                    )

                continue

            # ------------------------------------------------
            # np.lib.format.open_memmap(path, mode=...)
            # ------------------------------------------------

            if func in (
                "np.lib.format.open_memmap",
                "numpy.lib.format.open_memmap",
            ):

                if node.args:

                    q = formal(
                        node.args[0]
                    )

                    if q is not None:

                        roles[q].add(
                            classify_mode(
                                call_mode(
                                    node,
                                    default="r+",
                                )
                            )
                        )

                        covered_calls.add(
                            id(node)
                        )

                continue

            # ------------------------------------------------
            # builtin open(path, mode)
            # ------------------------------------------------

            if func == "open":

                if node.args:

                    q = formal(
                        node.args[0]
                    )

                    if q is not None:

                        roles[q].add(
                            classify_mode(
                                call_mode(
                                    node,
                                    default="r",
                                )
                            )
                        )

                        covered_calls.add(
                            id(node)
                        )

                continue

            # ------------------------------------------------
            # Path parameter methods
            # ------------------------------------------------

            if isinstance(
                node.func,
                ast.Attribute,
            ):

                q = formal(
                    node.func.value
                )

                if q is None:
                    continue

                method = (
                    node.func.attr
                )

                if method in (
                    "read_text",
                    "read_bytes",
                ):

                    roles[q].add(
                        "input"
                    )

                    covered_calls.add(
                        id(node)
                    )

                    continue

                if method in (
                    "write_text",
                    "write_bytes",
                ):

                    roles[q].add(
                        "output"
                    )

                    covered_calls.add(
                        id(node)
                    )

                    continue

                if method == "open":

                    roles[q].add(
                        classify_mode(
                            path_method_mode(
                                node,
                                default="r",
                            )
                        )
                    )

                    covered_calls.add(
                        id(node)
                    )

                    continue

                if method == "tofile":

                    # ndarray.tofile(path) has the path as arg,
                    # not receiver, so handled separately below.
                    pass

            # ------------------------------------------------
            # shutil.copy*(src, dst)
            # ------------------------------------------------

            if func in (
                "shutil.copy",
                "shutil.copy2",
                "shutil.copyfile",
            ):

                if len(node.args) >= 1:

                    q = formal(
                        node.args[0]
                    )

                    if q is not None:

                        roles[q].add(
                            "input"
                        )

                        covered_calls.add(
                            id(node)
                        )

                if len(node.args) >= 2:

                    q = formal(
                        node.args[1]
                    )

                    if q is not None:

                        roles[q].add(
                            "output"
                        )

                        covered_calls.add(
                            id(node)
                        )

        useful = {
            name:
                tuple(
                    sorted(
                        value
                    )
                )
            for name, value
            in roles.items()
            if value
        }

        if useful:

            helpers[
                fn.name
            ] = {
                "params":
                    params,

                "roles":
                    useful,
            }

    return (
        helpers,
        covered_calls,
    )


def inspect_script(
    script: Path,
    output_root: Path,
):

    source = script.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(script),
    )

    resolver = PathResolver()

    resolver.populate(
        tree
    )

    (
        helper_specs,
        covered_helper_io,
    ) = infer_helper_io(
        tree
    )

    refs: list[FileRef] = []

    unresolved_io = []

    def add_ref(
        kind: str,
        symbolic: Optional[str],
        api: str,
        line: int,
    ):

        if symbolic is None:

            unresolved_io.append({
                "kind":
                    kind,

                "api":
                    api,

                "line":
                    line,
            })

            return

        if not looks_like_file(
            symbolic
        ):
            return

        rel = output_relative(
            symbolic
        )

        exists, nmatch = path_status(
            output_root,
            rel,
        )

        refs.append(
            FileRef(
                kind=kind,
                symbolic_path=symbolic,
                output_relative=rel,
                api=api,
                line=line,
                exists=exists,
                match_count=nmatch,
            )
        )

    for node in ast.walk(tree):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        func = dotted_name(
            node.func
        )

        # ==============================================================
        # Simple helper-function path propagation
        # ==============================================================

        if func in helper_specs:

            spec = (
                helper_specs[
                    func
                ]
            )

            params = (
                spec[
                    "params"
                ]
            )

            actual = {}

            for i, arg in enumerate(
                node.args
            ):

                if i < len(params):

                    actual[
                        params[i]
                    ] = arg

            for kw in node.keywords:

                if (
                    kw.arg is not None
                    and
                    kw.arg in params
                ):

                    actual[
                        kw.arg
                    ] = kw.value

            for param, roles in (
                spec[
                    "roles"
                ].items()
            ):

                expr = actual.get(
                    param
                )

                if expr is None:
                    continue

                symbolic = (
                    resolver.resolve(
                        expr
                    )
                )

                for role in roles:

                    add_ref(
                        role,
                        symbolic,
                        (
                            f"helper:"
                            f"{func}:"
                            f"{param}"
                        ),
                        node.lineno,
                    )

        # ==============================================================
        # NumPy reads
        # ==============================================================

        if func in (
            "np.load",
            "numpy.load",
            "np.fromfile",
            "numpy.fromfile",
        ):

            if id(node) in covered_helper_io:
                continue

            add_ref(
                "input",
                resolve_first_arg(
                    resolver,
                    node,
                ),
                func,
                node.lineno,
            )

            continue

        # ==============================================================
        # NumPy writes
        # ==============================================================

        if func in (
            "np.save",
            "numpy.save",
            "np.savez",
            "numpy.savez",
            "np.savez_compressed",
            "numpy.savez_compressed",
            "np.savetxt",
            "numpy.savetxt",
        ):

            if id(node) in covered_helper_io:
                continue

            add_ref(
                "output",
                resolve_first_arg(
                    resolver,
                    node,
                ),
                func,
                node.lineno,
            )

            continue

        # ==============================================================
        # open_memmap
        # ==============================================================

        if func in (
            "np.lib.format.open_memmap",
            "numpy.lib.format.open_memmap",
        ):

            if id(node) in covered_helper_io:
                continue

            mode = call_mode(
                node,
                default="r+",
            )

            add_ref(
                classify_mode(
                    mode
                ),
                resolve_first_arg(
                    resolver,
                    node,
                ),
                func,
                node.lineno,
            )

            continue

        # ==============================================================
        # Builtin open()
        # ==============================================================

        if func == "open":

            if id(node) in covered_helper_io:
                continue

            mode = call_mode(
                node,
                default="r",
            )

            add_ref(
                classify_mode(
                    mode
                ),
                resolve_first_arg(
                    resolver,
                    node,
                ),
                "open",
                node.lineno,
            )

            continue

        # ==============================================================
        # Path methods
        # ==============================================================

        if isinstance(
            node.func,
            ast.Attribute,
        ):

            method = (
                node.func.attr
            )

            receiver = (
                resolver.resolve(
                    node.func.value
                )
            )

            if method in (
                "read_text",
                "read_bytes",
            ):

                add_ref(
                    "input",
                    receiver,
                    f"Path.{method}",
                    node.lineno,
                )

                continue

            if method in (
                "write_text",
                "write_bytes",
            ):

                add_ref(
                    "output",
                    receiver,
                    f"Path.{method}",
                    node.lineno,
                )

                continue

            if method == "open":

                if id(node) in covered_helper_io:
                    continue

                mode = path_method_mode(
                    node,
                    default="r",
                )

                add_ref(
                    classify_mode(
                        mode
                    ),
                    receiver,
                    "Path.open",
                    node.lineno,
                )

                continue

            if method == "tofile":

                add_ref(
                    "output",
                    resolve_first_arg(
                        resolver,
                        node,
                    ),
                    "ndarray.tofile",
                    node.lineno,
                )

                continue

        # ==============================================================
        # shutil copies
        # ==============================================================

        if func in (
            "shutil.copy",
            "shutil.copy2",
            "shutil.copyfile",
        ):

            if len(node.args) >= 1:

                add_ref(
                    "input",
                    resolver.resolve(
                        node.args[0]
                    ),
                    func,
                    node.lineno,
                )

            if len(node.args) >= 2:

                add_ref(
                    "output",
                    resolver.resolve(
                        node.args[1]
                    ),
                    func,
                    node.lineno,
                )

    # ------------------------------------------------------------------
    # De-duplicate
    # ------------------------------------------------------------------

    unique = {}

    for r in refs:

        key = (
            r.kind,
            r.symbolic_path,
        )

        if key not in unique:
            unique[key] = r

    refs = sorted(
        unique.values(),
        key=lambda x: (
            x.kind,
            x.symbolic_path,
        ),
    )

    produced = {
        x.symbolic_path
        for x in refs
        if (
            x.kind == "output"
            and
            x.symbolic_path.startswith(
                "$OUTPUT/"
            )
        )
    }

    refs = [
        x
        for x in refs
        if not (
            x.kind == "input"
            and
            x.symbolic_path in produced
        )
    ]

    exact_inputs = [
        r
        for r in refs
        if (
            r.kind == "input"
            and
            r.output_relative is not None
            and
            "*" not in r.output_relative
            and
            "?" not in r.output_relative
        )
    ]

    exact_outputs = [
        r
        for r in refs
        if (
            r.kind == "output"
            and
            r.output_relative is not None
            and
            "*" not in r.output_relative
            and
            "?" not in r.output_relative
        )
    ]

    pattern_refs = [
        r
        for r in refs
        if (
            r.output_relative is not None
            and
            (
                "*" in r.output_relative
                or
                "?" in r.output_relative
            )
        )
    ]

    missing_exact = [
        r
        for r in (
            exact_inputs
            +
            exact_outputs
        )
        if r.exists is False
    ]

    # AUTO_READY intentionally conservative.
    #
    # We want:
    #   - at least one exact output,
    #   - no unresolved I/O operation,
    #   - no missing exact declared current-product path,
    #   - no dynamic output pattern requiring human review.
    unmatched_patterns = [
        r
        for r in pattern_refs
        if r.exists is False
    ]

    has_output_candidate = (
        len(exact_outputs) > 0
        or
        any(
            r.kind == "output"
            and
            r.exists is True
            for r in pattern_refs
        )
    )

    auto_ready = (
        has_output_candidate
        and
        len(unresolved_io) == 0
        and
        len(missing_exact) == 0
        and
        len(unmatched_patterns) == 0
    )

    return {
        "script":
            script.name,

        "script_sha256":
            sha256_file(
                script
            ),

        "status":
            (
                "AUTO_READY"
                if auto_ready
                else "REVIEW"
            ),

        "resolved_file_refs":
            [
                asdict(x)
                for x in refs
            ],

        "exact_output_inputs":
            [
                x.output_relative
                for x in exact_inputs
            ],

        "exact_output_outputs":
            [
                x.output_relative
                for x in exact_outputs
            ],

        "dynamic_refs":
            [
                asdict(x)
                for x in pattern_refs
            ],

        "unresolved_io":
            unresolved_io,

        "missing_exact":
            [
                asdict(x)
                for x in missing_exact
            ],
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        required=True,
    )

    ap.add_argument(
        "--json",
        default=(
            "docs/release/"
            "stage_contract_inventory.json"
        ),
    )

    ap.add_argument(
        "--text",
        default=(
            "docs/release/"
            "stage_contract_inventory.txt"
        ),
    )

    args = ap.parse_args()

    cfg, config_path = load_config(
        args.config
    )

    paths = resolve_project_paths(
        cfg,
        config_path,
    )

    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    output_root = Path(
        paths.output_dir
    )

    print(
        "=" * 108
    )

    print(
        "pyPSDS-GAMMA automatic StageContract inventory"
    )

    print(
        "=" * 108
    )

    print(
        "config      :",
        config_path,
    )

    print(
        "output root :",
        output_root,
    )

    print(
        "stages      :",
        len(STAGES),
    )

    inventory = {
        "format":
            "pyPSDS-GAMMA-stage-contract-inventory-v1",

        "config":
            str(
                config_path
            ),

        "output_root":
            str(
                output_root
            ),

        "stage_count":
            len(STAGES),

        "stages":
            {},
    }

    text_lines = []

    auto_count = 0
    review_count = 0

    for i, stage in enumerate(
        STAGES,
        start=1,
    ):

        script = (
            root
            / "scripts"
            / stage.script
        )

        if not script.is_file():

            result = {
                "script":
                    stage.script,

                "status":
                    "MISSING_SCRIPT",
            }

            inventory[
                "stages"
            ][
                stage.name
            ] = result

            review_count += 1

            print(
                f"{i:02d} "
                f"{stage.name:38s} "
                f"MISSING_SCRIPT"
            )

            continue

        result = inspect_script(
            script,
            output_root,
        )

        inventory[
            "stages"
        ][
            stage.name
        ] = result

        status = (
            result[
                "status"
            ]
        )

        if status == "AUTO_READY":
            auto_count += 1
        else:
            review_count += 1

        nin = len(
            result[
                "exact_output_inputs"
            ]
        )

        nout = len(
            result[
                "exact_output_outputs"
            ]
        )

        ndyn = len(
            result[
                "dynamic_refs"
            ]
        )

        nun = len(
            result[
                "unresolved_io"
            ]
        )

        nmiss = len(
            result[
                "missing_exact"
            ]
        )

        line = (
            f"{i:02d} "
            f"{stage.name:38s} "
            f"{status:10s} "
            f"IN={nin:2d} "
            f"OUT={nout:2d} "
            f"DYN={ndyn:2d} "
            f"UNRES={nun:2d} "
            f"MISS={nmiss:2d}"
        )

        print(
            line
        )

        text_lines.append(
            line
        )

    inventory[
        "summary"
    ] = {
        "total":
            len(STAGES),

        "auto_ready":
            auto_count,

        "review":
            review_count,
    }

    json_path = (
        root
        / args.json
    )

    text_path = (
        root
        / args.text
    )

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path.write_text(
        json.dumps(
            inventory,
            indent=2,
            ensure_ascii=False,
        )
        +
        "\n",
        encoding="utf-8",
    )

    text_path.write_text(
        "\n".join(
            text_lines
        )
        +
        "\n\n"
        +
        f"TOTAL      : {len(STAGES)}\n"
        +
        f"AUTO_READY : {auto_count}\n"
        +
        f"REVIEW     : {review_count}\n",
        encoding="utf-8",
    )

    print()
    print(
        "=" * 108
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 108
    )

    print(
        "TOTAL      :",
        len(STAGES),
    )

    print(
        "AUTO_READY :",
        auto_count,
    )

    print(
        "REVIEW     :",
        review_count,
    )

    print(
        "JSON       :",
        json_path,
    )

    print(
        "TEXT       :",
        text_path,
    )

    # Production stage count is defined by STAGES itself.
    # Do not freeze a historical numeric stage count here.
    stage_names = [
        stage.name
        for stage in STAGES
    ]

    if len(stage_names) != len(set(stage_names)):
        raise SystemExit(
            "Duplicate production stage names detected."
        )

    if set(inventory["stages"]) != set(stage_names):
        raise SystemExit(
            "Stage inventory does not match current STAGES."
        )


if __name__ == "__main__":
    main()
