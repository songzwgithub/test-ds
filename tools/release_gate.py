#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile, venv, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def run(cmd, *, cwd=None):
    print("+", " ".join(map(str, cmd)))
    p = subprocess.run([str(x) for x in cmd], cwd=None if cwd is None else str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout, end="")
    if p.returncode != 0:
        raise RuntimeError(f"command failed with return code {p.returncode}")
    return p

def check_tests():
    run([sys.executable, "-m", "compileall", "-q", "pypsds", "tests", "tools"], cwd=ROOT)
    run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT)
    print("SOURCE TEST GATE: PASS")

def check_contract(config):
    config = Path(config).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="pypsds-contract-") as td:
        td = Path(td)
        inventory = td / "inventory.json"
        text = td / "inventory.txt"
        snapshot = td / "snapshot.json"
        run([sys.executable, str(ROOT / "tools" / "build_stage_contract_inventory.py"), "--config", str(config), "--json", str(inventory), "--text", str(text)], cwd=ROOT)

        from pypsds.pipeline import STAGE_CONTRACTS

        inventory_data = json.loads(inventory.read_text(encoding="utf-8"))
        enriched = []

        for stage_name, contract in STAGE_CONTRACTS.items():
            if not contract.validated or not contract.required_outputs:
                continue

            stage_info = inventory_data.get("stages", {}).get(stage_name)
            if stage_info is None:
                raise RuntimeError(
                    "validated StageContract missing from inventory: "
                    f"{stage_name}"
                )

            outputs = set(stage_info.get("exact_output_outputs", []))
            outputs.update(str(x) for x in contract.required_outputs)
            stage_info["exact_output_outputs"] = sorted(outputs)
            enriched.append((stage_name, len(contract.required_outputs)))

        inventory.write_text(
            json.dumps(inventory_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(
            "validated StageContract inventory enrichment:",
            ", ".join(f"{name}={count}" for name, count in enriched),
        )

        run([sys.executable, str(ROOT / "tools" / "freeze_stage_output_contracts.py"), "--config", str(config), "--inventory", str(inventory), "--snapshot", str(snapshot)], cwd=ROOT)
        run([sys.executable, str(ROOT / "tools" / "freeze_stage_output_contracts.py"), "--config", str(config), "--inventory", str(inventory), "--snapshot", str(snapshot), "--audit-only"], cwd=ROOT)
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        if data["stage_count"] != 38:
            raise RuntimeError(f"unexpected production stage count: {data['stage_count']}")
    print("DYNAMIC OUTPUT CONTRACT GATE: PASS")

def _venv_python(vdir):
    return vdir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

def check_wheel():
    with tempfile.TemporaryDirectory(prefix="pypsds-wheel-") as td:
        td = Path(td)
        wheel_dir = td / "wheel"
        wheel_dir.mkdir()
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", ".", "-w", str(wheel_dir)], cwd=ROOT)
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one wheel, found {len(wheels)}")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as zf:
            names = set(zf.namelist())
        required = {
            "pypsds/resources/default_config.yaml",
            "pypsds/resources/ds_production_policy_v1.json",
            "pypsds/stages/run_phase_linking.py",
            "pypsds/stages/build_exact_support_cache.py",
            "pypsds/stages/apply_reference.py",
            "pypsds/stages/build_point_geometry.py",
            "pypsds/stages/apply_atmosphere_correction.py",
            "pypsds/stages/run_scla.py",
            "pypsds/stages/run_scn.py",
            "pypsds/stages/build_final_los.py",
            "pypsds/stages/build_point_products.py",
            "pypsds/products/point_metrics.py",
            "pypsds/runtime_v11/gacos_runtime.py",
            "pypsds/runtime_v11/scn_runtime.py",
            "pypsds/runtime_v11/final_los_runtime.py",
            "pypsds/runtime_v11/point_metrics_runtime.py",
        }
        missing = sorted(required - names)
        if missing:
            raise RuntimeError("wheel missing production resources: " + ", ".join(missing))
        vdir = td / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(vdir)
        py = _venv_python(vdir)
        run([py, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)])
        smoke = td / "smoke"
        smoke.mkdir()
        code = '''from pathlib import Path
import importlib, sys, pypsds
pkg = Path(pypsds.__file__).resolve()
prefix = Path(sys.prefix).resolve()
assert pkg.is_relative_to(prefix), (pkg, prefix)
print(f'installed pypsds source: {pkg}')
from pypsds.pipeline import STAGES
assert pypsds.__version__ == "1.1.0"
assert len(STAGES) == 38
for stage in STAGES:
    importlib.import_module("pypsds.stages." + Path(stage.script).stem)
print("installed stage imports: PASS")
'''
        run([py, "-I", "-c", code], cwd=smoke)
        run([py, "-I", "-m", "pypsds.cli", "init", str(smoke / "project")], cwd=smoke)
        text = (smoke / "project" / "pypsds.yaml").read_text(encoding="utf-8")
        if "/home/" in text or "/mnt/" in text:
            raise RuntimeError("generated project config contains machine-specific paths")
    print("WHEEL / INSTALLED-PACKAGE GATE: PASS")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("tests")
    sub.add_parser("wheel")
    q = sub.add_parser("contract"); q.add_argument("--config", required=True)
    q = sub.add_parser("all"); q.add_argument("--config", required=True)
    args = ap.parse_args()
    if args.command == "tests": check_tests()
    elif args.command == "wheel": check_wheel()
    elif args.command == "contract": check_contract(args.config)
    elif args.command == "all":
        check_tests(); check_wheel(); check_contract(args.config)

if __name__ == "__main__": main()
