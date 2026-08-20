from __future__ import annotations

import platform
import shutil
import sys


def main():
    print("=" * 80)
    print("pyPSDS-GAMMA v0.9 environment audit")
    print("=" * 80)
    print("Python     :", sys.version.replace("\n", " "))
    print("Platform   :", platform.platform())

    failures = []
    for name in ("numpy", "scipy", "numba", "llvmlite", "yaml", "matplotlib", "threadpoolctl"):
        try:
            mod = __import__(name)
            print(f"{name:12s}: {getattr(mod, '__version__', 'OK')}")
        except Exception as exc:
            failures.append((name, repr(exc)))
            print(f"{name:12s}: ERROR {exc}")

    try:
        import numba
        print("Numba threads:", numba.get_num_threads())
    except Exception:
        pass

    print("\nGAMMA commands:")
    for cmd in ("SLC2pt", "data2pt", "phase_sim_orb_pt"):
        print(f"  {cmd:18s}: {shutil.which(cmd) or 'NOT FOUND'}")

    if failures:
        raise SystemExit("Environment audit failed; see errors above.")
    print("\nENVIRONMENT PASS")


if __name__ == "__main__":
    main()
