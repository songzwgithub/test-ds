# Reproducibility

pyPSDS-GAMMA separates frozen scientific defaults from study-area inputs. The distributed package does not contain a fixed data directory, acquisition count, co-registration reference date, radar-height file, network edge count, or spatial reference window.

Each project must explicitly provide or verify RSLC/RSLC_tab, the GAMMA co-registration reference acquisition, radar-coordinate height geometry, the reference radar window, and any intentional network-baseline changes. `pypsds doctor` fails when required project choices are missing.

CPU/RAM planning uses the current machine and actual acquisition count when runtime overrides are omitted. All production stages are installed under `pypsds.stages`; production execution does not depend on a source checkout or the repository `tools/` directory.

Maintainer checks:

```bash
python tools/release_gate.py tests
python tools/release_gate.py wheel
python tools/release_gate.py contract --config /path/to/completed/project/pypsds.yaml
```

The contract is generated from the current completed project, so one study area's absolute output path or network edge count is not a portable source constant.

Exact bitwise identity across arbitrary CPU architectures, BLAS/LAPACK builds, compilers and operating systems is not promised. Scientific reproducibility is defined by explicit project configuration, frozen algorithm parameters, output-contract validation and numerical regression.
