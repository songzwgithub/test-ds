# pyPSDS-GAMMA v1.1 Geometry source adaptation audit

This report statically analyzes the frozen authoritative Geometry/Incidence development implementations. No source was imported or executed.

## point_geolocation

- Source: `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/authoritative_sources/production/geometry/point_geolocation.py`
- SHA256: `1a252a48fbdfc005a1126ae630b1014c30a217a7b50b8b1e96efc29cd6ed3673`
- Lines: **375**
- Functions: **3**
- Main guard: **False**
- Top-level executable blocks: **40**
- Absolute path hits: **2**
- Development-name hits: **11**
- External process calls: **0**

### Imports

- L1: `pathlib.Path`
- L2: `json`
- L3: `math`
- L4: `numpy as np`

### Functions

- L18-24: `read_par(path)`
- L27-28: `val(d, key)`
- L108-121: `stamps_incidence(rg)`

### Absolute paths

- L6: `/home/ubuntu/Downloads/psds`
- L11: `/home/ubuntu/Downloads/RSLC/20151212.rslc.par`

### External/GAMMA calls

- L105: `incidence_angle`
- L280: `incidence_angle`
- L105: `incidence_angle`
- L281: `incidence_angle`
- L299: `data2pt`

### File I/O calls

- L6: `Path("/home/ubuntu/Downloads/psds")`
- L10: `Path(
    "/home/ubuntu/Downloads/RSLC/20151212.rslc.par"
)`
- L32: `np.load(
        PROC / "network_inversion" / "strict_point_ids.npy"
    )`
- L38: `np.load(
    PROC / "point_phase_stack" / "rows.npy",
    mmap_mode="r",
)`
- L43: `np.load(
    PROC / "point_phase_stack" / "cols.npy",
    mmap_mode="r",
)`
- L255: `np.save(
    DEST / "strict_point_ids.npy",
    strict_ids.astype(np.int32),
)`
- L260: `np.save(
    DEST / "radar_row.npy",
    rows,
)`
- L265: `np.save(
    DEST / "radar_col.npy",
    cols,
)`
- L270: `np.save(
    DEST / "longitude_deg.npy",
    lon,
)`
- L275: `np.save(
    DEST / "latitude_deg.npy",
    lat,
)`
- L280: `np.save(
    DEST / "incidence_angle_stamps_deg.npy",
    inc.astype(np.float32),
)`
- L285: `np.save(
    DEST / "valid_gacos_geometry_mask.npy",
    accepted,
)`
- L355: `json.dumps(
        manifest,
        indent=2,
    )`

### Top-level execution

- L53-53: `Assert`
- L70-70: `Assert`
- L71-71: `Assert`
- L170-170: `Expr`
- L171-171: `Expr`
- L172-172: `Expr`
- L174-174: `Expr`
- L175-175: `Expr`
- L176-176: `Expr`
- L177-177: `Expr`
- L179-183: `Expr`
- L185-189: `Expr`
- L191-191: `Expr`
- L192-195: `Expr`
- L197-200: `Expr`
- L202-205: `Expr`
- L207-210: `Expr`
- L212-212: `Expr`
- L214-216: `Expr`
- L218-225: `Expr`
- L227-230: `Expr`
- L232-235: `Expr`
- L238-239: `If`
- L241-242: `If`
- L244-245: `If`
- L247-248: `If`
- L255-258: `Expr`
- L260-263: `Expr`
- L265-268: `Expr`
- L270-273: `Expr`
- L275-278: `Expr`
- L280-283: `Expr`
- L285-288: `Expr`
- L354-360: `Expr`
- L363-363: `Expr`
- L364-367: `Expr`
- L369-369: `Expr`
- L370-370: `Expr`
- L371-374: `Expr`
- L375-375: `Expr`

### Development-name locations

- L171: `P15-3A v4b StaMPS/GAMMA geometry finalize`
- L293: `pyPSDS-GAMMA-P15-3A-StaMPS-geometry-v4b`
- L296: `PASS_RADAR_POINT_GEOLOCATION`
- L345: `P15-4_GACOS_POINT_SAMPLING_SMOKE`
- L372: `P15-3A FINAL RESULT: PASS_RADAR_POINT_GEOLOCATION`
- L171: `print("P15-3A v4b StaMPS/GAMMA geometry finalize")`
- L293: `"pyPSDS-GAMMA-P15-3A-StaMPS-geometry-v4b",`
- L296: `"PASS_RADAR_POINT_GEOLOCATION",`
- L345: `"P15-4_GACOS_POINT_SAMPLING_SMOKE",`
- L372: `"P15-3A FINAL RESULT: "`
- L373: `"PASS_RADAR_POINT_GEOLOCATION"`

### Migration assessment

- Script harness: `True`
- Reusable functions present: `True`
- Path parameterization required: `True`
- Public naming cleanup required: `True`
- External process integration required: `False`

## incidence

- Source: `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/authoritative_sources/production/geometry/incidence.py`
- SHA256: `14d749c8b9f744f8b256ecf81908ba21cf26e697a7dbc28507d612880b543034`
- Lines: **912**
- Functions: **5**
- Main guard: **False**
- Top-level executable blocks: **20**
- Absolute path hits: **2**
- Development-name hits: **12**
- External process calls: **0**

### Imports

- L1: `pathlib.Path`
- L2: `json`
- L3: `re`
- L4: `time`
- L6: `numpy as np`
- L7: `numba.njit`
- L7: `numba.prange`

### Functions

- L38-56: `read_par(path)`
- L59-65: `scalar(d, key)`
- L68-81: `vec3(d, key)`
- L256-360: `orbit_position(query_time)`
- L408-572: `incidence_fast(lon_deg, lat_deg, hgt_m, radar_row, sat_xyz, a, e2)`

### Absolute paths

- L11: `/home/ubuntu/Downloads/psds/output/processing/gacos_geometry`
- L15: `/home/ubuntu/Downloads/RSLC/20151212.rslc.par`

### External/GAMMA calls

- L864: `gc_map2`
- L864: `gc_map`
- L867: `geocode`
- L870: `data2pt`

### File I/O calls

- L10: `Path(
    "/home/ubuntu/Downloads/psds/output/processing/gacos_geometry"
)`
- L14: `Path(
    "/home/ubuntu/Downloads/RSLC/20151212.rslc.par"
)`
- L89: `np.load(
        ROOT / "longitude_deg.npy",
        mmap_mode="r",
    )`
- L97: `np.load(
        ROOT / "latitude_deg.npy",
        mmap_mode="r",
    )`
- L105: `np.load(
        ROOT / "radar_row.npy",
        mmap_mode="r",
    )`
- L808: `np.save(
    OUT,
    inc,
)`
- L886: `json.dumps(
        manifest,
        indent=2,
    )`

### Top-level execution

- L122-133: `If`
- L159-201: `If`
- L643-643: `Expr`
- L644-644: `Expr`
- L645-645: `Expr`
- L647-650: `Expr`
- L652-655: `Expr`
- L657-660: `Expr`
- L662-665: `Expr`
- L667-674: `Expr`
- L697-801: `If`
- L808-811: `Expr`
- L885-892: `Expr`
- L895-895: `Expr`
- L896-899: `Expr`
- L901-904: `Expr`
- L906-906: `Expr`
- L907-907: `Expr`
- L908-911: `Expr`
- L912-912: `Expr`

### Development-name locations

- L644: `P15-3H FINAL FAST GAMMA-COMPATIBLE INCIDENCE`
- L794: `PASS_STRONG`
- L816: `PASS_FAST_GAMMA_COMPATIBLE_INCIDENCE`
- L881: `P15-4_FAST_GACOS_POINT_SAMPLING`
- L909: `P15-3H FINAL RESULT: PASS_FAST_GAMMA_COMPATIBLE_INCIDENCE`
- L591: `# Full benchmark`
- L644: `print("P15-3H FINAL FAST GAMMA-COMPATIBLE INCIDENCE")`
- L794: `"PASS_STRONG"`
- L816: `"PASS_FAST_GAMMA_COMPATIBLE_INCIDENCE",`
- L881: `"P15-4_FAST_GACOS_POINT_SAMPLING",`
- L909: `"P15-3H FINAL RESULT: "`
- L910: `"PASS_FAST_GAMMA_COMPATIBLE_INCIDENCE"`

### Migration assessment

- Script harness: `True`
- Reusable functions present: `True`
- Path parameterization required: `True`
- Public naming cleanup required: `True`
- External process integration required: `False`

## Proposed v1.1 package destination

```text
pypsds/
└── geometry/
    ├── __init__.py
    ├── point_geolocation.py
    └── incidence.py
```

Development scripts remain frozen under `docs/release_v1_1/authoritative_sources/`; production code must not import them.
