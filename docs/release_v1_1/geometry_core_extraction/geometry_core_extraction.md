# Geometry core extraction

This report separates reusable function definitions from the development-script harness without changing algorithm source.

## point_geolocation

### Functions and implicit dependencies

| Function | Lines | Top-level globals used | Other free names |
|---|---:|---|---|
| `read_par` | 18-24 | `` | `` |
| `val` | 27-28 | `` | `` |
| `stamps_incidence` | 108-121 | `re, se` | `` |

### Top-level block categories

| Category | Blocks |
|---|---:|
| `input` | 4 |
| `logging` | 25 |
| `output` | 8 |
| `setup_or_compute` | 32 |
| `validation` | 7 |

### Top-level blocks

| ID | Lines | Category | Writes | Important reads |
|---:|---:|---|---|---|
| 1 | 6-6 | `setup_or_compute` | `PROJECT` | `Path` |
| 2 | 7-7 | `setup_or_compute` | `PROC` | `PROJECT` |
| 3 | 8-8 | `setup_or_compute` | `DEST` | `PROC` |
| 4 | 10-12 | `setup_or_compute` | `RSLC_PAR` | `Path` |
| 5 | 14-14 | `setup_or_compute` | `lon_bin` | `DEST` |
| 6 | 15-15 | `setup_or_compute` | `lat_bin` | `DEST` |
| 7 | 31-36 | `input` | `strict_ids` | `PROC, np` |
| 8 | 38-41 | `input` | `all_rows` | `PROC, np` |
| 9 | 43-46 | `input` | `all_cols` | `PROC, np` |
| 10 | 48-48 | `setup_or_compute` | `rows` | `all_rows, np, strict_ids` |
| 11 | 49-49 | `setup_or_compute` | `cols` | `all_cols, np, strict_ids` |
| 12 | 51-51 | `setup_or_compute` | `n` | `strict_ids` |
| 13 | 53-53 | `validation` | `` | `n` |
| 14 | 60-63 | `setup_or_compute` | `lon` | `lon_bin, np` |
| 15 | 65-68 | `setup_or_compute` | `lat` | `lat_bin, np` |
| 16 | 70-70 | `validation` | `` | `lon, n` |
| 17 | 71-71 | `validation` | `` | `lat, n` |
| 18 | 74-81 | `setup_or_compute` | `valid_ll` | `lat, lon, np` |
| 19 | 83-89 | `setup_or_compute` | `inside_gacos` | `lat, lon, valid_ll` |
| 20 | 97-97 | `input` | `p` | `RSLC_PAR, read_par` |
| 21 | 99-99 | `setup_or_compute` | `rgn` | `p, val` |
| 22 | 100-100 | `setup_or_compute` | `rps` | `p, val` |
| 23 | 101-101 | `setup_or_compute` | `se` | `p, val` |
| 24 | 102-102 | `setup_or_compute` | `re` | `p, val` |
| 25 | 104-104 | `setup_or_compute` | `rgc` | `p, val` |
| 26 | 105-105 | `setup_or_compute` | `gamma_center_inc` | `p, val` |
| 27 | 127-130 | `setup_or_compute` | `rg` | `cols, np, rgn, rps` |
| 28 | 132-132 | `setup_or_compute` | `inc` | `rg, stamps_incidence` |
| 29 | 136-138 | `setup_or_compute` | `stamps_center_inc` | `float, rgc, stamps_incidence` |
| 30 | 140-143 | `setup_or_compute` | `center_difference` | `gamma_center_inc, stamps_center_inc` |
| 31 | 146-150 | `setup_or_compute` | `valid_inc` | `inc, np` |
| 32 | 152-155 | `setup_or_compute` | `accepted` | `inside_gacos, valid_inc` |
| 33 | 158-158 | `setup_or_compute` | `ll_frac` | `float, valid_ll` |
| 34 | 159-159 | `setup_or_compute` | `gacos_frac` | `float, inside_gacos` |
| 35 | 160-160 | `setup_or_compute` | `inc_frac` | `float, valid_inc` |
| 36 | 161-161 | `setup_or_compute` | `accepted_frac` | `accepted, float` |
| 37 | 164-167 | `setup_or_compute` | `q` | `inc, np, valid_inc` |
| 38 | 170-170 | `logging` | `` | `print` |
| 39 | 171-171 | `logging` | `` | `print` |
| 40 | 172-172 | `logging` | `` | `print` |
| 41 | 174-174 | `logging` | `` | `n, print` |
| 42 | 175-175 | `logging` | `` | `ll_frac, print` |
| 43 | 176-176 | `logging` | `` | `gacos_frac, print` |
| 44 | 177-177 | `logging` | `` | `print` |
| 45 | 179-183 | `logging` | `` | `lon, print, valid_ll` |
| 46 | 185-189 | `logging` | `` | `lat, print, valid_ll` |
| 47 | 191-191 | `logging` | `` | `print` |
| 48 | 192-195 | `logging` | `` | `gamma_center_inc, print` |
| 49 | 197-200 | `logging` | `` | `print, stamps_center_inc` |
| 50 | 202-205 | `logging` | `` | `center_difference, print` |
| 51 | 207-210 | `logging` | `` | `print` |
| 52 | 212-212 | `logging` | `` | `print` |
| 53 | 214-216 | `logging` | `` | `print` |
| 54 | 218-225 | `logging` | `x` | `print, q, x` |
| 55 | 227-230 | `logging` | `` | `inc_frac, print` |
| 56 | 232-235 | `logging` | `` | `accepted_frac, print` |
| 57 | 238-239 | `validation` | `` | `RuntimeError, ll_frac` |
| 58 | 241-242 | `validation` | `` | `RuntimeError, gacos_frac` |
| 59 | 244-245 | `validation` | `` | `RuntimeError, inc_frac` |
| 60 | 247-248 | `validation` | `` | `RuntimeError, accepted_frac` |
| 61 | 255-258 | `output` | `` | `DEST, np, strict_ids` |
| 62 | 260-263 | `output` | `` | `DEST, np, rows` |
| 63 | 265-268 | `output` | `` | `DEST, cols, np` |
| 64 | 270-273 | `output` | `` | `DEST, lon, np` |
| 65 | 275-278 | `output` | `` | `DEST, lat, np` |
| 66 | 280-283 | `output` | `` | `DEST, inc, np` |
| 67 | 285-288 | `output` | `` | `DEST, accepted, np` |
| 68 | 291-346 | `setup_or_compute` | `manifest` | `accepted_frac, center_difference, float, gacos_frac, gamma_center_inc, inc_frac, int, ll_frac, n, q, stamps_center_inc` |
| 69 | 349-352 | `setup_or_compute` | `manifest_path` | `DEST` |
| 70 | 354-360 | `output` | `` | `json, manifest, manifest_path` |
| 71 | 363-363 | `logging` | `` | `print` |
| 72 | 364-367 | `logging` | `` | `manifest_path, print` |
| 73 | 369-369 | `logging` | `` | `print` |
| 74 | 370-370 | `logging` | `` | `print` |
| 75 | 371-374 | `logging` | `` | `print` |
| 76 | 375-375 | `logging` | `` | `print` |

Exact block source: `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/geometry_core_extraction/point_geolocation_top_level_blocks.txt`

Verbatim function extraction: `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/geometry_core_extraction/candidate/point_geolocation_functions.py`

## incidence

### Functions and implicit dependencies

| Function | Lines | Top-level globals used | Other free names |
|---|---:|---|---|
| `read_par` | 38-56 | `` | `` |
| `scalar` | 59-65 | `NUM_RE` | `` |
| `vec3` | 68-81 | `NUM_RE` | `` |
| `orbit_position` | 256-360 | `nsv, pos, sv_dt, sv_t0, vel` | `` |
| `incidence_fast` | 408-572 | `` | `` |

### Top-level block categories

| Category | Blocks |
|---|---:|
| `compute_or_control` | 1 |
| `input` | 4 |
| `logging` | 15 |
| `output` | 2 |
| `setup_or_compute` | 31 |
| `validation` | 2 |

### Top-level blocks

| ID | Lines | Category | Writes | Important reads |
|---:|---:|---|---|---|
| 1 | 10-12 | `setup_or_compute` | `ROOT` | `Path` |
| 2 | 14-16 | `setup_or_compute` | `PAR` | `Path` |
| 3 | 18-21 | `setup_or_compute` | `OUT` | `ROOT` |
| 4 | 23-26 | `setup_or_compute` | `MANIFEST` | `ROOT` |
| 5 | 33-35 | `setup_or_compute` | `NUM_RE` | `re` |
| 6 | 88-94 | `input` | `lon` | `ROOT, np` |
| 7 | 96-102 | `input` | `lat` | `ROOT, np` |
| 8 | 104-110 | `input` | `row` | `ROOT, np` |
| 9 | 112-117 | `setup_or_compute` | `hgt` | `ROOT, np` |
| 10 | 120-120 | `setup_or_compute` | `N` | `lon` |
| 11 | 122-133 | `validation` | `` | `N, RuntimeError, hgt, lat, row` |
| 12 | 140-142 | `input` | `p` | `PAR, read_par` |
| 13 | 144-151 | `setup_or_compute` | `nlines` | `int, p, round, scalar` |
| 14 | 153-156 | `setup_or_compute` | `line_dt` | `p, scalar` |
| 15 | 159-201 | `compute_or_control` | `center_time, row_time, start_time` | `center_time, line_dt, nlines, np, p, scalar, start_time` |
| 16 | 208-215 | `setup_or_compute` | `nsv` | `int, p, round, scalar` |
| 17 | 217-220 | `setup_or_compute` | `sv_t0` | `p, scalar` |
| 18 | 222-225 | `setup_or_compute` | `sv_dt` | `p, scalar` |
| 19 | 228-236 | `setup_or_compute` | `i, pos` | `i, np, nsv, p, range, vec3` |
| 20 | 239-247 | `setup_or_compute` | `i, vel` | `i, np, nsv, p, range, vec3` |
| 21 | 363-365 | `setup_or_compute` | `sat_by_row` | `orbit_position, row_time` |
| 22 | 372-372 | `setup_or_compute` | `A` | `` |
| 23 | 374-378 | `setup_or_compute` | `F` | `` |
| 24 | 380-388 | `setup_or_compute` | `E2` | `F` |
| 25 | 579-587 | `setup_or_compute` | `_` | `A, E2, hgt, incidence_fast, lat, lon, row, sat_by_row` |
| 26 | 594-594 | `setup_or_compute` | `t0` | `time` |
| 27 | 596-604 | `setup_or_compute` | `inc` | `A, E2, hgt, incidence_fast, lat, lon, row, sat_by_row` |
| 28 | 606-610 | `setup_or_compute` | `elapsed` | `t0, time` |
| 29 | 613-619 | `setup_or_compute` | `valid` | `inc, np` |
| 30 | 622-628 | `setup_or_compute` | `deg` | `inc, np, valid` |
| 31 | 631-640 | `setup_or_compute` | `q` | `deg, np` |
| 32 | 643-643 | `logging` | `` | `print` |
| 33 | 644-644 | `logging` | `` | `print` |
| 34 | 645-645 | `logging` | `` | `print` |
| 35 | 647-650 | `logging` | `` | `N, print` |
| 36 | 652-655 | `logging` | `` | `elapsed, print` |
| 37 | 657-660 | `logging` | `` | `N, elapsed, print` |
| 38 | 662-665 | `logging` | `` | `print, valid` |
| 39 | 667-674 | `logging` | `x` | `print, q, x` |
| 40 | 681-685 | `setup_or_compute` | `gamma_file` | `ROOT` |
| 41 | 688-690 | `setup_or_compute` | `truth_status` | `` |
| 42 | 692-692 | `setup_or_compute` | `rms` | `` |
| 43 | 693-693 | `setup_or_compute` | `p99` | `` |
| 44 | 694-694 | `setup_or_compute` | `max_abs` | `` |
| 45 | 697-801 | `validation` | `ad, diff, gamma, m, max_abs, p99, rms, truth_status` | `N, RuntimeError, ad, diff, float, gamma, gamma_file, inc, m, max_abs, np, p99, print, rms, valid` |
| 46 | 808-811 | `output` | `` | `OUT, inc, np` |
| 47 | 814-882 | `setup_or_compute` | `manifest` | `N, elapsed, float, int, max_abs, p99, q, rms, truth_status, valid` |
| 48 | 885-892 | `output` | `` | `MANIFEST, json, manifest` |
| 49 | 895-895 | `logging` | `` | `print` |
| 50 | 896-899 | `logging` | `` | `OUT, print` |
| 51 | 901-904 | `logging` | `` | `MANIFEST, print` |
| 52 | 906-906 | `logging` | `` | `print` |
| 53 | 907-907 | `logging` | `` | `print` |
| 54 | 908-911 | `logging` | `` | `print` |
| 55 | 912-912 | `logging` | `` | `print` |

Exact block source: `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/geometry_core_extraction/incidence_top_level_blocks.txt`

Verbatim function extraction: `/home/ubuntu/software/pyPSDS-GAMMA-v1.0/docs/release_v1_1/geometry_core_extraction/candidate/incidence_functions.py`

