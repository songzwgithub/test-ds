# Geometry producer lineage audit v2

Malformed development snapshots are recorded but do not stop analysis of later validated candidates.

## Version summary

| Version | Syntax | Lines | Role | Process execution | data2pt | Loads | Saves |
|---|---|---:|---|---|---|---:|---:|
| `v2` | FAIL | 2137 | `unusable_development_snapshot` | N/A | — | 0 | 0 |
| `v3` | FAIL | 1110 | `unusable_development_snapshot` | N/A | text-hit | 0 | 0 |
| `v4` | PASS | 462 | `geometry_producer` | True | True | 3 | 7 |
| `v4b` | PASS | 375 | `geometry_inline_producer_or_finalizer` | False | True | 3 | 7 |

## Syntax failures

- `v2`: line 851, `expected ':'`; source preserved unchanged.
- `v3`: line 357, `expected ':'`; source preserved unchanged.

## Function lineage

| Function | v2 | v3 | v4 | v4b |
|---|---|---|---|---|
| `incidence_from_range` | `—` | `—` | `761d728612` | `—` |
| `read_par` | `—` | `—` | `72a4c721f7` | `72a4c721f7` |
| `stamps_incidence` | `—` | `—` | `—` | `123ad58dc8` |
| `val` | `—` | `—` | `1775612a59` | `1775612a59` |

## v3 data2pt context

### line 118

```python
0103:         return None
0104: 
0105:     try:
0106:         return float(
0107:             d[key].split()[0]
0108:         )
0109: 
0110:     except Exception:
0111:         return None
0112: 
0113: 
0114: print("=" * 96)
0115: print("P15-3A v3 StaMPS-style point geometry")
0116: print()
0117: print("USES EXISTING RADAR-COORDINATE lon/lat GRIDS")
0118: print("USES GAMMA data2pt")
0119: print("NO PHASE MODIFICATION")
0120: print("NO GACOS CORRECTION")
0121: print("=" * 96)
0122: 
0123: 
0124: # =============================================================================
0125: # 1. Input contracts
0126: # =============================================================================
0127: 
0128: for p in (
0129:     RSLC_PAR,
0130:     GEO_PAR,
0131:     LON_RASTER,
0132:     LAT_RASTER,
0133:     STRICT_IDS,
0134:     ROWS,
0135:     COLS,
0136: ):
0137: 
0138:     check(
0139:         f"exists: {p.name}",
0140:         p.is_file(),
```

### line 331

```python
0316: 
0317: check(
0318:     "plist byte size",
0319:     plist.stat().st_size
0320:     ==
0321:     N
0322:     *
0323:     2
0324:     *
0325:     4,
0326:     plist.stat().st_size,
0327: )
0328: 
0329: 
0330: # =============================================================================
0331: # 4. Resolve data2pt
0332: # =============================================================================
0333: 
0334: data2pt = shutil.which(
0335:     "data2pt"
0336: )
0337: 
0338: 
0339: if data2pt is None:
0340: 
0341:     candidates = list(
0342:         Path(
0343:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0344:         ).rglob(
0345:             "data2pt"
0346:         )
0347:     )
0348: 
0349:     candidates = [
0350:         p
0351:         for p in candidates
0352:         if p.is_file()
0353:     ]
```

### line 334

```python
0319:     plist.stat().st_size
0320:     ==
0321:     N
0322:     *
0323:     2
0324:     *
0325:     4,
0326:     plist.stat().st_size,
0327: )
0328: 
0329: 
0330: # =============================================================================
0331: # 4. Resolve data2pt
0332: # =============================================================================
0333: 
0334: data2pt = shutil.which(
0335:     "data2pt"
0336: )
0337: 
0338: 
0339: if data2pt is None:
0340: 
0341:     candidates = list(
0342:         Path(
0343:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0344:         ).rglob(
0345:             "data2pt"
0346:         )
0347:     )
0348: 
0349:     candidates = [
0350:         p
0351:         for p in candidates
0352:         if p.is_file()
0353:     ]
0354: 
0355:     if len(
0356:         candidates
```

### line 335

```python
0320:     ==
0321:     N
0322:     *
0323:     2
0324:     *
0325:     4,
0326:     plist.stat().st_size,
0327: )
0328: 
0329: 
0330: # =============================================================================
0331: # 4. Resolve data2pt
0332: # =============================================================================
0333: 
0334: data2pt = shutil.which(
0335:     "data2pt"
0336: )
0337: 
0338: 
0339: if data2pt is None:
0340: 
0341:     candidates = list(
0342:         Path(
0343:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0344:         ).rglob(
0345:             "data2pt"
0346:         )
0347:     )
0348: 
0349:     candidates = [
0350:         p
0351:         for p in candidates
0352:         if p.is_file()
0353:     ]
0354: 
0355:     if len(
0356:         candidates
0357:     )
```

### line 339

```python
0324:     *
0325:     4,
0326:     plist.stat().st_size,
0327: )
0328: 
0329: 
0330: # =============================================================================
0331: # 4. Resolve data2pt
0332: # =============================================================================
0333: 
0334: data2pt = shutil.which(
0335:     "data2pt"
0336: )
0337: 
0338: 
0339: if data2pt is None:
0340: 
0341:     candidates = list(
0342:         Path(
0343:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0344:         ).rglob(
0345:             "data2pt"
0346:         )
0347:     )
0348: 
0349:     candidates = [
0350:         p
0351:         for p in candidates
0352:         if p.is_file()
0353:     ]
0354: 
0355:     if len(
0356:         candidates
0357:     )
0358:     ==
0359:     1:
0360: 
0361:         data2pt = str(
```

### line 345

```python
0330: # =============================================================================
0331: # 4. Resolve data2pt
0332: # =============================================================================
0333: 
0334: data2pt = shutil.which(
0335:     "data2pt"
0336: )
0337: 
0338: 
0339: if data2pt is None:
0340: 
0341:     candidates = list(
0342:         Path(
0343:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0344:         ).rglob(
0345:             "data2pt"
0346:         )
0347:     )
0348: 
0349:     candidates = [
0350:         p
0351:         for p in candidates
0352:         if p.is_file()
0353:     ]
0354: 
0355:     if len(
0356:         candidates
0357:     )
0358:     ==
0359:     1:
0360: 
0361:         data2pt = str(
0362:             candidates[0]
0363:         )
0364: 
0365: 
0366: check(
0367:     "data2pt resolved",
```

### line 361

```python
0346:         )
0347:     )
0348: 
0349:     candidates = [
0350:         p
0351:         for p in candidates
0352:         if p.is_file()
0353:     ]
0354: 
0355:     if len(
0356:         candidates
0357:     )
0358:     ==
0359:     1:
0360: 
0361:         data2pt = str(
0362:             candidates[0]
0363:         )
0364: 
0365: 
0366: check(
0367:     "data2pt resolved",
0368:     data2pt is not None,
0369:     data2pt,
0370: )
0371: 
0372: 
0373: if errors:
0374:     raise SystemExit(1)
0375: 
0376: 
0377: # =============================================================================
0378: # 5. Sample longitude + latitude exactly like pyPSDS height sampling
0379: # =============================================================================
0380: 
0381: lon_pt = (
0382:     DEST
0383:     / "longitude_deg.gamma_pt"
```

### line 367

```python
0352:         if p.is_file()
0353:     ]
0354: 
0355:     if len(
0356:         candidates
0357:     )
0358:     ==
0359:     1:
0360: 
0361:         data2pt = str(
0362:             candidates[0]
0363:         )
0364: 
0365: 
0366: check(
0367:     "data2pt resolved",
0368:     data2pt is not None,
0369:     data2pt,
0370: )
0371: 
0372: 
0373: if errors:
0374:     raise SystemExit(1)
0375: 
0376: 
0377: # =============================================================================
0378: # 5. Sample longitude + latitude exactly like pyPSDS height sampling
0379: # =============================================================================
0380: 
0381: lon_pt = (
0382:     DEST
0383:     / "longitude_deg.gamma_pt"
0384: )
0385: 
0386: lat_pt = (
0387:     DEST
0388:     / "latitude_deg.gamma_pt"
0389: )
```

### line 368

```python
0353:     ]
0354: 
0355:     if len(
0356:         candidates
0357:     )
0358:     ==
0359:     1:
0360: 
0361:         data2pt = str(
0362:             candidates[0]
0363:         )
0364: 
0365: 
0366: check(
0367:     "data2pt resolved",
0368:     data2pt is not None,
0369:     data2pt,
0370: )
0371: 
0372: 
0373: if errors:
0374:     raise SystemExit(1)
0375: 
0376: 
0377: # =============================================================================
0378: # 5. Sample longitude + latitude exactly like pyPSDS height sampling
0379: # =============================================================================
0380: 
0381: lon_pt = (
0382:     DEST
0383:     / "longitude_deg.gamma_pt"
0384: )
0385: 
0386: lat_pt = (
0387:     DEST
0388:     / "latitude_deg.gamma_pt"
0389: )
0390: 
```

### line 369

```python
0354: 
0355:     if len(
0356:         candidates
0357:     )
0358:     ==
0359:     1:
0360: 
0361:         data2pt = str(
0362:             candidates[0]
0363:         )
0364: 
0365: 
0366: check(
0367:     "data2pt resolved",
0368:     data2pt is not None,
0369:     data2pt,
0370: )
0371: 
0372: 
0373: if errors:
0374:     raise SystemExit(1)
0375: 
0376: 
0377: # =============================================================================
0378: # 5. Sample longitude + latitude exactly like pyPSDS height sampling
0379: # =============================================================================
0380: 
0381: lon_pt = (
0382:     DEST
0383:     / "longitude_deg.gamma_pt"
0384: )
0385: 
0386: lat_pt = (
0387:     DEST
0388:     / "latitude_deg.gamma_pt"
0389: )
0390: 
0391: 
```

### line 396

```python
0381: lon_pt = (
0382:     DEST
0383:     / "longitude_deg.gamma_pt"
0384: )
0385: 
0386: lat_pt = (
0387:     DEST
0388:     / "latitude_deg.gamma_pt"
0389: )
0390: 
0391: 
0392: commands = [
0393:     (
0394:         "longitude",
0395:         [
0396:             data2pt,
0397:             str(
0398:                 LON_RASTER
0399:             ),
0400:             str(
0401:                 GEO_PAR
0402:             ),
0403:             str(
0404:                 plist
0405:             ),
0406:             str(
0407:                 RSLC_PAR
0408:             ),
0409:             str(
0410:                 lon_pt
0411:             ),
0412:             "1",
0413:             "2",
0414:         ],
0415:     ),
0416:     (
0417:         "latitude",
0418:         [
```

### line 419

```python
0404:                 plist
0405:             ),
0406:             str(
0407:                 RSLC_PAR
0408:             ),
0409:             str(
0410:                 lon_pt
0411:             ),
0412:             "1",
0413:             "2",
0414:         ],
0415:     ),
0416:     (
0417:         "latitude",
0418:         [
0419:             data2pt,
0420:             str(
0421:                 LAT_RASTER
0422:             ),
0423:             str(
0424:                 GEO_PAR
0425:             ),
0426:             str(
0427:                 plist
0428:             ),
0429:             str(
0430:                 RSLC_PAR
0431:             ),
0432:             str(
0433:                 lat_pt
0434:             ),
0435:             "1",
0436:             "2",
0437:         ],
0438:     ),
0439: ]
0440: 
0441: 
```

### line 481

```python
0466: 
0467:     print(
0468:         "\n".join(
0469:             (
0470:                 proc.stdout
0471:                 or
0472:                 ""
0473:             ).splitlines()[
0474:                 -10:
0475:             ]
0476:         )
0477:     )
0478: 
0479: 
0480:     check(
0481:         f"data2pt {label}",
0482:         proc.returncode == 0,
0483:         proc.returncode,
0484:     )
0485: 
0486: 
0487: if errors:
0488:     raise SystemExit(1)
0489: 
0490: 
0491: # GAMMA FLOAT output
0492: lon = np.fromfile(
0493:     lon_pt,
0494:     dtype=">f4",
0495: ).astype(
0496:     np.float64
0497: )
0498: 
0499: 
0500: lat = np.fromfile(
0501:     lat_pt,
0502:     dtype=">f4",
0503: ).astype(
```

### line 966

```python
0951:         ),
0952: 
0953:     "source_geometry_par":
0954:         str(
0955:             GEO_PAR
0956:         ),
0957: 
0958:     "reference_rslc_par":
0959:         str(
0960:             RSLC_PAR
0961:         ),
0962: 
0963:     "sampling":
0964:         {
0965:             "command":
0966:                 "GAMMA data2pt",
0967: 
0968:             "plist_order":
0969:                 "range_col, azimuth_row",
0970: 
0971:             "plist_dtype":
0972:                 "big-endian int32",
0973: 
0974:             "output_dtype":
0975:                 "big-endian float32",
0976:         },
0977: 
0978:     "incidence":
0979:         {
0980:             "method":
0981:                 "StaMPS_spherical_radar_geometry",
0982: 
0983:             "formula":
0984:                 (
0985:                     "acos((se^2-re^2-rg^2)"
0986:                     "/(2*re*rg))"
0987:                 ),
0988: 
```

## v4 data2pt context

### line 109

```python
0094: 
0095: plist = DEST / "strict_points.plist"
0096: 
0097: np.column_stack(
0098:     (cols, rows)
0099: ).astype(
0100:     ">i4"
0101: ).tofile(
0102:     plist
0103: )
0104: 
0105: assert plist.stat().st_size == n * 8
0106: 
0107: 
0108: # ----------------------------------------------------------------------
0109: # 4. Resolve data2pt
0110: # ----------------------------------------------------------------------
0111: 
0112: data2pt = shutil.which("data2pt")
0113: 
0114: if data2pt is None:
0115:     cands = [
0116:         p
0117:         for p in Path(
0118:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0119:         ).rglob("data2pt")
0120:         if p.is_file()
0121:     ]
0122: 
0123:     if len(cands) != 1:
0124:         raise RuntimeError(
0125:             f"Cannot resolve data2pt uniquely: {cands[:10]}"
0126:         )
0127: 
0128:     data2pt = str(cands[0])
0129: 
0130: print("data2pt                  :", data2pt)
0131: 
```

### line 112

```python
0097: np.column_stack(
0098:     (cols, rows)
0099: ).astype(
0100:     ">i4"
0101: ).tofile(
0102:     plist
0103: )
0104: 
0105: assert plist.stat().st_size == n * 8
0106: 
0107: 
0108: # ----------------------------------------------------------------------
0109: # 4. Resolve data2pt
0110: # ----------------------------------------------------------------------
0111: 
0112: data2pt = shutil.which("data2pt")
0113: 
0114: if data2pt is None:
0115:     cands = [
0116:         p
0117:         for p in Path(
0118:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0119:         ).rglob("data2pt")
0120:         if p.is_file()
0121:     ]
0122: 
0123:     if len(cands) != 1:
0124:         raise RuntimeError(
0125:             f"Cannot resolve data2pt uniquely: {cands[:10]}"
0126:         )
0127: 
0128:     data2pt = str(cands[0])
0129: 
0130: print("data2pt                  :", data2pt)
0131: 
0132: 
0133: # ----------------------------------------------------------------------
0134: # 5. Sample lon / lat
```

### line 114

```python
0099: ).astype(
0100:     ">i4"
0101: ).tofile(
0102:     plist
0103: )
0104: 
0105: assert plist.stat().st_size == n * 8
0106: 
0107: 
0108: # ----------------------------------------------------------------------
0109: # 4. Resolve data2pt
0110: # ----------------------------------------------------------------------
0111: 
0112: data2pt = shutil.which("data2pt")
0113: 
0114: if data2pt is None:
0115:     cands = [
0116:         p
0117:         for p in Path(
0118:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0119:         ).rglob("data2pt")
0120:         if p.is_file()
0121:     ]
0122: 
0123:     if len(cands) != 1:
0124:         raise RuntimeError(
0125:             f"Cannot resolve data2pt uniquely: {cands[:10]}"
0126:         )
0127: 
0128:     data2pt = str(cands[0])
0129: 
0130: print("data2pt                  :", data2pt)
0131: 
0132: 
0133: # ----------------------------------------------------------------------
0134: # 5. Sample lon / lat
0135: # ----------------------------------------------------------------------
0136: 
```

### line 119

```python
0104: 
0105: assert plist.stat().st_size == n * 8
0106: 
0107: 
0108: # ----------------------------------------------------------------------
0109: # 4. Resolve data2pt
0110: # ----------------------------------------------------------------------
0111: 
0112: data2pt = shutil.which("data2pt")
0113: 
0114: if data2pt is None:
0115:     cands = [
0116:         p
0117:         for p in Path(
0118:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0119:         ).rglob("data2pt")
0120:         if p.is_file()
0121:     ]
0122: 
0123:     if len(cands) != 1:
0124:         raise RuntimeError(
0125:             f"Cannot resolve data2pt uniquely: {cands[:10]}"
0126:         )
0127: 
0128:     data2pt = str(cands[0])
0129: 
0130: print("data2pt                  :", data2pt)
0131: 
0132: 
0133: # ----------------------------------------------------------------------
0134: # 5. Sample lon / lat
0135: # ----------------------------------------------------------------------
0136: 
0137: lon_bin = DEST / "longitude_deg.gamma_pt"
0138: lat_bin = DEST / "latitude_deg.gamma_pt"
0139: 
0140: jobs = [
0141:     (LON, lon_bin, "longitude"),
```

### line 125

```python
0110: # ----------------------------------------------------------------------
0111: 
0112: data2pt = shutil.which("data2pt")
0113: 
0114: if data2pt is None:
0115:     cands = [
0116:         p
0117:         for p in Path(
0118:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0119:         ).rglob("data2pt")
0120:         if p.is_file()
0121:     ]
0122: 
0123:     if len(cands) != 1:
0124:         raise RuntimeError(
0125:             f"Cannot resolve data2pt uniquely: {cands[:10]}"
0126:         )
0127: 
0128:     data2pt = str(cands[0])
0129: 
0130: print("data2pt                  :", data2pt)
0131: 
0132: 
0133: # ----------------------------------------------------------------------
0134: # 5. Sample lon / lat
0135: # ----------------------------------------------------------------------
0136: 
0137: lon_bin = DEST / "longitude_deg.gamma_pt"
0138: lat_bin = DEST / "latitude_deg.gamma_pt"
0139: 
0140: jobs = [
0141:     (LON, lon_bin, "longitude"),
0142:     (LAT, lat_bin, "latitude"),
0143: ]
0144: 
0145: for src, dst, name in jobs:
0146: 
0147:     cmd = [
```

### line 128

```python
0113: 
0114: if data2pt is None:
0115:     cands = [
0116:         p
0117:         for p in Path(
0118:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0119:         ).rglob("data2pt")
0120:         if p.is_file()
0121:     ]
0122: 
0123:     if len(cands) != 1:
0124:         raise RuntimeError(
0125:             f"Cannot resolve data2pt uniquely: {cands[:10]}"
0126:         )
0127: 
0128:     data2pt = str(cands[0])
0129: 
0130: print("data2pt                  :", data2pt)
0131: 
0132: 
0133: # ----------------------------------------------------------------------
0134: # 5. Sample lon / lat
0135: # ----------------------------------------------------------------------
0136: 
0137: lon_bin = DEST / "longitude_deg.gamma_pt"
0138: lat_bin = DEST / "latitude_deg.gamma_pt"
0139: 
0140: jobs = [
0141:     (LON, lon_bin, "longitude"),
0142:     (LAT, lat_bin, "latitude"),
0143: ]
0144: 
0145: for src, dst, name in jobs:
0146: 
0147:     cmd = [
0148:         data2pt,
0149:         str(src),
0150:         str(GEO_PAR),
```

### line 130

```python
0115:     cands = [
0116:         p
0117:         for p in Path(
0118:             "/home/ubuntu/software/GAMMA_SOFTWARE"
0119:         ).rglob("data2pt")
0120:         if p.is_file()
0121:     ]
0122: 
0123:     if len(cands) != 1:
0124:         raise RuntimeError(
0125:             f"Cannot resolve data2pt uniquely: {cands[:10]}"
0126:         )
0127: 
0128:     data2pt = str(cands[0])
0129: 
0130: print("data2pt                  :", data2pt)
0131: 
0132: 
0133: # ----------------------------------------------------------------------
0134: # 5. Sample lon / lat
0135: # ----------------------------------------------------------------------
0136: 
0137: lon_bin = DEST / "longitude_deg.gamma_pt"
0138: lat_bin = DEST / "latitude_deg.gamma_pt"
0139: 
0140: jobs = [
0141:     (LON, lon_bin, "longitude"),
0142:     (LAT, lat_bin, "latitude"),
0143: ]
0144: 
0145: for src, dst, name in jobs:
0146: 
0147:     cmd = [
0148:         data2pt,
0149:         str(src),
0150:         str(GEO_PAR),
0151:         str(plist),
0152:         str(RSLC_PAR),
```

### line 148

```python
0133: # ----------------------------------------------------------------------
0134: # 5. Sample lon / lat
0135: # ----------------------------------------------------------------------
0136: 
0137: lon_bin = DEST / "longitude_deg.gamma_pt"
0138: lat_bin = DEST / "latitude_deg.gamma_pt"
0139: 
0140: jobs = [
0141:     (LON, lon_bin, "longitude"),
0142:     (LAT, lat_bin, "latitude"),
0143: ]
0144: 
0145: for src, dst, name in jobs:
0146: 
0147:     cmd = [
0148:         data2pt,
0149:         str(src),
0150:         str(GEO_PAR),
0151:         str(plist),
0152:         str(RSLC_PAR),
0153:         str(dst),
0154:         "1",
0155:         "2",
0156:     ]
0157: 
0158:     print()
0159:     print("$", " ".join(cmd))
0160: 
0161:     p = subprocess.run(
0162:         cmd,
0163:         text=True,
0164:         stdout=subprocess.PIPE,
0165:         stderr=subprocess.STDOUT,
0166:     )
0167: 
0168:     print(
0169:         "\n".join(
0170:             (p.stdout or "").splitlines()[-12:]
```

### line 176

```python
0161:     p = subprocess.run(
0162:         cmd,
0163:         text=True,
0164:         stdout=subprocess.PIPE,
0165:         stderr=subprocess.STDOUT,
0166:     )
0167: 
0168:     print(
0169:         "\n".join(
0170:             (p.stdout or "").splitlines()[-12:]
0171:         )
0172:     )
0173: 
0174:     if p.returncode != 0:
0175:         raise RuntimeError(
0176:             f"data2pt {name} failed: {p.returncode}"
0177:         )
0178: 
0179: 
0180: lon = np.fromfile(
0181:     lon_bin,
0182:     dtype=">f4",
0183: ).astype(
0184:     np.float64
0185: )
0186: 
0187: lat = np.fromfile(
0188:     lat_bin,
0189:     dtype=">f4",
0190: ).astype(
0191:     np.float64
0192: )
0193: 
0194: assert lon.size == n, (lon.size, n)
0195: assert lat.size == n, (lat.size, n)
0196: 
0197: 
0198: # ----------------------------------------------------------------------
```

### line 429

```python
0414: 
0415: np.save(
0416:     DEST / "incidence_angle_deg.npy",
0417:     inc.astype(np.float32),
0418: )
0419: 
0420: np.save(
0421:     DEST / "valid_gacos_geometry_mask.npy",
0422:     accepted,
0423: )
0424: 
0425: manifest = {
0426:     "status": "PASS_RADAR_POINT_GEOLOCATION",
0427:     "method": (
0428:         "StaMPS_GAMMA_style_"
0429:         "data2pt_lonlat_plus_analytic_incidence"
0430:     ),
0431:     "points": int(n),
0432:     "valid_lonlat_fraction": ll_frac,
0433:     "gacos_coverage_fraction": gacos_frac,
0434:     "incidence_valid_fraction": inc_frac,
0435:     "accepted_fraction": accepted_fraction,
0436:     "center_incidence_par_deg": center_inc_par,
0437:     "center_incidence_calc_deg": center_inc_calc,
0438:     "incidence_p01_p05_p50_p95_p99_deg": [
0439:         float(x)
0440:         for x in q
0441:     ],
0442:     "next_step": "P15-4_GACOS_POINT_SAMPLING_SMOKE",
0443: }
0444: 
0445: manifest_path = (
0446:     DEST
0447:     / "gacos_geometry_manifest.json"
0448: )
0449: 
0450: manifest_path.write_text(
0451:     json.dumps(
```

## v4b data2pt context

### line 57

```python
0042: 
0043: all_cols = np.load(
0044:     PROC / "point_phase_stack" / "cols.npy",
0045:     mmap_mode="r",
0046: )
0047: 
0048: rows = np.asarray(all_rows[strict_ids], dtype=np.int32)
0049: cols = np.asarray(all_cols[strict_ids], dtype=np.int32)
0050: 
0051: n = strict_ids.size
0052: 
0053: assert n == 881315
0054: 
0055: 
0056: # ------------------------------------------------------------
0057: # Existing data2pt products
0058: # ------------------------------------------------------------
0059: 
0060: lon = np.fromfile(
0061:     lon_bin,
0062:     dtype=">f4",
0063: ).astype(np.float64)
0064: 
0065: lat = np.fromfile(
0066:     lat_bin,
0067:     dtype=">f4",
0068: ).astype(np.float64)
0069: 
0070: assert lon.size == n
0071: assert lat.size == n
0072: 
0073: 
0074: valid_ll = (
0075:     np.isfinite(lon)
0076:     & np.isfinite(lat)
0077:     & (lon > -180)
0078:     & (lon < 180)
0079:     & (lat > -90)
```

### line 299

```python
0284: 
0285: np.save(
0286:     DEST / "valid_gacos_geometry_mask.npy",
0287:     accepted,
0288: )
0289: 
0290: 
0291: manifest = {
0292:     "format":
0293:         "pyPSDS-GAMMA-P15-3A-StaMPS-geometry-v4b",
0294: 
0295:     "status":
0296:         "PASS_RADAR_POINT_GEOLOCATION",
0297: 
0298:     "longitude_latitude_method":
0299:         "GAMMA_data2pt_existing_rdc_lon_lat",
0300: 
0301:     "incidence_method":
0302:         "StaMPS_mt_ml_select_gamma_spherical_geometry",
0303: 
0304:     "incidence_formula":
0305:         (
0306:             "acos((se^2-re^2-rg^2)/(2*re*rg))"
0307:         ),
0308: 
0309:     "points":
0310:         int(n),
0311: 
0312:     "valid_lonlat_fraction":
0313:         ll_frac,
0314: 
0315:     "gacos_coverage_fraction":
0316:         gacos_frac,
0317: 
0318:     "incidence_valid_fraction":
0319:         inc_frac,
0320: 
0321:     "accepted_fraction":
```

## v3 → v4

Unified diff lines: **1466**

See `v3_to_v4.diff`.

## Selection rule

The malformed v2 snapshot is provenance evidence only. The v1.1 producer decision must be based on the latest syntactically valid implementation that actually creates the longitude/latitude intermediates for a fresh study area.
