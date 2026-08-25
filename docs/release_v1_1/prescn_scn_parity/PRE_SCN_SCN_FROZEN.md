# pre-SCN + SCN v1.1 Frozen Contract

Status: FROZEN

Production domain:

- 881315 points
- 38 acquisitions
- geometric master: 20151212
- temporal reference: 20141006

pre-SCN:

- formula: phi_preSCN = phi_GACOS - ph_scla - C_ps_uw
- output shape: 881315 x 38
- output dtype: float32

SCN:

- StaMPS Stage-8 compatible spatial/temporal filtering
- final output: ph_scn_slave_rad.npy
- output shape: 881315 x 38
- output dtype: float64
- first-PS SCN reference preserved
- geometric-master SCN forced to zero

Validation:

- authoritative vs migrated pre-SCN: byte exact
- authoritative vs migrated ph_hpt: byte exact
- authoritative vs migrated ph_scn_slave: byte exact

Upstream:

- Geometry: FROZEN
- GACOS: FROZEN
- SCLA: FROZEN

Pipeline: unchanged / 33 stages
