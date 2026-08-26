from importlib import resources
import json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]

def test_production_defaults():
    cfg=yaml.safe_load((ROOT/'config/pypsds.yaml').read_text())
    shp=cfg['selection']['shp']; pl=cfg['phase_linking']; ds=cfg['selection']['ds']
    assert shp['method']=='rayleigh_glrt' and float(shp['alpha'])==0.005
    assert int(shp['half_row'])==5 and int(shp['half_col'])==11 and int(shp['min_count'])==48
    assert pl['method']=='robust_emi' and pl['temporal']['strategy']=='sequential'
    assert int(pl['temporal']['ministack_size'])==19 and int(pl['temporal']['max_num_compressed'])==5
    assert float(ds['temporal_coherence_min'])==0.8 and ds['accept_evd'] is True
    assert 'adaptive_filter' not in cfg

def test_packaged_policy_is_clean():
    q=json.loads(resources.files('pypsds.resources').joinpath('ds_production_policy.json').read_text())
    text=json.dumps(q)
    assert q['schema_version']==1 and 'default_profile' in q
    for token in ('P9','P10','P11','P15','migration','prototype'):
        assert token not in text
