from pathlib import Path
from pypsds.pipeline import STAGES

ROOT=Path(__file__).resolve().parents[1]

def test_stable_runtime_names():
    assert (ROOT/'pypsds/runtime_backend').is_dir()
    assert (ROOT/'pypsds/stages/_stage_common.py').is_file()

def test_pipeline_tail():
    assert len(STAGES) == 39
    assert [x.name for x in STAGES[-5:]]==[
        'atmosphere_correction',
        'scla',
        'scn',
        'final_los',
        'point_products',
    ]

def test_no_current_project_hardcoding_in_active_source():
    bad=[]
    tokens=(
        '/home/ubuntu/Downloads/psds',
        '/home/ubuntu/software/pyPSDS-GAMMA-v1.0',
        '20151212_4_1',
        'expected 38 acquisitions',
        'expected 108 IFGs',
        'unexpected Sentinel-1 wavelength',
        'P15-',
        'P11D',
        'P10B',
        'P9F',
    )
    for base in (ROOT/'pypsds',ROOT/'tools'):
        for p in base.rglob('*'):
            if (
                p.is_file()
                and p.suffix.lower() in {
                    '.py','.yaml','.yml','.json','.toml','.md','.txt'
                }
                and '__pycache__' not in p.parts
            ):
                text=p.read_text(errors='ignore')
                for tok in tokens:
                    if tok in text:
                        bad.append((str(p.relative_to(ROOT)),tok))
    assert not bad, bad[:50]

def test_temporal_reference_contract():
    import yaml
    p=ROOT/'pypsds/resources/default_config.yaml'
    cfg=yaml.safe_load(p.read_text())
    assert int(cfg['phase_linking']['temporal_reference_index'])==0
