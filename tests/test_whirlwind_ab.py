import json
import sys
from pathlib import Path

import numpy as np
import pytest
from pypsds.unwrap.whirlwind_ab import (
    load_gamma_crop, unwrap_whirlwind, compare_same_input,
    snaphu_config, _parity_max, main,
)


def write_data(tmp_path, endian='big', corr_layout='float32'):
    h,w=7,8
    y,x=np.indices((h,w))
    raw = np.exp(1j*(0.3*x + 0.2*y)).astype(np.complex64)
    coherence = np.full((h,w),.8,np.float32)
    coherence[0,0]=0
    co=tmp_path/'pair.cc'; ig=tmp_path/'pair.diff'
    raw.astype(('>' if endian=='big' else '<')+'c8').tofile(ig)
    if corr_layout=='alt-line':
        v=np.stack((np.ones_like(coherence),coherence),axis=1)
    else:v=coherence
    v.astype(('>' if endian=='big' else '<')+'f4').tofile(co)
    return ig,co,raw,coherence


@pytest.mark.parametrize('endian', ['big','little'])
@pytest.mark.parametrize('layout', ['float32','alt-line'])
def test_gamma_complex_coherence_and_crop(tmp_path,endian,layout):
    ig,co,raw,corr=write_data(tmp_path,endian,layout)
    out, c, mask=load_gamma_crop(ifg_path=ig,corr_path=co,full_height=7,full_width=8,
        row0=0,col0=0,height=7,width=8,endian=endian,corr_layout=layout)
    np.testing.assert_allclose(out[mask],raw[mask],atol=2e-7)
    np.testing.assert_allclose(c[mask],corr[mask],atol=1e-6)
    assert not mask[0,0]
    out,c,mask=load_gamma_crop(ifg_path=ig,corr_path=co,full_height=7,full_width=8,
        row0=1,col0=1,height=4,width=5,endian=endian,corr_layout=layout)
    np.testing.assert_allclose(out,raw[1:5,1:6],atol=2e-7)
    assert mask.all()


def test_reject_wrong_corr_layout_and_invalid_looks(tmp_path):
    ig,co,*_=write_data(tmp_path)
    with pytest.raises(ValueError,match='expected='):
        load_gamma_crop(ifg_path=ig,corr_path=co,full_height=7,full_width=8,
            row0=0,col0=0,height=7,width=8,endian='big',corr_layout='alt-line')
    with pytest.raises(ValueError,match='looks'):
        snaphu_config(.4,.055)


def test_ww_does_not_allow_synthetic_bridges():
    wrap = np.ones((4,4),dtype=np.complex64)
    mask = np.ones((4,4),bool)
    seen={}
    def fake(igram,corr,**kwargs):
        seen.update(kwargs)
        return np.zeros(igram.shape,np.float32), np.ones(igram.shape,np.uint8)
    ww,cc,stats=unwrap_whirlwind(wrap,np.full((4,4),.8,np.float32),mask,4,unwrap_func=fake)
    assert seen['bridge'] is False
    assert seen['connect_gaps'] is False
    assert seen['interpolate'] is False
    assert stats['wrap_parity_max_rad']<1e-5


def test_ww_rejects_2pi_mismatched_output():
    phase = np.exp(1j*np.ones((4,4))).astype(np.complex64)
    with pytest.raises(RuntimeError,match='parity'):
        unwrap_whirlwind(phase,np.full((4,4),.7,np.float32),
            np.ones((4,4),bool),3,
            unwrap_func=lambda *a, **kw:(np.zeros((4,4)),np.ones((4,4),np.int8)))


def test_same_input_gauge_alignment_and_real_error():
    a=np.zeros((2,4),np.float32)
    b=np.zeros_like(a)
    a[:,0:2]+=2*np.pi*4
    a[:,2:4]+=2*np.pi*-3
    comps=np.array([[1,1,2,2],[1,1,2,2]],dtype=np.int32)
    rpt=compare_same_input(a,b,comps,np.ones_like(comps),np.ones_like(comps,bool))
    assert rpt['fraction_cycle_mismatch']==0
    a[0,1]+=2*np.pi
    rpt=compare_same_input(a,b,comps,np.ones_like(comps),np.ones_like(comps,bool))
    assert rpt['fraction_cycle_mismatch'] == pytest.approx(1/8)
    assert rpt['joint_components']==2


def test_mask_excludes_bad_coherence_and_respects_npy(tmp_path):
    ig,co,raw,corr=write_data(tmp_path)
    m=np.ones((7,8),bool);m[3,4]=False
    mp=tmp_path/'mask.npy';np.save(mp,m)
    _,_,valid=load_gamma_crop(ifg_path=ig,corr_path=co,
        full_height=7,full_width=8,row0=0,col0=0,height=7,width=8,
        endian='big',corr_layout='float32',mask_path=mp)
    assert not valid[3,4]
    assert valid.sum()==54


def test_cli_input_audit_without_whirlwind_install(tmp_path):
    ig,co,*_=write_data(tmp_path)
    dest=tmp_path/'out'
    rc=main(['--ifg',str(ig),'--corr',str(co),'--full-height','7',
        '--full-width','8','--height','7','--width','8',
        '--nlooks','3.2','--wavelength-m','0.05546576',
        '--output-dir',str(dest),'--audit-only'])
    assert rc==0
    j=json.loads((dest/'input_audit.json').read_text())
    assert j['status']=='INPUT_AUDIT'
    assert j['valid_pixels']==55
    assert not (dest/'snaphu_scratch').exists()
    # Input inspection must not guess the effective looks or wavelength.
    other=tmp_path/'no_looks'
    assert main(['--ifg',str(ig),'--corr',str(co),'--full-height','7',
        '--full-width','8','--height','7','--width','8',
        '--output-dir',str(other),'--audit-only']) == 0
    audit=json.loads((other/'input_audit.json').read_text())
    assert audit['nlooks_effective_coherence'] is None


def test_cli_same_input_complete_both_solvers(tmp_path,monkeypatch):
    ig,co,*_=write_data(tmp_path)
    # Mock the real SNAPHU process while preserving real file I/O contract.
    from pypsds.unwrap import whirlwind_ab as mod
    monkeypatch.setattr(mod.shutil,'which',lambda n:'/fake/snaphu')
    class Proc: returncode=0
    def subprocess_mock(cmd, cwd, stdout, stderr, check):
        x=np.fromfile(cwd/'input.cpx',np.complex64).reshape(7,8)
        np.angle(x).astype('float32').tofile(cwd/'unwrapped.f32')
        np.ones((7,8),dtype=np.uint8).tofile(cwd/'conncomp.u8')
        return Proc()
    monkeypatch.setattr(mod.subprocess,'run',subprocess_mock)
    def fake_ww(ifg,corr, **kwargs):
        return np.angle(ifg).astype(np.float32),np.ones(ifg.shape,np.uint8)
    dest=tmp_path/'out'
    assert main(['--ifg',str(ig),'--corr',str(co),'--full-height','7',
        '--full-width','8','--height','7','--width','8',
        '--nlooks','3','--wavelength-m','0.05546576',
        '--output-dir',str(dest)],whirlwind_func=fake_ww)==0
    j=json.loads((dest/'summary.json').read_text())
    assert j['production_phase_modified'] is False
    assert j['component_gauge_invariant_comparison']['fraction_cycle_mismatch']==0
    assert (dest/'snaphu_scratch/snaphu.conf').is_file()
