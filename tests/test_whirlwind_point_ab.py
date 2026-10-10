"""Tests for phase-linked PS/DS input A/B, without requiring Whirlwind installed."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from pypsds.unwrap.whirlwind_point_ab import (
    build_point_ifg, read_network, main, _baseline_comparison,
)


def data():
    grid = np.array([[0, 1, -1, 2], [3, -1, -1, 4], [5, 6, -1, 7]], np.int32)
    ph = np.arange(8 * 3, dtype=np.float32).reshape(8, 3) * 0.19
    return ph, grid


def test_point_ifg_uses_linked_acquisition_difference_not_gamma():
    ph, grid = data()
    ifg, q, mask = build_point_ifg(ph, grid, 0, 2)
    assert mask.sum() == 8
    assert np.all(ifg[~mask] == 0)
    assert np.all(q[~mask] == 0)
    np.testing.assert_allclose(np.angle(ifg[mask]), ph[grid[mask], 2] - ph[grid[mask], 0], atol=1e-6)
    assert np.all(q[mask] == np.float32(0.8))


def test_resultant_quality_is_proxy_and_does_not_fill_gaps():
    ph, grid = data()
    res = np.ones_like(ph)
    res[:, 0] = 0.64
    res[:, 2] = 0.81
    _, q, mask = build_point_ifg(ph, grid, 0, 2, quality_mode="resultant", resultant_length=res)
    np.testing.assert_allclose(q[mask], 0.72, rtol=1e-6)
    assert (q[~mask] == 0).all()


def test_phase_mode_refuses_missing_proxy_and_invalid_grid():
    ph, grid = data()
    with pytest.raises(ValueError, match="resultant"):
        build_point_ifg(ph, grid, 0, 2, quality_mode="resultant")
    bad = grid.copy(); bad[0,1] = 0
    with pytest.raises(ValueError, match="contiguous node ID"):
        build_point_ifg(ph, bad, 0, 2)


def test_baseline_wrapped_parity(tmp_path):
    ph, grid = data()
    out = tmp_path / "baseline"; out.mkdir()
    base = ph + 2 * np.pi * np.arange(8)[:, None]
    np.save(out / "coarse_acquisition_phase_unwrapped_rad.npy", base)
    ifg, _, mask = build_point_ifg(ph, grid, 0, 2)
    uw = np.angle(ifg); cc = np.where(mask, 1, 0).astype(np.int32)
    stats = _baseline_comparison(baseline_dir=out, node_phase=ph, node_grid=grid,
                                 index_i=0, index_j=2, solver_uw=uw, solver_cc=cc, mask=mask)
    assert stats["node_phase_max_wrapped_difference_rad"] < 1e-5
    assert stats["solver_vs_statcost"]["comparable_pixels"] == 8


def test_itab_one_based(tmp_path):
    p=tmp_path/'network.itab';p.write_text('1 2 0 0\n2 3 0 0\n')
    assert read_network(p,3)==[(0,1),(1,2)]
    p.write_text('0 3\n')
    with pytest.raises(ValueError, match='Out-of-bounds'):
        read_network(p,3)


def test_full_audit_cli_no_solver(tmp_path):
    ph,grid=data()
    pp=tmp_path/'ph.npy';np.save(pp,ph)
    gg=tmp_path/'grid.npy';np.save(gg,grid)
    dates=tmp_path/'dates.txt';dates.write_text('20240101 20240113 20240125')
    itab=tmp_path/'network.itab';itab.write_text('1 3\n')
    out=tmp_path/'out'
    assert main(['--node-phase',str(pp),'--node-grid',str(gg),'--dates',str(dates),
                 '--itab',str(itab),'--output-dir',str(out),'--audit-only']) == 0
    summary=json.loads((out/'input_audit.json').read_text())
    assert summary['valid_nodes'] == 8
    assert summary['quality_is_true_interferometric_coherence'] is False
    assert summary['empty_cells_interpolated'] is False


def test_full_ab_with_mock_solvers(tmp_path,monkeypatch):
    ph,grid=data()
    pp=tmp_path/'ph.npy';np.save(pp,ph)
    gg=tmp_path/'grid.npy';np.save(gg,grid)
    dates=tmp_path/'dates.txt';dates.write_text('20240101 20240113 20240125')
    itab=tmp_path/'network.itab';itab.write_text('1 3\n')
    def fakeww(igram,corr,**kw):
        assert kw['mask'].sum() == 8
        assert kw['bridge'] is False
        cc=np.where(kw['mask'],1,0).astype(np.int32)
        return np.angle(igram).astype(np.float32),cc
    def fakesnap(igram,corr,mask,nlooks,wavelength_m,*,scratch,executable):
        return np.angle(igram).astype(np.float32), np.where(mask,1,0).astype(np.int32), {'seconds':0,'wrap_parity_max_rad':0}
    out=tmp_path/'out'
    assert main(['--node-phase',str(pp),'--node-grid',str(gg),'--dates',str(dates),
                 '--itab',str(itab),'--output-dir',str(out),'--nlooks','1','--wavelength-m','0.056'],
                whirlwind_func=fakeww,snaphu_func=fakesnap) == 0
    summary=json.loads((out/'summary.json').read_text())
    assert summary['gauge_invariant_solver_comparison']['fraction_cycle_mismatch'] == 0
    assert np.load(out/'whirlwind_node_ifg_unwrapped_rad.npy').shape == (8,)
    assert summary['production_phase_modified'] is False
