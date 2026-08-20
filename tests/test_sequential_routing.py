import numpy as np

from pypsds.phase_linking.sequential_routing import (
    build_sequential_routing,
)


def test_routing_partition():

    prior = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    )

    valid = np.ones_like(
        prior,
        dtype=bool,
    )

    ps = np.zeros_like(
        prior,
        dtype=bool,
    )

    # One prior pixel is PS and therefore
    # must never enter formal DS.
    ps[1, 1] = True

    original_k = np.array(
        [
            [60, 55, 47, 80],
            [70, 90, 90, 90],
        ],
        dtype=np.int16,
    )

    effective_k = np.array(
        [
            [58, 35, 47, 80],
            [48, 90, 90, 90],
        ],
        dtype=np.int16,
    )

    r = build_sequential_routing(
        center_prior=prior,
        valid=valid,
        ps=ps,
        original_shp_count=original_k,
        effective_shp_count=effective_k,
        formal_min_shp=48,
        state_min_shp=24,
    )

    # Formal DS ignores the historical center prior.
    #
    # valid non-PS and original K>=48:
    # (0,0), (0,1), (0,3),
    # (1,0), (1,2), (1,3)
    assert r.formal_count == 6

    # Effective K>=48:
    # all formal except (0,1), whose effective K=35.
    assert r.sequential_count == 5

    # Remaining formal DS uses original-support full SCM.
    assert r.fallback_count == 1

    # These were excluded by the old center_prior,
    # but satisfy the formal GLRT definition.
    assert r.formal_ds[0, 3]
    assert r.formal_ds[1, 2]
    assert r.formal_ds[1, 3]

    assert not np.any(
        r.sequential
        &
        r.fallback
    )

    np.testing.assert_array_equal(
        r.sequential
        |
        r.fallback,
        r.formal_ds,
    )


def test_state_threshold_does_not_change_formal_ds():

    prior = np.ones(
        (1, 3),
        dtype=bool,
    )

    valid = np.ones_like(
        prior,
    )

    ps = np.zeros_like(
        prior,
    )

    original_k = np.array(
        [[48, 60, 47]],
        dtype=np.int16,
    )

    effective_k = np.array(
        [[48, 24, 47]],
        dtype=np.int16,
    )

    r = build_sequential_routing(
        center_prior=prior,
        valid=valid,
        ps=ps,
        original_shp_count=original_k,
        effective_shp_count=effective_k,
        formal_min_shp=48,
        state_min_shp=24,
    )

    assert r.formal_count == 2
    assert r.sequential_count == 1
    assert r.fallback_count == 1


def test_center_prior_cannot_remove_formal_ds():

    prior = np.zeros(
        (1, 3),
        dtype=bool,
    )

    valid = np.ones(
        (1, 3),
        dtype=bool,
    )

    ps = np.zeros(
        (1, 3),
        dtype=bool,
    )

    original_k = np.array(
        [[60, 47, 80]],
        dtype=np.int16,
    )

    effective_k = np.array(
        [[60, 47, 30]],
        dtype=np.int16,
    )

    r = build_sequential_routing(
        center_prior=prior,
        valid=valid,
        ps=ps,
        original_shp_count=original_k,
        effective_shp_count=effective_k,
        formal_min_shp=48,
        state_min_shp=24,
    )

    # Formal eligibility comes from exact GLRT K,
    # not from the all-false diagnostic prior.
    assert r.formal_count == 2
    assert r.formal_ds[0, 0]
    assert r.formal_ds[0, 2]

    assert r.sequential_count == 1
    assert r.fallback_count == 1
