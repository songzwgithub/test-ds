import numpy as np
from pypsds.unwrap.spatial_bridges import (
    candidate_pairs, occupied_components, _top_pairs, summarize_candidate_network,
)


def test_component_ids_and_diagonal_candidates():
    g = np.array([[0,-1],[-1,1]],dtype=np.int32)
    c, sizes, xy = occupied_components(g)
    assert len(sizes)==2
    assert list(c)==[0,1]
    c, sizes, pairs, dist = candidate_pairs(g,grid_size_m=200,max_distance_m=285)
    np.testing.assert_array_equal(pairs,[[0,1]])
    np.testing.assert_allclose(dist,[200*np.sqrt(2)],rtol=1e-5)


def test_does_not_count_same_component_edges_as_bridges():
    g = np.array([[0,1,-1,2]],dtype=np.int32)
    c, sizes, pairs, dist = candidate_pairs(g,grid_size_m=200,max_distance_m=410)
    assert set(map(tuple,pairs)) == {(1,2)}


def test_no_bridges_does_not_create_false_gauge():
    g = np.array([[0,-1,-1,-1,1]],dtype=np.int32)
    c, sizes, pairs, dist = candidate_pairs(g,grid_size_m=200,max_distance_m=400)
    assert pairs.shape == (0,2)
    result = summarize_candidate_network(c,sizes,pairs,dist,radii_m=[400])
    assert result[0]['screened_component_count']==2
    assert result[0]['geometric_component_count']==2


def test_redundant_support_screening_requires_both_sides():
    g = np.array([[0,1,-1,4,5],
                  [2,3,-1,6,7]],dtype=np.int32)
    c, sizes, pairs, dist = candidate_pairs(g,grid_size_m=200,max_distance_m=650)
    pairs, dist, _ = _top_pairs(c,pairs,dist,limit_per_component_pair=16)
    sup = np.full(8,16,dtype=np.int32)
    noise = np.full(len(pairs),0.2,dtype=np.float32)
    out = summarize_candidate_network(c,sizes,pairs,dist,radii_m=[650],node_support=sup,
                                       noise_std=noise,min_independent_pairs=3)
    assert out[0]['screened_component_count']==1
    assert out[0]['screened_largest_node_fraction']==1.0
    sup[4:] = 2
    out = summarize_candidate_network(c,sizes,pairs,dist,radii_m=[650],node_support=sup,
                                       noise_std=noise,min_independent_pairs=3)
    assert out[0]['screened_component_count']==2


def test_nearby_pair_list_cannot_represent_gauge_solution():
    g = np.array([[0,-1,1]],dtype=np.int32)
    c, sizes, pairs, dist = candidate_pairs(g,grid_size_m=200,max_distance_m=450)
    assert pairs.shape == (1,2)
    out = summarize_candidate_network(c,sizes,pairs,dist,radii_m=[450],
                                       min_independent_pairs=3)
    assert out[0]['candidate_component_pairs']==1
    assert out[0]['screened_component_pairs']==0
    assert out[0]['screened_component_count']==2
