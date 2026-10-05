import numpy as np

from ddz.elo_ladder import bootstrap_intervals, fit_elo, planned_edges


def edge(candidate, reference, points):
    return {'candidate_step': candidate, 'reference_step': reference,
            'pair_points': points}


def test_identical_role_mirrored_results_anchor_all_ratings():
    steps = [0, 500, 1000]
    edges = [edge(500, 0, [1] * 24), edge(1000, 0, [1] * 24),
             edge(1000, 500, [1] * 24)]
    np.testing.assert_allclose(fit_elo(steps, edges), [1000, 1000, 1000], atol=1e-5)


def test_ladder_graph_and_stronger_checkpoint_order():
    steps = list(range(0, 4001, 500))
    graph = planned_edges(steps)
    assert (500, 0) in graph and (4000, 3500) in graph
    assert (4000, 3000) in graph and (4000, 0) in graph
    ratings = fit_elo([0, 500, 1000], [
        edge(500, 0, [2] * 12 + [1] * 12),
        edge(1000, 0, [2] * 20 + [1] * 4),
        edge(1000, 500, [2] * 15 + [1] * 9),
    ])
    assert ratings[2] > ratings[1] > ratings[0]


def test_perfect_sweep_retains_uncertainty():
    matchup = edge(500, 0, [2] * 24)
    estimate = fit_elo([0, 500], [matchup])
    lower, upper = bootstrap_intervals([0, 500], [matchup],
                                       rounds=100, seed=41)
    assert lower[0] == upper[0] == 1000
    assert lower[1] < estimate[1] < upper[1]
    assert upper[1] - lower[1] > 20
