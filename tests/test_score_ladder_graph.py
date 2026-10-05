import numpy as np

from ddz.score_ladder_graph import fit_score_graph, planned_edges


def test_every_later_checkpoint_faces_recent_three_and_initial_anchor():
    steps = list(range(0, 6501, 500))
    edges = planned_edges(steps)
    assert len(edges) == 46
    assert (6500, 0) in edges
    assert {(6500, 5000), (6500, 5500), (6500, 6000)}.issubset(edges)
    assert (6500, 4500) not in edges


def test_expected_score_graph_fits_pairwise_scores_and_common_bootstrap():
    steps = [0, 500, 1000, 1500]
    strength = {0: 0., 500: 1., 1000: 2., 1500: 3.}
    matches = []
    for a, b in planned_edges(steps):
        pair = np.array([-1., 0., 1.]) + strength[a] - strength[b]
        matches.append({'candidate_step': a, 'reference_step': b,
                        'paired_mean_scores': pair.tolist(), 'games': 6})
    rows, residual = fit_score_graph(steps, matches, rounds=500, seed=41)
    np.testing.assert_allclose([r['expected_score'] for r in rows], [0., 1., 2., 3.])
    np.testing.assert_allclose([r['change_from_previous'] for r in rows[1:]], [1., 1., 1.])
    np.testing.assert_allclose(residual, 0., atol=1e-12)
    assert rows[-1]['opponents'] == [0, 500, 1000]
    assert rows[-1]['score_low'] < 3. < rows[-1]['score_high']
