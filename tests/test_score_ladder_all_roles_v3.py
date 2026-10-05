import numpy as np

from ddz.score_ladder_all_roles_v3 import ROLE_NAMES, paired_scores
from ddz.score_ladder_graph_all_roles_v3 import _metric_rows


def test_complementary_leg_scores_are_equal_role_weighted_and_antisymmetric():
    assert ROLE_NAMES == ('landlord', 'landlord_next', 'door')
    # Two complementary games for each role, two common deals.
    legs = np.array([
        [[8., 12.], [4., 8.]],
        [[3., 5.], [1., 3.]],
        [[-2., 0.], [2., 4.]],
    ])
    roles, total = paired_scores(legs, 2, forced_bid=True)
    np.testing.assert_allclose(roles, [[3., 5.], [2., 4.], [0., 2.]])
    np.testing.assert_allclose(total, [5 / 3, 11 / 3])
    swapped, swapped_total = paired_scores(-legs[:, ::-1], 2, forced_bid=True)
    np.testing.assert_allclose(swapped, -roles)
    np.testing.assert_allclose(swapped_total, -total)


def test_natural_auction_graph_uses_all_three_seats():
    steps = [0, 500]
    match = {'candidate_step': 500, 'reference_step': 0,
             'deals': 3, 'natural_paired_scores': [1., 2., 3.]}
    rows, residual = _metric_rows(steps, [match], 'natural_paired_scores',
                                  rounds=200, seed=41, games_per_deal=6)
    assert rows[-1]['expected_score'] == 2.
    assert rows[-1]['games'] == 18
    assert residual == 0.
