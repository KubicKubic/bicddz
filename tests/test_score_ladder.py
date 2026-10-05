import xml.etree.ElementTree as ET

import numpy as np

from ddz.score_ladder import make_svg, score_summary


def test_score_ladder_uses_common_deal_resampling_for_changes():
    matches = [
        {'candidate_step': 500, 'paired_mean_scores': [1., 3., 5.],
         'candidate_wins': 4, 'games': 6},
        {'candidate_step': 1000, 'paired_mean_scores': [2., 4., 6.],
         'candidate_wins': 5, 'games': 6},
    ]
    rows = score_summary([0, 500, 1000], matches, rounds=500, seed=41)
    np.testing.assert_allclose([r['expected_score'] for r in rows], [0., 3., 4.])
    np.testing.assert_allclose([rows[2]['change_low'], rows[2]['change_high']], [1., 1.])
    assert rows[1]['score_low'] < rows[1]['expected_score'] < rows[1]['score_high']
    ET.fromstring(make_svg(rows))
