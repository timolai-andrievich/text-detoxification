"""Test module. Run via `pytest src`
"""
import pandas as pd

import utils


def test_pair_dataset():
    """Tests the `PairDataset` class.
    """
    data = {
        'reference': ['toxic 1', 'normal 2'],
        'translation': ['normal 1', 'toxic 2'],
        'similarity': [0.5, 0.5],
        'ref_tox': [1.0, 0.0],
        'trn_tox': [0.0, 1.0],
    }
    dataframe = pd.DataFrame(data=data)
    dataset = utils.PairDataset(dataframe)
    assert len(dataset) == 2
    assert dataset[0] == ('normal 1', 'toxic 1', 0.0, 1.0, 0.5)
    assert dataset[1] == ('normal 2', 'toxic 2', 0.0, 1.0, 0.5)
