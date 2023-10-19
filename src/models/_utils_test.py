"""Test module. Run via `pytest src`
"""
import pandas as pd
import torch
from torch.nn import functional as F
import numpy as np

from . import utils


def test_pair_dataset():
    """Tests the `PairDataset` class.
    """
    np.random.seed(42)
    torch.manual_seed(42)
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


def test_logistic_classifier():
    """Tests the `BagOfWordsLogisticClassifier` class.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    x = torch.tensor([[0, 1, 0, 1], [0, 2, 0, 2]]).long()
    y = torch.tensor([0, 1]).float()
    classifier = utils.BagOfWordsLogisticClassifier(3)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=1)
    for _ in range(100):
        pred = classifier(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert np.max(np.abs(classifier.get_weights() - np.array([.5, 0, 1]))) < .1
