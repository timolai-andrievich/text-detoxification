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
    assert dataset[0] == ('toxic 1', 'normal 1', 1.0, 0.0, 0.5)
    assert dataset[1] == ('toxic 2', 'normal 2', 1.0, 0.0, 0.5)


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


def test_dictionary_model():
    """Tests `DictionaryModel` class.
    """
    dictionary = {'a': 'A'}
    model = utils.DictionaryModel(dictionary)
    text = ['a', 'B', 'B', 'a', 'B', 'C']
    transformed_text = model.transform(text)
    target_text = ['A', 'B', 'B', 'A', 'B', 'C']
    assert transformed_text == target_text


def test_rnn():
    """Tests `RNN` class.
    """
    # A simple sanity check by making sure it can overfit on simple datac
    torch.random.manual_seed(42)
    dummy_inputs = torch.randint(0, 10, (16, 32))
    dummy_translation = torch.randint(0, 10, (16, 33))
    dummy_target = dummy_translation[:, 1:]
    dummy_translation = dummy_translation[:, :-1]
    loss_fn = torch.nn.CrossEntropyLoss()
    model = utils.RNN(16, 10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(400):
        pred = model(dummy_inputs, dummy_translation)
        pred = torch.flatten(pred, end_dim=-2)
        target = torch.flatten(dummy_target)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    outputs = model(dummy_inputs, dummy_translation)
    accuracy = torch.mean(outputs.argmax(dim=-1) == dummy_target, dtype=float)
    assert accuracy > .9
