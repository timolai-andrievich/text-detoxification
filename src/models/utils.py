"""Module containing code used both for training and inference.
"""
from typing import Tuple, Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class PairDataset(Dataset):
    """Constructs a dataset of reference/translation pairs.
    Ensures that the reference is more toxic than the translation.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """Initializes the dataset. When indexed, returns the tuple of
        (`reference`, `translation`, `ref_tox`, `trn_tox`, `similarity`)

        Args:
            dataframe (Dataframe): The dataframe containing the dataset.
            Must contain columns: `reference`, `translation`, `similarity`,
            `ref_tox`, `trn_tox`.
        """
        assert all(
            column in dataframe.columns for column in
            ['reference', 'translation', 'similarity', 'ref_tox', 'trn_tox'])
        self._data = []
        for _index, row in dataframe.iterrows():
            ref_text = row['reference']
            trn_text = row['translation']
            similarity = row['similarity']
            ref_tox = row['ref_tox']
            trn_tox = row['trn_tox']
            if ref_tox > trn_tox:
                ref_text, trn_text = trn_text, ref_text
                ref_tox, trn_tox = trn_tox, ref_tox
            self._data.append((
                ref_text,
                trn_text,
                ref_tox,
                trn_tox,
                similarity,
            ))

    def __getitem__(self, index: int) -> Tuple[str, str, float, float, float]:
        """Returns the `index`-th element in the dataset, as a tuple of
        (`reference`, `translation`, `ref_tox`, `trn_tox`, `similarity`).

        Args:
            index(int): The index of the element.

        Returns:
            Tuple[str, str, float, float, float]:
            (`reference`, `translation`, `ref_tox`, `trn_tox`, `similarity`).
        """
        return self._data[index]

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self._data)


class BagOfWordsLogisticClassifier(nn.Module):
    """A simple BoW logistic classifier. Implemented through
    summing the outputs of `nn.Embeddding` layer.

    Takes in the token sequences, returs logits of the toxicity
    score of the sequence.

    Input shape: `(n, s)`, `n` - batch dimension, `s` - sequence length.
    Output shape: `n`
    """

    def __init__(self, vocab_size: int):
        """A simple BoW logistic classifier. Implemented through
        summing the outputs of `nn.Embeddding` layer.

        Takes in the token sequences, returs logits of the toxicity
        score of the sequence.

        Input shape: `(n, s)`, `n` - batch dimension, `s` - sequence length.
        Output shape: `n`

        Args:
            vocab_size (int): The size of the token vocabulary.
        """
        super().__init__()
        self._classifier = nn.Embedding(vocab_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifies sequences.

        Input shape: `(n, s)`, `n` - batch dimension, `s` - sequence length.
        Output shape: `n`

        Args:
            x (Tensor): Input sequences.

        Returns:
            Tensor: The toxicity score logit for each of the sequences.
        """
        x = self._classifier(x)
        x = x[..., 0]
        x = x.sum(dim=-1)
        return x

    def get_weights(self) -> np.ndarray:
        """Return the toxicity score of each token in the dictionary.

        Returns:
            ndarray: Tensor of shape (`v`,), where `v` is the vocabulary
            size. `i`-th number corresponds to the score of the token
            with id `i`.
        """
        return self._classifier.weight.data.sigmoid().flatten().detach().cpu(
        ).numpy()


class DictionaryModel:
    """Replaces all the words present in the dictionary by synonyms.
    Tokinization of sentences should be done separately.
    """

    def __init__(self, dictionary: Dict[str, str]):
        """Replaces all the words present in the dictionary by synonyms.
        Tokinization of sentences should be done separately.

        Args:
            dictionary (Dict[str, str]): The dictionary for replacing words.
        """
        self._dictionary = dictionary

    def transform(self, sentences: List[List[str]]) -> List[List[str]]:
        """Transforms input sentences, replacing words present
        in the dictionary with synonyms from the dictionary.

        Args:
            sentences (List[List[str]]): Sentences that need to be transformed.

        Returns:
            List[List[str]]: Sentences with toxic words replaced.
        """
        return [[self._dictionary.get(word, word) for word in sentence]
                for sentence in sentences]

    def get_dictionary(self) -> Dict[str, str]:
        """Returns the internal dictionary.

        Returns:
            Dict[str, str]: Dictionary of toxic words and their synonyms.
        """
        return self._dictionary
