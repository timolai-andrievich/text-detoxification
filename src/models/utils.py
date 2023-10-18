"""Module containing code used both for training and inference.
"""
from typing import Tuple

from torch.utils.data import Dataset
import pandas as pd


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
