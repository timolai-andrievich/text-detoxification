"""Module containing code used both for training and inference.
"""
from typing import Tuple, Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset
import torchtext
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
            if ref_tox < trn_tox:
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
        self._classifier.weight.data.fill_(0)

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

    def transform(self, words: List[str]) -> List[str]:
        """Transforms input sentences, replacing words present
        in the dictionary with synonyms from the dictionary.

        Args:
            words (List[str]): Sentences that need to be transformed.

        Returns:
            List[str]: Sentences with toxic words replaced.
        """
        return [self._dictionary.get(word, word) for word in words]

    def get_dictionary(self) -> Dict[str, str]:
        """Returns the internal dictionary.

        Returns:
            Dict[str, str]: Dictionary of toxic words and their synonyms.
        """
        return self._dictionary


class RNN(nn.Module):
    """A simple recurrent neural network using LSTM as a backbone.
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int):
        """A simple recurrent neural network using LSTM as a backbone.

        Args:
            hidden_size (int): The hidden dimension of LSTM and embeddings.
            vocab_size (int): The size of the vocabulary.
            num_layers (int): The number of layers in LSTM.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size,
                               hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        self.decoder = nn.LSTM(hidden_size,
                               hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, reference: torch.Tensor, translation: torch.Tensor):
        ref = self.embed(reference)
        trn = self.embed(translation)
        _, context = self.encoder(ref)
        x, _ = self.decoder(trn, context)
        x = self.fc(x)
        return x


class RNNModel:

    def __init__(self,
                 vocab: torchtext.vocab.Vocab,
                 model: RNN,
                 max_len=100,
                 device='cpu'):
        self.vocab = vocab
        self.model = model
        self._bos_index = vocab.get_stoi()['<bos>']
        self._unk_index = vocab.get_default_index()
        self.max_len = max_len
        self.device = device

    def transform(self, sentence: List[str]) -> List[str]:
        generated_indicies = [self._bos_index]
        sentence_indicies = self.vocab(sentence)
        sentence_tensor = torch.tensor(sentence_indicies).view(1, -1)
        for _ in range(self.max_len):
            generated_tensor = torch.tensor(generated_indicies).view(1, -1)
            outputs = self.model(sentence_tensor.to(self.device),
                                 generated_tensor.to(self.device))
            outputs[:, :, self._unk_index] = -torch.inf
            new_token = outputs.argmax(dim=-1)[0, -1].item()
            if self.vocab.lookup_token(new_token) == '<eos>':
                break
            generated_indicies.append(new_token)
        generated_words = [
            self.vocab.lookup_token(token) for token in generated_indicies
        ]
        return generated_words[1:]
