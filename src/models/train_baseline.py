#!python
"""Script for training the baseline dictionary model.
"""
import argparse
from collections import Counter, OrderedDict
import json
import os
import sys
import random  # For seeding
from typing import TypedDict, List, Tuple, Optional

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchtext
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import tqdm
import tqdm.notebook

import utils


class Args(TypedDict):
    dataset_file: str
    quiet: bool
    batch_size: int
    epochs: int
    seed: Optional[int]
    dict_len: int
    output_file: int
    device: str
    learning_rate: float


def parse_args() -> Args:
    """Parses command line arguments and returns them
    as a namespace object.

    Returns:
        Args: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--quiet',
                        action='store_true',
                        dest='quiet',
                        help='Whether to display messages or not.')
    parser.add_argument('--dataset-file',
                        type=str,
                        dest='dataset_file',
                        required=True,
                        help='Path to the .tsv dataset file.')
    parser.add_argument('--epoch',
                        type=int,
                        dest='epochs',
                        default=5,
                        help='Number of training epochs.')
    parser.add_argument('--batch-size',
                        type=int,
                        dest='batch_size',
                        default=64,
                        help='The size of training minibatches.')
    parser.add_argument('--seed',
                        type=int,
                        dest='seed',
                        help='Seed for random generators. '
                        'Random if not specified.',
                        default=None)
    parser.add_argument('--dictionary-length',
                        type=int,
                        dest='dict_len',
                        help='The length of the resulting dictionary. '
                        'Dictionary might be shorter if the synonyms '
                        'for certain words are not found.',
                        default=100)
    parser.add_argument('--output-file',
                        type=str,
                        dest='output_file',
                        help='The path to the output JSON file. Will be '
                        'created if doesn\'t exist, and overwritten '
                        'otherwise.',
                        default='dictionary.json')
    parser.add_argument('--device',
                        type=str,
                        choices=['cuda', 'cpu'],
                        dest='device',
                        default='cpu',
                        help='Device to train the model on. Defaults to cpu.')
    parser.add_argument('--learning-rate',
                        type=float,
                        dest='learning_rate',
                        default=1e-3,
                        help='Learning rate passed to the optimizer. '
                        '0.001 by default')
    args = parser.parse_args()
    return args


def collate_batch(batch: List[Tuple[str, str, float, float, float]],
                  vocabulary: torchtext.vocab.Vocab,
                  pad_len: int = 200,
                  device: str = 'cpu'):
    """Pads the sequences and converts them to tensors.

    Args:
      batch (List[Tuple[str, str, float, float, float]]): A batch of tuples (
        `reference_text: str`,
        `translation_text: str`,
        `reference_toxicity: float`
        `translation_toxicity: float`,
        `similarity: float`
      )
      vocabulary (Vocab): torchtext vocabulary to tokenize the words.
      pad_len (int): The length the sequences are getting padded to. Defaults
      to 200.
      device (str): Device to put the tensors on.

    Returns:
      Tuple[Tensor, Tensor, Tensor, Tensor]: Batched tensors: (
        `reference_tokens`,
        `translation_tokens`,
        `reference_toxicity`,
        `reference_toxicity`,
      )
    """

    rref = []
    rtrn = []
    rrt = []
    rtt = []
    pad_token = vocabulary.get_stoi()['<pad>']
    for ref, trn, ref_tox, trn_tox, _similarity in batch:
        ref_tokens = vocabulary(word_tokenize(ref.lower()))
        ref_tokens = ref_tokens[:pad_len]
        ref_tokens = ref_tokens + [pad_token] * (pad_len - len(ref_tokens))

        trn_tokens = vocabulary(word_tokenize(trn.lower()))
        trn_tokens = trn_tokens[:pad_len]
        trn_tokens = trn_tokens + [pad_token] * (pad_len - len(trn_tokens))
        rref.append(ref_tokens)
        rtrn.append(trn_tokens)
        rrt.append(ref_tox)
        rtt.append(trn_tox)
        return tuple(
            map(lambda t: t.to(device),
                map(torch.tensor, [rref, rtrn, rrt, rtt])))


def build_vocab(
        dataset: Dataset,
        quiet: bool = False) -> Tuple[torchtext.vocab.Vocab, Counter[str]]:
    """Builds vocabulary from a torch dataset.

    Args:
        dataset (Dataset): The dataset to build vocabulary from.
        quiet (bool): Whether to hide progress bar or not. Defaults to `False`.

    Returns:
        Vocab: Torchtext vocabulary containing words from the dataset. Contains
        `<unk>` and `<pad>` tokens.
        Counter[str]: Counter with frequencies of words in dataset.
    """
    counter = Counter()
    with tqdm.tqdm(total=len(dataset), desc='Building vocabulary') as pbar:
        for ref_text, trn_text, *_ in dataset:
            ref_words = word_tokenize(ref_text.lower())
            counter.update(ref_words)
            trn_words = word_tokenize(trn_text.lower())
            counter.update(trn_words)
            pbar.update(1)
    sorted_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    vocabulary = torchtext.vocab.vocab(OrderedDict(sorted_tuples),
                                       specials=['<pad>', '<unk>'],
                                       special_first=False)
    unk_index = vocabulary.get_stoi()['<unk>']
    vocabulary.set_default_index(unk_index)
    return vocabulary, counter


def train_model(classifier: nn.Module, train_loader: DataLoader,
                test_loader: DataLoader, args: Args):
    """Trains the classifier. Modifies the model in-place.

    Args:
        classifier (Module): Logistic classifier model.
        train_loader (DataLoader): Train data loader.
        test_loader (DataLoader): Test data loader.
        args (Args): Arguments namepace.
    """
    classifier = classifier.to(args.device)
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    with tqdm.tqdm(total=args.epochs * len(train_loader),
                   desc='Training',
                   disable=args.quiet) as pbar, tqdm.tqdm(
                       total=len(test_loader),
                       desc='Test',
                       disable=args.quiet,
                       position=1) as test_pbar:
        for epoch in range(args.epochs):
            classifier.train()
            losses = []
            for ref_tokens, trn_tokens, ref_tox, trn_tox in train_loader:
                assert torch.all(ref_tox > trn_tox)
                ref_pred = classifier(ref_tokens)
                ref_target = torch.ones_like(ref_tox)
                ref_loss = F.binary_cross_entropy_with_logits(
                    ref_pred, ref_target)
                trn_pred = classifier(trn_tokens)
                trn_target = torch.zeros_like(trn_tox)
                trn_loss = F.binary_cross_entropy_with_logits(
                    trn_pred, trn_target)
                loss = ref_loss + trn_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                pbar.update(1)
            pbar.refresh()
            classifier.eval()
            total_correct = 0
            total = 0
            test_pbar.reset()
            with torch.no_grad():
                for ref_tokens, trn_tokens, *_ in test_loader:
                    ref_pred = classifier(ref_tokens)
                    ref_accurate = torch.sum((ref_pred >= 0).long())

                    trn_pred = classifier(trn_tokens)
                    trn_accurate = torch.sum((trn_pred < 0).long())

                    total += ref_tokens.shape[0] + trn_tokens.shape[0]
                    total_correct += ref_accurate + trn_accurate
                    test_pbar.update(1)
            test_pbar.refresh()
            accuracy = total_correct / total
            train_loss = np.mean(losses)
            pbar.set_postfix({
                'Epoch': epoch + 1,
                'Test accuracy': f'{accuracy * 100:6.2f}%',
                'Train loss': f'{train_loss:.4f}'
            })


def get_toxicity(vocab: torchtext.vocab.Vocab,
                 classifier: utils.BagOfWordsLogisticClassifier,
                 counter: Counter[str]) -> np.ndarray:
    """Returns a list of toxicity scores for token indicies.

    Args:
        vocab (Vocab): Vocabulary with words that were scored.
        classifier (BagOfWordsLogisticClassifier): Logistic BoW classifier.
        counter (Counter[str]): Counter with frequencies of words in dataset.

    Returns:
        ndarray: Numpy array with toxicity scores. `i`-th number corresponds
        to the token with index `i`.
    """
    raw_toxicity = classifier.get_weights()
    frequencies = np.array(
        [counter[vocab.lookup_token(index)] for index in range(len(vocab))])
    # Because words that are rare might have toxicity that is too high,
    # correct scores of words based on their frequency, using mean of beta
    # distribution, and treating raw scores as the number of times word occurs
    # in toxic sentences divided by total number of occurences.
    # That also makes distribution of toxicity closer to gaussian.
    corrected_toxicity = (raw_toxicity * frequencies + 1) / (frequencies + 2)
    corrected_toxicity[vocab.get_stoi()['<pad>']] = .5
    corrected_toxicity[vocab.get_stoi()['<unk>']] = .5
    return corrected_toxicity


def main():
    """Main function of the script.
    """
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    if not os.path.exists(args.dataset_file) or not os.path.isfile(
            args.dataset_file):
        print(f'File {args.dataset_file} does not exist.')
        sys.exit(1)
    dataframe = pd.read_csv(args.dataset_file, sep='\t', index_col=0)
    test_proportion = .1
    dataset = utils.PairDataset(dataframe)
    test_len = int(len(dataset) * test_proportion)
    train_len = len(dataset) - test_len
    train_dataset, test_dataset = random_split(dataset, (train_len, test_len))
    vocabulary, counter = build_vocab(train_dataset, args.quiet)

    def collate_fn(batch):
        return collate_batch(batch, vocabulary, device=args.device)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             collate_fn=collate_fn)

    classifier = utils.BagOfWordsLogisticClassifier(len(vocabulary))
    train_model(classifier, train_loader, test_loader, args)
    toxicity = get_toxicity(vocabulary, classifier, counter)
    toxic_tokens = toxicity.argsort()[::-1][:args.dict_len]
    toxic_words = vocabulary.lookup_tokens(toxic_tokens)

    def get_synonyms(word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return synonyms

    synonyms_dict = {}
    for word in toxic_words:
        synonyms = get_synonyms(word)
        synonyms_dict[word] = synonyms

    def map_toxicity(word: str):
        if word not in vocabulary:
            return 1
        return toxicity[vocabulary.get_stoi()[word]]

    filtered_synonyms = {
        word: min(synonyms, key=map_toxicity)
        for word, synonyms in tqdm.tqdm(synonyms_dict.items(),
                                        desc='Finding synonyms') if synonyms
    }
    with open(args.output_file, 'w', encoding='utf-8') as file:
        json.dump(filtered_synonyms, file)


if __name__ == "__main__":
    main()
