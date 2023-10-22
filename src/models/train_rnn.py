import argparse
from collections import Counter
import copy
from typing import TypedDict

import nltk
from nltk import tokenize
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchtext
import tqdm

import utils


def get_vocab(dataset: utils.PairDataset, specials) -> torchtext.vocab.Vocab:
    counter = Counter()
    pbar = tqdm.tqdm(total=len(dataset), desc='Building vocabulary')
    for ref_txt, trn_txt, *_ in dataset:
        ref_words = tokenize.word_tokenize(ref_txt)
        trn_words = tokenize.word_tokenize(trn_txt)
        counter.update(ref_words)
        counter.update(trn_words)
        pbar.update(1)
    pbar.close()
    vocab = torchtext.vocab.vocab(counter, specials=specials)
    return vocab


class Args(TypedDict):
    dataset_file: str
    learning_rate: float
    epochs: int
    pad_len: int
    batch_size: int
    device: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        dest='dataset_file')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--pad-len', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')
    args = parser.parse_args()
    return args


def evaluate_model(model, loader, loss_fn):
    model.eval()
    losses = []
    total_len = 0
    with torch.no_grad():
        for ref, trn in loader:
            pred = model(ref)
            pred = torch.flatten(pred, end_dim=-2)
            target = torch.flatten(trn)
            loss = loss_fn(pred, target)
            total_len += len(ref)
            losses.append(loss.item() * len(ref))
    mean_loss = np.sum(losses) / total_len
    return {'Loss': mean_loss}


def train_model(model, vocab, args, train_loader, val_loader):
    pad_index = vocab.get_stoi()['<pad>']
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pbar = tqdm.tqdm(total=len(train_loader) * args.epochs)
    best = None
    last = None
    best_val_loss = float('inf')
    postfix = {}
    for epoch in range(args.epochs):
        train_losses = []
        train_total = 0
        model.train()
        postfix.update({'Epoch': epoch})
        for ref, trn in train_loader:
            ref_pred = model(ref)
            ref_pred = torch.flatten(ref_pred, end_dim=-2)
            trn = torch.flatten(trn)
            loss = loss_fn(ref_pred, trn)
            optimizer.zero_grad()
            loss.backward()
            train_losses.append(loss.item() * len(ref))
            optimizer.step()
            pbar.update(1)
            train_total += len(ref)
        train_loss = np.mean(train_losses) / train_total
        postfix.update({'Mean train loss': f'{train_loss:.4f}'})
        pbar.set_postfix(postfix)
        last = copy.deepcopy(model.state_dict())
        metrics = evaluate_model(model, val_loader, loss_fn)
        val_loss = metrics['Loss']
        postfix.update({'Mean validation loss': f'{val_loss:.4f}'})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best = copy.deepcopy(model.state_dict())
        pbar.set_postfix(postfix)
    pbar.close()
    return best, last


def main():
    nltk.download('punkt', quiet=True)
    args = parse_args()
    dataframe = pd.read_csv(args.dataset_file, sep='\t')
    dataframe = dataframe.head(1000)  # DEBUG
    dataset = utils.PairDataset(dataframe)
    vocab = get_vocab(dataset, specials=['<unk>', '<pad>'])
    pad_index = vocab.get_stoi()['<pad>']
    unk_index = vocab.get_stoi()['<unk>']
    vocab.set_default_index(unk_index)
    model = utils.RNN(64, len(vocab), 1).to(args.device)

    def collate_batch(batch):
        refs = []
        trns = []
        for ref, trn, *_ in batch:
            ref_words = tokenize.word_tokenize(ref)
            ref_tokens = vocab(ref_words)
            ref_tokens = ref_tokens[:args.pad_len]
            ref_tokens = ref_tokens + [pad_index] * \
                (args.pad_len - len(ref_tokens))
            refs.append(ref_tokens)
            trn_words = tokenize.word_tokenize(trn)
            trn_tokens = vocab(trn_words)
            trn_tokens = trn_tokens[:args.pad_len]
            trn_tokens = trn_tokens + [pad_index] * \
                (args.pad_len - len(trn_tokens))
            trns.append(trn_tokens)
        return torch.tensor(refs).to(args.device), torch.tensor(trns).to(
            args.device)

    val_proportion, test_proportion = .1, .1
    test_size = int(len(dataset) * test_proportion)
    val_size = int(len(dataset) * val_proportion)
    train_size = len(dataset) - val_size - test_size

    train_ds, val_ds, test_ds = random_split(dataset,
                                             (train_size, val_size, test_size))

    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              collate_fn=collate_batch)
    val_loader = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            collate_fn=collate_batch)
    test_loader = DataLoader(test_ds,
                             batch_size=args.batch_size,
                             collate_fn=collate_batch)
    best, last = train_model(model, vocab, args, train_loader, val_loader)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    model.load_state_dict(best)
    torch.save(model, 'best_rnn.pt')
    best_metrics = evaluate_model(model, test_loader, loss_fn)
    print('Metrics of the best model on test set:')
    for name, value in best_metrics.items():
        print(f'{name}: {value:.4f}')
    model.load_state_dict(last)
    torch.save(model, 'last_rnn.pt')
    last_metrics = evaluate_model(model, test_loader, loss_fn)
    print('Metrics of the last model on test set:')
    for name, value in last_metrics.items():
        print(f'{name}: {value:.4f}')


if __name__ == '__main__':
    main()
