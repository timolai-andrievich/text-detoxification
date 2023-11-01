import argparse
from collections import Counter
import copy
from typing import TypedDict, List, Tuple

import nltk
from nltk import tokenize
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils import tensorboard as torchboard
import torchtext
import tqdm

import utils


class Args(TypedDict):
    """Command line arguments.
    """
    dataset_file: str
    learning_rate: float
    epochs: int
    pad_len: int
    batch_size: int
    device: str
    d_model: int
    num_layers: int
    max_tokens: int
    seed: int
    log_dir: str


def get_vocab(dataset: utils.PairDataset, specials: List[str],
              args: Args) -> torchtext.vocab.Vocab:
    """Builds vocabulary from a dataset.

    Args:
        dataset (PairDataset): Dataset containing reference-translation
        pairs.
        specials (List[str]): List of special tokens.
        args (Args): Command line arguments.

    Returns:
        Vocab: `torchtext.vocab.Vocab` vocabulary.
    """
    counter = Counter()
    pbar = tqdm.tqdm(total=len(dataset), desc='Building vocabulary')
    for ref_txt, trn_txt, *_ in dataset:
        ref_words = tokenize.word_tokenize(ref_txt.lower())
        trn_words = tokenize.word_tokenize(trn_txt.lower())
        counter.update(ref_words)
        counter.update(trn_words)
        pbar.update(1)
    pbar.close()
    words = list(sorted(counter.keys(), key=lambda x: counter[x],
                        reverse=True))[:args.max_tokens]
    vocab = torchtext.vocab.vocab({word: counter[word]
                                   for word in words},
                                  specials=specials)
    return vocab


def parse_args() -> Args:
    """Parses command line arguments.

    Returns:
        Args: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Name of the dataset file in .tsv format.',
                        dest='dataset_file')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate of the model.')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of training epochs.')
    parser.add_argument('--pad-len',
                        type=int,
                        default=200,
                        help='Length that the sequences are being padded to.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Size of training minibatches.')
    parser.add_argument('--device',
                        choices=['cuda', 'cpu'],
                        default='cpu',
                        help='The device the model is trained on')
    parser.add_argument('--layers',
                        type=int,
                        dest='num_layers',
                        default=1,
                        help='The number of LSTM layers in the model.')
    parser.add_argument('--d-model',
                        type=int,
                        dest='d_model',
                        default=64,
                        help='The dimensionality of token embeddings.')
    parser.add_argument('--max-vocab-size',
                        type=int,
                        dest='max_tokens',
                        help='Maximum size of vocabulary',
                        default=4096)
    parser.add_argument('--seed',
                        type=int,
                        dest='seed',
                        help='Seed for random generators. '
                        'Random if not specified.',
                        default=None)
    parser.add_argument('--log-dir',
                        type=int,
                        dest='log_dir',
                        help='A directory for tensorboard logs. '
                        './runs/ by default.',
                        default=None)
    args = parser.parse_args()
    return args


def evaluate_model(model: nn.Module, loader: DataLoader, loss_fn) -> dict[str, float]:
    """Evaluates model and returns dictionary of metrics.
    Metrics:
    - Loss

    Args:
        model (Module): Model to be evaluated.
        loader (DataLoader): Evaluation data loader.
        loss_fn (Tensor -> Tensor): Loss function. Passed because `pad_index`
        should be ignored when calculating loss.

    Returns:
        Dict[str, float]: Dictionary of metric name -> metric value.
    """
    model.eval()
    losses = []
    total_len = 0
    with torch.no_grad():
        for ref, trn in loader:
            translation = trn[:, :-1]
            pred = model(ref, translation)
            target = trn[:, 1:]
            target = torch.flatten(target)
            pred = torch.flatten(pred, end_dim=-2)
            loss = loss_fn(pred, target)
            total_len += len(ref)
            losses.append(loss.item() * len(ref))
    mean_loss = np.sum(losses) / total_len
    return {'Loss': mean_loss}


def train_model(model: nn.Module,
                vocab: torchtext.vocab.Vocab,
                args: Args,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epoch_callback=None,
                batch_callback=None):
    """Trains the model in-place.

    Args:
        model (Module): Model to be trained.
        vocab (Vocab): Vocabulary.
        args (Args): Command-line arguments.
        train_loader (DataLoader): Loader for training set.
        val_loader (DataLoader): Loader for validation set.
        epoch_callback (Callable): Function that gets called at the end
        of each epoch. Arguments are two state dicts: (best, last)
        batch_callback (Callable): Function that gets called at the end
        of each epoch, and after procissing a batch. Takes one argument:
        a dictionary mapping a name of the metric to its value.
    """
    if batch_callback is None:
        def batch_callback(*_args, **_kwargs):
            pass
    if epoch_callback is None:
        def epoch_callback(*_args, **_kwargs):
            pass
    pad_index = vocab.get_stoi()['<pad>']
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pbar = tqdm.tqdm(total=len(train_loader) * args.epochs)
    best = None
    last = None
    best_val_loss = float('inf')
    postfix = {}
    step = 0
    for epoch in range(args.epochs):
        model.train()
        postfix.update({'Epoch': epoch})
        for ref, trn in train_loader:
            translation = trn[:, :-1]
            ref_pred = model(ref, translation)
            ref_pred = torch.flatten(ref_pred, end_dim=-2)
            trn_pred = model(translation, translation)
            trn_pred = torch.flatten(trn_pred, end_dim=-2)
            target = trn[:, 1:]
            target = torch.flatten(target)
            loss_ref = loss_fn(ref_pred, target)
            loss_trn = loss_fn(trn_pred, target)
            loss = (loss_ref + loss_trn) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
            step += 1
            metrics = {'Epoch': epoch, 'Step': step, 'Training loss': loss.item()}
            batch_callback(metrics)
        last = copy.deepcopy(model.state_dict())
        metrics = evaluate_model(model, val_loader, loss_fn)
        metrics.update({'Epoch': epoch, 'Step': step})
        batch_callback(metrics)
        val_loss = metrics['Loss']
        postfix.update({'Mean validation loss': f'{val_loss:.4f}'})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best = copy.deepcopy(model.state_dict())
        if epoch_callback is not None:
            epoch_callback(best, last)
        pbar.set_postfix(postfix)
    pbar.close()
    return best, last


def main():
    """Main function of the program.
    """
    nltk.download('punkt', quiet=True)
    args = parse_args()
    if args.seed is not None:
        torch.random.manual_seed(args.seed)
        np.random.seed(args.seed)
    dataframe = pd.read_csv(args.dataset_file, sep='\t')
    # dataframe = dataframe.head(1000)  # DEBUG
    dataset = utils.PairDataset(dataframe)
    vocab = get_vocab(dataset,
                      specials=['<unk>', '<pad>', '<bos>', '<eos>'],
                      args=args)
    pad_index = vocab.get_stoi()['<pad>']
    unk_index = vocab.get_stoi()['<unk>']
    bos_index = vocab.get_stoi()['<bos>']
    eos_index = vocab.get_stoi()['<eos>']
    vocab.set_default_index(unk_index)
    torch.save(vocab, 'vocab_rnn.ckpt')
    model = utils.RNN(args.d_model, len(vocab),
                      args.num_layers).to(args.device)
    dummy_model = utils.RNN(args.d_model, len(vocab),
                            args.num_layers).to(args.device)

    def collate_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collates batch into tensors.

        Args:
            batch: Batch of tuples from `PairDataset`

        Returns:
            Tuple[Tensor, Tensor]: Tuple of (reference, translation) tokens.
        """
        refs = []
        trns = []
        for ref, trn, *_ in batch:
            ref_words = tokenize.word_tokenize(ref.lower())
            ref_tokens = vocab(ref_words)
            ref_tokens = ref_tokens[:args.pad_len]
            ref_tokens = ref_tokens + [pad_index] * \
                (args.pad_len - len(ref_tokens))
            refs.append(ref_tokens)
            trn_words = tokenize.word_tokenize(trn.lower())
            trn_tokens = vocab(trn_words)
            trn_tokens = trn_tokens[:args.pad_len - 1]
            trn_tokens = [bos_index] + trn_tokens + [eos_index]
            trn_tokens = trn_tokens + [pad_index] * \
                (args.pad_len + 1 - len(trn_tokens))
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

    def epoch_callback(best, last):
        """Saves the model checkpoints at the end of each epoch.
        """
        dummy_model.load_state_dict(best)
        torch.save(dummy_model, 'best_rnn.ckpt')
        dummy_model.load_state_dict(last)
        torch.save(dummy_model, 'last_rnn.ckpt')
    
    writer = torchboard.writer.SummaryWriter(args.log_dir)

    def batch_callback(metrics: dict[str, float]):
        """Saves metrics into tensorboard logs.
        """
        for metric, value in metrics.items():
            writer.add_scalar(metric, value, global_step=metrics['Step'])
        writer.flush()

    best, last = train_model(model,
                             vocab,
                             args,
                             train_loader,
                             val_loader,
                             batch_callback=batch_callback,
                             epoch_callback=epoch_callback)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    model.load_state_dict(best)
    torch.save(model, 'best_rnn.ckpt')
    best_metrics = evaluate_model(model, test_loader, loss_fn)
    print('Metrics of the best model on test set:')
    for name, value in best_metrics.items():
        print(f'{name}: {value:.4f}')
    model.load_state_dict(last)
    torch.save(model, 'last_rnn.ckpt')
    last_metrics = evaluate_model(model, test_loader, loss_fn)
    print('Metrics of the last model on test set:')
    for name, value in last_metrics.items():
        print(f'{name}: {value:.4f}')


if __name__ == '__main__':
    main()
