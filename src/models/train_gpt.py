import argparse

import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm


class Args:
    """Command line arguments.
    """
    dataset_file: str
    learning_rate: float
    epochs: int
    device: str
    model_dir: str
    seed: int
    log_dir: str
    max_pairs: int
    warmup_steps: int
    quiet: bool
    checkpoint: str


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
                        default=3e-5,
                        help='Learning rate of the model.')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help='Number of training epochs.')
    parser.add_argument('--device',
                        choices=['cuda', 'cpu'],
                        default='cpu',
                        help='The device the model is trained on')
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
    parser.add_argument('--max-pairs',
                        type=int,
                        dest='max_pairs',
                        help='Maximum number of reference-translation pairs '
                        'to finetune on.',
                        default=10000)
    parser.add_argument('--warmup-steps',
                        type=int,
                        dest='warmup_steps',
                        help='The amount of warmup steps during which the '
                        'learning rate increases',
                        default=500)
    parser.add_argument('--quiet',
                        dest='quiet',
                        action='store_true',
                        help='Whether to hide progress bars or not.')
    parser.add_argument('--model-dir',
                        dest='model_dir',
                        default='./gpt/',
                        help='The name of the directory the weights will be '
                        'saved into. Defaults to ./gpt/')
    parser.add_argument('--checkpoint-dir',
                        dest='checkpoint',
                        default=None,
                        help='The name of the directory the weights will be '
                        'loaded from.')
    args = parser.parse_args()
    return args

class ToxicDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = []
        separate_token = '<|endoftext|>'
        eos_token = '<|endoftext|>'
        for _, (ref, trn, ref_tox, trn_tox) in dataframe[['reference', 'translation', 'ref_tox', 'trn_tox']].iterrows():
            if ref_tox < trn_tox:
                ref, trn = trn, ref
            self.data.append(f'{ref}{separate_token}{trn}{eos_token}')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def main():
    args = parse_args()
    dataframe = pd.read_csv(args.dataset_file, index_col=0, sep='\t')
    if len(dataframe) > args.max_pairs:
        dataframe = dataframe.head(args.max_pairs)
    model_path = 'gpt2'
    if args.checkpoint is not None:
        model_path = args.checkpoint
    model = transformers.AutoModelWithLMHead.from_pretrained(
        model_path).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
    optimizer = transformers.AdamW(model.parameters(), lr=args.learning_rate)
    training_steps = len(dataframe) * args.epochs - args.warmup_steps
    sheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, args.warmup_steps, training_steps)
    train_dataset = ToxicDataset(dataframe)
    with tqdm.tqdm(total=len(train_dataset) * args.epochs, disable=args.quiet) as pbar:
        for _epoch in range(args.epochs):
            for text in train_dataset:
                tokens = tokenizer.encode(text)
                tokens_tensor = torch.tensor(tokens).to(args.device)
                loss = model(tokens_tensor, labels=tokens_tensor)['loss']
                optimizer.zero_grad()
                model.zero_grad()
                loss.backward()
                optimizer.step()
                sheduler.step()
                pbar.update()
    model.save_pretrained(args.model_dir)

if __name__ == '__main__':
    main()

# TODO Add validation and tensorboard logs
# TODO Fix deprecation warnings