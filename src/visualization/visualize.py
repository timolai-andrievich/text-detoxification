import argparse
import os
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Args:
    dataset: str
    out_dir: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to the .tsv file containing the dataset.',
        type=str,
        dest='dataset',
    )
    parser.add_argument(
        '--out-dir',
        required=False,
        default='./figures/',
        type=str,
        dest='out_dir',
        help='Output directory for the figures. Defaults to ./figures/'
    )
    args = parser.parse_args()
    return args


def visualize_dataset(dataset_path: str, out_dir: str):
    dataframe = pd.read_csv(dataset_path, sep='\t', index_col=0)
    dataframe['ref_filtered'] = np.where(
        dataframe['ref_tox'] >= dataframe['trn_tox'], dataframe['ref_tox'],
        dataframe['trn_tox'])
    dataframe['trn_filtered'] = np.where(
        dataframe['ref_tox'] >= dataframe['trn_tox'], dataframe['trn_tox'],
        dataframe['ref_tox'])

    sns.histplot(data=dataframe, x='ref_filtered', bins=20)
    plt.title('Reference toxicity')
    plt.savefig(os.path.join(out_dir, 'ref_tox.png'))
    plt.close()
    plt.clf()

    sns.histplot(data=dataframe, x='trn_filtered', bins=20)
    plt.title('Translation toxicity')
    plt.savefig(os.path.join(out_dir, 'trn_tox.png'))
    plt.close()
    plt.clf()


def main():
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    visualize_dataset(args.dataset, args.out_dir)


if __name__ == '__main__':
    main()
