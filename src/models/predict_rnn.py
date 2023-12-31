"""A script for transforming the sentences using the RNN model.
"""
import argparse
from typing import TypedDict, List
import sys

import nltk
from nltk import tokenize
import torch
import torchtext

import utils


class Args(TypedDict):
    """Arguments namespace. Needed for typing hints."""
    input_file: str
    output_file: str
    checkpoint_path: str
    vocab_path: str


def parse_args() -> Args:
    """Parses command line arguments.

    Returns:
        Args: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        dest='input_file',
                        required=True,
                        type=str,
                        help='The input file. If "stdin", '
                        'lines are read from standard input.')
    parser.add_argument('--output',
                        dest='output_file',
                        required=True,
                        type=str,
                        help='The output file. If "stdout", '
                        'lines are written to standard output.')
    parser.add_argument('--vocab',
                        dest='vocab_path',
                        required=True,
                        type=str,
                        help='Vocabulary checkpoint.')
    parser.add_argument('--checkpoint',
                        dest='checkpoint_path',
                        required=True,
                        type=str,
                        help='Model checkpoint.')
    parser.add_argument('--device',
                        type=str,
                        choices=['cuda', 'cpu'],
                        dest='device',
                        default='cpu',
                        help='Device to run the model on. Defaults to cpu.')
    return parser.parse_args()


def autocapitalize(words: List[str]) -> List[str]:
    """Applies capitalization to words.

    Args:
        words (List[str]): Words to be capitalized.

    Returns:
        List[str]: Words with capitalization applied.
    """
    if not words:
        return []
    tagged_words = nltk.pos_tag(words)
    tags = ['NNP', 'NNPS']
    capitalized = [
        word.capitalize() if tag in tags else word
        for word, tag in tagged_words
    ]
    capitalized[0] = capitalized[0].capitalize()
    return capitalized


def main():
    """Main function of the program.
    """
    args = parse_args()
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    rnn: utils.RNN = torch.load(args.checkpoint_path, map_location=args.device)
    vocab: torchtext.vocab.Vocab = torch.load(args.vocab_path)
    model = utils.RNNModel(vocab, rnn, device=args.device)
    if args.input_file != 'stdin':
        fin = open(args.input_file, 'r', encoding='utf-8')
    else:
        fin = sys.stdin
    if args.output_file != 'stdout':
        fout = open(args.output_file, 'w', encoding='utf-8')
    else:
        fout = sys.stdout
    tokenizer = tokenize.TreebankWordTokenizer()
    detokenizer = tokenize.TreebankWordDetokenizer()
    if fin == sys.stdin:
        print('Reading stdin...', file=sys.stderr)
    try:
        for line in fin:
            line = line.lower()
            sentences = tokenize.sent_tokenize(line)
            result = []
            for sentence in sentences:
                words = tokenizer.tokenize(sentence)
                transformed_words = model.transform(words)
                transformed_words = autocapitalize(transformed_words)
                transformed_text = detokenizer.detokenize(transformed_words)
                result.append(transformed_text)
            fout.write(' '.join(result) + '\n')
            fout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        fin.close()
        fout.close()


if __name__ == '__main__':
    main()
