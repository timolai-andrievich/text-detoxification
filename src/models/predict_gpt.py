"""A script for transforming the sentences using the RNN model.
"""
import argparse
from typing import TypedDict, List
import sys

import nltk
from nltk import tokenize
import torch
import transformers


class Args(TypedDict):
    """Arguments namespace. Needed for typing hints."""
    input_file: str
    output_file: str
    device: str
    checkpoint_path: str


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


def main():
    """Main function of the program.
    """
    args = parse_args()
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.checkpoint_path).to(args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')

    def detoxify(text: str, max_len=100):
        tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
        initial_len = len(tokens)
        for _ in range(max_len):
            tokens_tensor = torch.tensor(tokens).to(args.device)
            logits = model(tokens_tensor)[0]
            next_token = logits[-1, :].argmax()
            if next_token == tokenizer.eos_token_id:
                break
            tokens.append(next_token)
        detoxified_tokens = tokens[initial_len:]
        return tokenizer.decode(detoxified_tokens)

    if args.input_file != 'stdin':
        fin = open(args.input_file, 'r', encoding='utf-8')
    else:
        fin = sys.stdin
    if args.output_file != 'stdout':
        fout = open(args.output_file, 'w', encoding='utf-8')
    else:
        fout = sys.stdout
    if fin == sys.stdin:
        print('Reading stdin...', file=sys.stderr)
    try:
        for line in fin:
            line = line.lower()
            sentences = tokenize.sent_tokenize(line)
            result = []
            for sentence in sentences:
                transformed_text = detoxify(sentence)
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
