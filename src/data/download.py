#!python
"""
A script for downloading the raw data.
"""
import argparse
from dataclasses import dataclass
import os
from typing import List
import zipfile

import requests
import tqdm

DOWNLOAD_URL = "https://github.com/skoltech-nlp/detox/" +\
    "releases/download/emnlp2021/filtered_paranmt.zip"
REPOSITORY_NAME = 'text-detoxification'


@dataclass
class ArgsTuple:
    """Wrapper class for program arguments, needed for typing.
    """
    url: str
    quiet: bool
    output_directory: str
    no_guessing: bool
    unzip: bool


def parse_args() -> ArgsTuple:
    """Parses arguments passed to the function and returns them.

    Returns:
        ArgsTuple: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        'download.py', "Downloads the text-detoxification dataset")
    parser.add_argument('--url',
                        type=str,
                        default=DOWNLOAD_URL,
                        help='The url of the .zip file containing the dataset')
    parser.add_argument('--quiet',
                        action='store_true',
                        help='Disable progress bars.')
    parser.add_argument('--output-directory',
                        default='./',
                        help='Where to put the dataset.')
    parser.add_argument('--no-guessing',
                        action='store_true',
                        help='Do not try to find the /data/raw directory, ' +
                        'and download directly into the output directory.')
    parser.add_argument('--no-unzip',
                        action='store_false',
                        help='Do not unzip the downloaded data automatically')
    return parser.parse_args()


def get_path_directories(path: os.PathLike) -> List[str]:
    """Returns the list of names of all directories that
    are either ancestors of `path`, including the name of `path`, if `path` is a folder.

    Args:
        path (os.PathLike): The path to be parsed.

    Returns:
        List[str]: List of names of directories leading up to `path`, including `path` itself.
    """
    ancestor_directories = []
    while True:
        path, directory = os.path.split(path)
        if directory:
            ancestor_directories.append(directory)
        else:
            ancestor_directories.append(path)
            break
    ancestor_directories.reverse()
    return ancestor_directories


def get_subdirectories(path: str) -> List[str]:
    """Returns the list of names of directories that are in the directory,
    specified for `path`

    Args:
        path (str): The path to the target directory.

    Returns:
        List[str]: The list of names of directories that are in the directory.
    """
    _top, subdirectories, _files = next(os.walk(path))
    return subdirectories


def guess_data_directory(target_directory: str) -> str:
    """Tries to find the /data/raw directory, where / is the root of the repository.

    Args:
        target_directory (str): The directory from where script was called.

    Returns:
        str: The absolute path to the `/data/raw` directory
    """
    target_directory = os.path.abspath(target_directory)

    # Check if the target directory exists
    if not os.path.exists(target_directory):
        raise FileNotFoundError(f'Directory {target_directory} not found.')
    if not os.path.isdir(target_directory):
        raise FileExistsError(f'{target_directory} is not a directory.')

    # Attempt to get to the root repository directory
    path_directories = get_path_directories(target_directory)
    if path_directories and path_directories[-2:] == ['src', 'data']:
        # Check if the script is run from the same directory it is in.
        target_directory = os.path.join(
            *path_directories[:-2])  # pylint: disable=no-value-for-parameter
    elif path_directories and path_directories[-1] == 'src':
        # Check if the script is run from the src directory
        target_directory = os.path.join(
            *path_directories[:-1])  # pylint: disable=no-value-for-parameter
    elif REPOSITORY_NAME not in path_directories:
        # If the name of the repository is not in the path, check children directories
        subdirectories = get_subdirectories(target_directory)
        if REPOSITORY_NAME in subdirectories:
            target_directory = os.path.join(target_directory, REPOSITORY_NAME)

    if REPOSITORY_NAME in path_directories:
        # If the target directory is the root of the repository, create ./data/raw/ directory
        # if it doesn't exist, and set it as a target directory
        data_dir = os.path.join(target_directory, 'data')
        raw_data_dir = os.path.join(target_directory, 'data', 'raw')
        if os.path.exists(data_dir) and not os.path.isdir(data_dir):
            raise FileExistsError(f'{data_dir} is not a directory.')
        if os.path.exists(raw_data_dir) and not os.path.isdir(raw_data_dir):
            raise FileExistsError(f'{raw_data_dir} is not a directory.')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        if not os.path.exists(raw_data_dir):
            os.mkdir(raw_data_dir)
        target_directory = raw_data_dir

    return target_directory


def download_file(target_directory: str, url: str, quiet: bool) -> str:
    """Downloads the `filtered.zip` file to the target_directory.

    Args:
        target_directory (str): The directory to download the file to.
        url (str): The url of the file.
        quiet (bool): Whether to hide the progress bar or not.

    Returns:
        str: The name of the downloaded file.
    """
    filename = os.path.join(target_directory, 'filtered.zip')
    with requests.get(url, stream=True,
                      timeout=10) as request, open(filename, 'wb') as file:
        request.raise_for_status()
        total_size = int(request.headers.get('Content-Length', 0))
        chunk_size = 16 * 2**10
        if total_size == 0:
            bar_format = '{elapsed} {rate_noinv_fmt}'
        else:
            bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n:.3f}{unit}/{total:.3f}{unit} ' +\
                '[{elapsed}<{remaining}, {rate_noinv_fmt}]'
        with tqdm.tqdm(total=total_size / 2**20,
                       disable=quiet,
                       unit='MB',
                       desc='Downloading',
                       bar_format=bar_format) as pbar:
            for chunk in request.iter_content(chunk_size):
                file.write(chunk)
                pbar.update(len(chunk) / 2**20)
    return filename


def unzip(file_path: str, target_directory: str):
    """Unzips the downloaded file.

    Args:
        file_path (str): The path to the downloaded .zip file.
        target_directory (str): The directory to unzip the file to.
    """
    with zipfile.ZipFile(file_path) as zfile:
        for file in zfile.namelist():
            zfile.extract(file, target_directory)


def main():
    """The main function of the program.
    """
    args = parse_args()
    if args.no_guessing:
        target_directory = args.output_directory
    else:
        target_directory = guess_data_directory(args.output_directory)
    if not args.quiet:
        print(f'Downloading into directory {target_directory}.')
    downloaded_file = download_file(target_directory, args.url, args.quiet)
    if args.unzip:
        unzip(downloaded_file, target_directory)
        os.remove(downloaded_file)


if __name__ == '__main__':
    main()
