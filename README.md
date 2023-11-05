# Text Detoxification

Timolai Andrievich, t.andrievich@innopolis.university, group DS01-BS21.

---

A repository for the first assignment of the practical machine learning course.

## Requirements

To install the required pip libraries, run

```bash
python -m pip install -r requirements.txt
```

## How to Train

### Using Console Scripts

First, download the dataset file:

```bash
python src/data/download.py --output-directory ./data/raw/
```

This will download and unzip the dataset file.

```bash
python src/models/train_gpt.py --dataset ./data/raw/ --device cuda
```

For both of these scripts, usage instructions can be viewed using the `-h` or `--help` arguments.

### Using IPython Notebooks

Just run `/notebooks/2.0-final-solution.ipynb`.

## How to Run

Acquire weights:

- Finetune the model from scratch, or
- Download and unzip pre-trained weights from [Google Disc](https://drive.google.com/file/d/1-uCCFQ15zSNe52kKvJZtWfS-UpTDlKex/view?usp=sharing).

Run

```bash
python src/models/predict_gpt.py --input <INPUT_FILE> --output <OUTPUT_FILE> --checkpoint <WEIGHTS_DIR>
```

Where `WEIGHTS_DIR` is a directory containing `config.json`, `generated_config.json`, and `pytorch_model.bin`. To run the prediction script in a sort of "interactive" mode, you can specify `stdin` as input file, and `stdout` as output file.
