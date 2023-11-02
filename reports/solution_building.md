# Solution-Building Report

## Dictionary

The baseline model is simply a lookup table that maps toxic words to non-toxic synonyms. This arguably is the simplest model for detoxifying sentences, and also the least effective. The table can be constructed either by hand-picking toxic words, but this requires human effort. Dale et al. [1] suggest training logistic Bag-of-Words sentence-level classifier on the toxicity dataset, which is the implementation chosen. The source code for baseline is available at `src/models/train_baseline.py` and `src/models/predict_baseline.py`. The disadvantages of the model are clear: it is unable to take the context of the word into account, and depending on the implementation might fail when the form of word is changed (e.g. table/tables or milk/milking).

## Recurrent Neural Network

Recurrent neural networks were developed, in part, for natural language processing, and they can be viewed as a light-weight alternative to modern NLM methods, such as transformers. For this assignment, Long-Short Term Memory network was chosen as it is among the most popular RNN architectures. The source code can be viewed at `src/models/train_rnn.py` and `src/models/predict_rnn.py`. LSTM performs better than the dictionary model, but training a RNN model requires time and computational resources, and the model is rather weak. As such, while it lowers the toxicity of the text, it completely loses the original meaning of the text. Example: A quick brown fox jumps over the lazy dog/With the other, the other people are the one who's got a hell of a lot of money? This may also be explained by the small vocabulary size which is necessary due to memory constraint.

## Finetuning GPT-2

If training the model from scratch is too expensive, then finetuning a pre-trained model is the natural next choice. Instead of a paraphraser model, such as T5, GPT-2 model was chosen. This is due to multiple reasons, including the fact that causal NLP models are easier to train, as they do not require a specifically prepared dataset, but rather just a huge collection of texts in a language. This approach proved to be more effective both in the quality of detoxified texts and the amount of time needed for training. Because of learned embeddings, only a part of the dataset is needed for fine-tuning the model, reducing the training time to less than an hour on commercially available GPUs.

## Results

After building understanding of the problem through building the baseline and experimenting with NLP model architectures, the best approach proved to be fine-tuning existing models.

## References

[1] David Dale, Daryna Dementieva, et al. "Text Detoxification using Large Pre-trained Neural Models," in CoRR, vol. abs/2109.08914, 2021.
