# LSTM-Language-Model

A word-level language model built with an LSTM in PyTorch. Given a sequence
of words, the model predicts the most likely next word and can generate
short pieces of text.

## Overview
The model is trained on a short story ("The Last Train Journey"). Every
sentence is expanded into many training samples so the network learns to
predict the next word from the words before it.

## Pipeline
1. **Tokenization:** text is lowercased and split into words with NLTK
2. **Vocabulary:** each unique word gets an index (`<unk>` for unknown words)
3. **Sequence generation:** one sentence produces several samples
   `[w1, w2, w3]` becomes `[w1, w2]` and `[w1, w2, w3]` (last word = target)
4. **Padding:** sequences are left-padded with zeros to equal length
5. **Dataset and DataLoader:** custom `Dataset`, batch size 32, shuffled
6. **Training:** CrossEntropyLoss with the Adam optimizer
7. **Prediction:** the word with the highest score is chosen

## Model Architecture
Input word indices -> Embedding (100) -> LSTM (150) -> Linear (vocab_size)

## Tech Stack
Python, PyTorch, NLTK, NumPy, Google Colab

## How to Run
1. Clone the repository
   git clone https://github.com/tanzimahamed/LSTM-Language-Model.git
2. Install dependencies
   pip install -r requirements.txt
3. Open the notebook in Jupyter or Google Colab
4. Run all cells

## Example
Input: "Daniel checked the ticket"
Output: (এখানে তোমার model-এর আসল output বসাও)

## What I Learned
- How language modeling works (predicting the next word)
- Tokenization, vocabulary building and indexing
- Why padding is needed, and why left-padding suits an LSTM
- Building a custom Dataset and DataLoader in PyTorch
- Using the final hidden state of an LSTM for classification

## Limitations
- Very small dataset, so the model mostly memorizes the story
- `<pad>` and `<unk>` currently share index 0
- No validation split

## Future Work
- Separate `<pad>` and `<unk>` tokens
- Train on a larger corpus
- Add accuracy, perplexity and a validation set
- Compare with GRU and a character-level model
- Bangla next-word prediction

## Author
Tanzim, Phitron AI/ML Batch 2

## License
MIT
