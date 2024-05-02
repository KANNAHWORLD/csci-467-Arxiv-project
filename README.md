# CSCI467 Project: Topic Classification of Academic Papers


## Dataset
We used [this dataset](https://huggingface.co/datasets/ccdv/arxiv-classification) from Hugging Face for the arXiv documents and corresponding labels. The data is already divided into train, val, and test sets.

## Usage

### Naive Bayes
Use `naive_bayes.py` to gather all results using Naive Bayes. This script contains code to generate the plot of label occurrences, to run hyperparameter selection, and train & test the Naive Bayes model.

### BERT with Truncation
Use `bert_baseline.py` to gather results using BERT with Truncation. First, the directories `tokenized_original_data'`, `tokenized_binary_data`, `bert_graphs`, and `models` should be created at the same level of the python script. Set `MODEL_NAME` to `bert-base-uncased`, unless using a saved model. All hyperparameters can also be modified within the script using the corresponding global variables. `PREPROCESS_DATA` is used to retokenize the text, and `TRAIN_MODEL` is used to train a model given the hyperparameters.

This code can be used for hyperparameter selection, running LIME, creation of the confusion matrix, and testing the BERT with Truncation model.

For hyperparameter tuning, we used `bert_baseline_lime_args.py` for ease of training multiple models with different hyperparameters. This file has identical functionality to `bert_baseline.py`, but has argument parsing functionalities. Create a file called `input_1`, for example, of the following format:
```
--lr=<learning rate>
--batch=<batch size>
--epochs=<number of epochs>
```
Then, run:
```
python bert_baseline_lime_args.py --input=input_1
```

### BERT + Random
Use `bert_random.py` to run the BERT + Random method. This has almost identical functionalities with `bert_baseline.py`, with many of the same global variables to set. There is an additional hyperparameter (MLP hidden layer size), which can be modified either directly in the file or by using an input file with argument `hidden=<hidden layer size>`.

### Longformer
Use `longformer.py` to run the Longformer method. This has similar functionalities to `bert_baseline.py`. Hyperparameters can only be modified using the input files as described above.
