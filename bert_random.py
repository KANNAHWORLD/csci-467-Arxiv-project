from datasets import load_dataset, load_from_disk, Dataset
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import spacy
import random
import os, argparse, sys 
import shlex
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from torch.nn.functional import softmax
from transformers.tokenization_utils_base import BatchEncoding

## Note: expects directories called 'tokenized_random_data', 'tokenized_random_test_data', 'models'
# AND: 'conf_matrix_output'
CONF_MATRIX_OUTPUT_DIR = 'conf_matrix_output/'
# to exist and be in the same directory as this script

# if no MODEL_NAME provided, defaults to creating model name assuming it is saved
# Model and MLP name automatically calculated using hyperparams, no need to fill in during test mode
MODEL_NAME = 'bert-base-uncased' 
MLP_NAME = '' 
DATASET_NAME = 'ccdv/arxiv-classification'
# If using a saved MODEL_NAME or MLP_NAME above, still make sure to update these to match:
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
MLP_HIDDEN_SIZE = 1024
TRAIN_MODEL = False
PREPROCESS_DATA = False

# If true, use val set (else, use test set)
USE_VAL_SET = True

# Error analysis
ERROR_ANALYSIS = True
LIME_SAMPLES = 100 # used only if above is True

SET_SEEDS = True # makes output deterministic, using following seed
SEED = 42

# Use test flag to only use 3 examples for each split
TEST = True

ORIGINAL_LABELS = ['math.AC', 'cs.CV', 'cs.AI', 'cs.SY', 'math.GR', 'cs.DS', 'cs.CE', 'cs.PL', 'cs.IT', 'cs.NE', 'math.ST']


# Usage: salloc --time=2:00:00 --cpus-per-task=8 --mem=32G --gres=gpu:1 --partition=gpu --account=robinjia_1265


def fix_labels(instance):
    classConversion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    instance['labels'] = classConversion[instance['label']]
    return instance

# Tokenize random sentences
def tokenize_batch_random(batch):
    # Create list of sentences using spacy
    nlp = spacy.load("en_core_web_sm")
    # Set max length limit (max document size is 2553768)
    nlp.max_length = 2750000

    # Keep track of outputs
    output_texts = []

    # Save each row to list
    examples = batch['text']
    if isinstance(examples, str):
        examples = [examples]

    # Iterate
    for example in examples:
        doc = nlp(example)
        sentences = list(doc.sents)

        # Remove the first 512 tokens, since already present in other half of tokenized data
        removed_tokens = 0
        num_removed_sentences = 0
        while removed_tokens < 512 and num_removed_sentences < len(sentences):
            num_tokens = len(tokenizer.tokenize(str(sentences[num_removed_sentences])))
            removed_tokens += num_tokens 
            num_removed_sentences += 1
        sentences = sentences[num_removed_sentences:]

        # Select sentence indexes randomly until 512 tokens reached
        sent_idxs = list(range(len(sentences))) # all indexes
        selected_idxs = []
        num_tokens_so_far = 0
        while num_tokens_so_far <= (512-2) and sent_idxs:
            # Choose an index and remove so it's not picked again
            idx = random.choice(sent_idxs)
            sent_idxs.remove(idx)
            # Find sentence length
            curr_sentence = str(sentences[idx])
            num_tokens_so_far += len(tokenizer.tokenize(curr_sentence))
            selected_idxs.append(idx)
        
        # Order the indexes, and create one string
        output_text = ''
        reordered_idxs = sorted(selected_idxs)
        for idx in reordered_idxs:
            output_text += str(sentences[idx]) + ' '
        output_texts.append(output_text)
    return tokenizer(output_texts, max_length=512, truncation=True, return_tensors='pt')

def tokenize_batch(batch):
    return tokenizer(batch['text'], max_length=512, truncation=True, return_tensors='pt')

def load_process_data_from_hub():
    split = '[:3]' if TEST else ''
    train_data_fixed = load_dataset(DATASET_NAME, "no_ref", split=f'train{split}')
    test_data_fixed = load_dataset(DATASET_NAME, "no_ref", split=f'test{split}')
    val_data_fixed = load_dataset(DATASET_NAME, "no_ref", split=f'validation{split}')

    train_data_fixed = train_data_fixed.map(fix_labels)
    test_data_fixed = test_data_fixed.map(fix_labels)
    val_data_fixed = val_data_fixed.map(fix_labels)


    # Preprocessing
    train_data = train_data_fixed.map(tokenize_batch, batched=False)
    train_data_random = train_data_fixed.map(tokenize_batch_random, batched=False)

    test_data = test_data_fixed.map(tokenize_batch)
    test_data_random = test_data_fixed.map(tokenize_batch_random)
    
    val_data = val_data_fixed.map(tokenize_batch)
    val_data_random = val_data_fixed.map(tokenize_batch_random)

    ## Save the tokenized datasets
    test_str = 'test_' if TEST else ''
    train_data.save_to_disk(f'./tokenized_random_{test_str}data/train')
    test_data.save_to_disk(f'./tokenized_random_{test_str}data/test')
    val_data.save_to_disk(f'./tokenized_random_{test_str}data/val')
    train_data_random.save_to_disk(f'./tokenized_random_{test_str}data/train_random')
    test_data_random.save_to_disk(f'./tokenized_random_{test_str}data/test_random')
    val_data_random.save_to_disk(f'./tokenized_random_{test_str}data/val_random')

def train(train_loader, test_loader, val_loader, model, device):

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    p_bar = tqdm(range(num_training_steps))
    model.train()

    # Define 2-layer MLP
    bert_output_size = 768 * 2
    num_outputs = len(ORIGINAL_LABELS)
    mlp = nn.Sequential(
        nn.Linear(bert_output_size, MLP_HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(MLP_HIDDEN_SIZE, num_outputs)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            batch = {key: batch[key].to(device).squeeze() for key in batch}
            optimizer.zero_grad()
            # Model output for first 512 tokens, random 512 tokens
            _, truncated_outputs = model(input_ids=batch['input_ids'][:, 0, :], attention_mask=batch['attention_mask'][:, 0, :], return_dict=False)
            _, random_outputs = model(input_ids=batch['input_ids'][:, 1, :], attention_mask=batch['attention_mask'][:, 1, :], return_dict=False)
            # Concatenate
            concatenated_outputs = torch.cat((truncated_outputs, random_outputs), dim=1)
            # Take average
            # averaged_outputs = torch.mean(concatenated_outputs, dim=1)
            # This is the input to the MLP
            mlp_output = mlp(concatenated_outputs)
            # Calculate loss
            loss = criterion(mlp_output, batch['labels'])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            p_bar.update(1)
            p_bar.set_postfix({'loss': loss.item()})
    
    test_str = 'test_' if TEST else ''
    model.save_pretrained(f'./models/bert_{test_str}random_epochs_{NUM_EPOCHS}_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_hidden_{MLP_HIDDEN_SIZE}')
    torch.save(mlp.state_dict(), f'./models/mlp_{test_str}random_epochs_{NUM_EPOCHS}_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_hidden_{MLP_HIDDEN_SIZE}.pt')

    # Return model and mlp
    return model, mlp


# Helper functino to pad to a tensor to reach the expected size
def pad_tensor(tensor, expected_size):
    if tensor.size(1) < expected_size:
        padding = torch.zeros((1, expected_size - tensor.size(1)), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=1)
    return tensor


# Helper function that concatenates tokenized data with tokenized random data
def concatenate_helper(data, data_random, overall_label=None):
    # Initialize an empty dictionary to store concatenated representations
    concatenated_examples = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }

    # print(f"Data Keys: {data.keys()}")

    if isinstance(data, BatchEncoding):
        # Assume that label provided

        def transform_to_dataset(batch):
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask

            dataset_dict = {'input_ids': [], 'attention_mask': [], 'labels': []}
            for i in range(len(input_ids)):
                dataset_dict['input_ids'].append([input_ids[i]])
                dataset_dict['attention_mask'].append([attention_mask[i]])
                dataset_dict['labels'].append(overall_label)
            return Dataset.from_dict(dataset_dict)
        
        data = transform_to_dataset(data)
        data_random = transform_to_dataset(data_random)

    # Iterate
    for example, example_random in tqdm(zip(data, data_random), total=len(data)):

        # Convert to tensors to concatenate
        input_ids_tensor = torch.tensor(example['input_ids'])
        input_ids_random_tensor = torch.tensor(example_random['input_ids'])
        attention_mask_tensor = torch.tensor(example['attention_mask'])
        attention_mask_random_tensor = torch.tensor(example_random['attention_mask'])
        
        # if input_ids_random_tensor.shape[1] != 512:
        #     print(f'Input Ids Random: {input_ids_random_tensor.shape}')
        #     print(f'Attention Mask Random: {attention_mask_random_tensor.shape}\n')

        # Concatenate representations
        # Pad to random if needed
        input_ids_random_tensor = pad_tensor(input_ids_random_tensor, 512)
        attention_mask_random_tensor = pad_tensor(attention_mask_random_tensor, 512)
        concatenated_examples['input_ids'].append(torch.cat([input_ids_tensor, input_ids_random_tensor]))
        concatenated_examples['attention_mask'].append(torch.cat([attention_mask_tensor, attention_mask_random_tensor]))
        # Add label
        concatenated_examples['labels'].append(example['labels'])

    # Convert lists to tensors
    concatenated_examples['input_ids'] = torch.stack(concatenated_examples['input_ids'])
    concatenated_examples['attention_mask'] = torch.stack(concatenated_examples['attention_mask'])

    # Create dataset
    final_data = Dataset.from_dict(concatenated_examples)

    # Set format
    final_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    return final_data


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    global NUM_EPOCHS
    global BATCH_SIZE
    global LEARNING_RATE
    global MLP_HIDDEN_SIZE
    global MODEL_NAME
    global MLP_NAME
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='Input file name')
    file_args = parser.parse_args()
    file = file_args.input

    def parse_file(filename):
        with open(filename, 'r') as f:
            args = shlex.split(f.read())
        return args

    parser = argparse.ArgumentParser()
    ## parse the arguments for ep, weighted, atoms, sparsity, seed, epochs
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of epochs for training')
    parser.add_argument('--hidden', type=int, default=MLP_HIDDEN_SIZE, help='MLP hidden layer size')

    if (file is not None) and len(file) > 0:
        args = parser.parse_args(parse_file(file))
        NUM_EPOCHS = args.epochs
        BATCH_SIZE = args.batch
        LEARNING_RATE = args.lr
        MLP_HIDDEN_SIZE = args.hidden

    # Create the BERT and MLP name (either for saving or loading)
    test_str = 'test_' if TEST else ''
    if not TRAIN_MODEL:
        MODEL_NAME = f'./models/bert_{test_str}random_epochs_{NUM_EPOCHS}_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_hidden_{MLP_HIDDEN_SIZE}'
        MLP_NAME = f'./models/mlp_{test_str}random_epochs_{NUM_EPOCHS}_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_hidden_{MLP_HIDDEN_SIZE}.pt'
        print(f'Using BERT model name: {MODEL_NAME}')
        print(f'Using MLP model name: {MLP_NAME}')


# Helper function to display / save confusion matrix
def save_confusion_matrix(labels, preds):
    conf_matrix = confusion_matrix(labels, preds)
    print('\nConfusion Matrix:')
    print(conf_matrix)

    # Save confusion matrix
    conf_matrix_filename = f'conf_matrix_{eval_str}_epochs_{NUM_EPOCHS}_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}_hidden_{MLP_HIDDEN_SIZE}.png'
    conf_matrix_filename = CONF_MATRIX_OUTPUT_DIR + conf_matrix_filename

    plt.figure(figsize=(9, 7))
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=ORIGINAL_LABELS)
    display.plot(values_format='d')
    tmp_str = 'Val' if USE_VAL_SET else 'Test'
    plt.title(f'Confusion Matrix On {tmp_str} Set')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(conf_matrix_filename)
    print(f'\nSaved to filename: {conf_matrix_filename}')


# Global flag for current label
curr_label = None

def predict_proba(texts):
    batch_texts = {'text': texts}
    tokenized = tokenizer(texts, max_length=512, truncation=True, return_tensors='pt', padding='max_length')
    tokenized_random = tokenize_batch_random(batch_texts)
    concatenated = concatenate_helper(tokenized, tokenized_random, curr_label)
    with torch.no_grad():
        # inputs = {name: tensor.to(device) for name, tensor in concatenated.items()}

        input_ids = concatenated['input_ids'].to(device)
        attention_mask = concatenated['attention_mask'].to(device)

        
        # Model output for first 512 tokens, random 512 tokens
        _, truncated_outputs = model(input_ids=input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :], return_dict=False)
        _, random_outputs = model(input_ids=input_ids[:, 1, :], attention_mask=attention_mask[:, 1, :], return_dict=False)

        # Concatenate
        concatenated_output = torch.cat((truncated_outputs, random_outputs), dim=1)

        # MLP output
        mlp_output = mlp(concatenated_output)

        probs = softmax(mlp_output, dim=-1)

    return probs.squeeze().detach().cpu().numpy()


def display_errors(val_preds, val_labels):
    '''
    Print out original labels and text for incorrect predictions, one at a time
    Ask for user input to continue to next error
    '''
    global curr_label

    errors = []
    for i, pred in enumerate(val_preds):
        if pred != val_labels[i]:
            errors.append(i)
    original_val_ds = load_dataset(DATASET_NAME, 'no_ref', split='validation')
    tokenized_val_ds = load_from_disk('./tokenized_random_data/val')

    explainer = LimeTextExplainer(class_names=ORIGINAL_LABELS)

    for i in errors:
        print("Original label: ", ORIGINAL_LABELS[original_val_ds[i]['label']])
        print("Predicted label: ", ORIGINAL_LABELS[val_preds[i]])
        print("Correct label: ", ORIGINAL_LABELS[val_labels[i]])
        print("Text: ", tokenizer.decode(tokenized_val_ds[i]['input_ids'][0]))
        print("Text from DS: ", original_val_ds[i]['text'][:1000])

        curr_label = val_labels[i]
        explanation = explainer.explain_instance(tokenizer.decode(tokenized_val_ds[i]['input_ids'][0]), predict_proba, num_features=10, num_samples=LIME_SAMPLES)

        fig = explanation.as_pyplot_figure()
        plt.savefig(f'explanations/explanation_{i}.png')

        input("Press Enter to continue...")


if __name__ == '__main__':
    # Parse arguments
    parse_arguments()


    if SET_SEEDS:
        set_random_seeds(SEED)

    model = AutoModel.from_pretrained(MODEL_NAME)
    global tokenizer 
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #change to torch.device('mps') if running on mac
    model = model.to(device)

    mlp = None

    if PREPROCESS_DATA:
        load_process_data_from_hub()

    ## Load the tokenized datasets from disk
    test_str = 'test_' if TEST else ''
    train_data = load_from_disk(f'./tokenized_random_{test_str}data/train')
    test_data = load_from_disk(f'./tokenized_random_{test_str}data/test')
    val_data = load_from_disk(f'./tokenized_random_{test_str}data/val')
    train_data_random = load_from_disk(f'./tokenized_random_{test_str}data/train_random')
    test_data_random = load_from_disk(f'./tokenized_random_{test_str}data/test_random')
    val_data_random = load_from_disk(f'./tokenized_random_{test_str}data/val_random')

    full_train_data = concatenate_helper(train_data, train_data_random)
    full_test_data = concatenate_helper(test_data, test_data_random)
    full_val_data = concatenate_helper(val_data, val_data_random)
    
    train_loader = DataLoader(full_train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(full_test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    val_loader = DataLoader(full_val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    
    if TRAIN_MODEL:
        model, mlp = train(train_loader, test_loader, val_loader, model, device)
    else:\
        # load in MLP
        bert_output_size = 768 * 2
        num_outputs = len(ORIGINAL_LABELS)
        mlp = nn.Sequential(
        nn.Linear(bert_output_size, MLP_HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(MLP_HIDDEN_SIZE, num_outputs)
        ).to(device)
        mlp.load_state_dict(torch.load(MLP_NAME))
    

    model.eval()
    mlp.eval()
    val_labels = []
    val_preds = []

    # evaluate on chosen set
    eval_loader = val_loader if USE_VAL_SET else test_loader
    for batch in tqdm(eval_loader):
        batch = {key: batch[key].to(device).squeeze() for key in batch}
        with torch.no_grad():
            # Model output for first 512 tokens, random 512 tokens
            _, truncated_outputs = model(input_ids=batch['input_ids'][:, 0, :], attention_mask=batch['attention_mask'][:, 0, :], return_dict=False)
            _, random_outputs = model(input_ids=batch['input_ids'][:, 1, :], attention_mask=batch['attention_mask'][:, 1, :], return_dict=False)
            # Concatenate
            concatenated_output = torch.cat((truncated_outputs, random_outputs), dim=1)
            # Average pooling
            #averaged_output = torch.mean(concatenated_output, dim=1)
            # Pass averaged output through MLP
            mlp_output = mlp(concatenated_output)
            # Get predicted labels
            _, predicted_labels = torch.max(mlp_output, 1)
        val_labels.extend(batch['labels'].cpu().numpy().tolist())
        val_preds.extend(predicted_labels.cpu().numpy().tolist())

    accuracy = metrics.accuracy_score(val_labels, val_preds)
    precision = metrics.precision_score(val_labels, val_preds, average='macro')
    recall = metrics.recall_score(val_labels, val_preds, average='macro')
    f1 = metrics.f1_score(val_labels, val_preds, average='macro')

    print('Hyperparameters: ')
    print(f'Epochs: {NUM_EPOCHS}')
    print(f'Learning Rate: {LEARNING_RATE}')
    print(f'Batch Size: {BATCH_SIZE}')
    print(f'Hidden Layer Size: {MLP_HIDDEN_SIZE}\n')

    eval_str = 'val' if USE_VAL_SET else 'test'
    print(f'Evaluating on {eval_str} set:')
    print("Num samples", len(eval_loader) * BATCH_SIZE)
    # print("Num samples", len(eval_labels))
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    # Create and print confusion matrix
    # save_confusion_matrix(val_labels, val_preds)

    # Do error analysis
    if ERROR_ANALYSIS:
        display_errors(val_preds, val_labels)
