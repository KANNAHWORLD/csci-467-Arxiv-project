from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn import metrics
import spacy
import random

## Note: expects directories called 'tokenized_random_data', 'tokenized_random_test_data', 'models'
# to exist and be in the same directory as this script

MODEL_NAME = 'bert-base-uncased' # something like ./models/bert_epochs_1_lr_1e-05_batch_16 when loading trained model
DATASET_NAME = 'ccdv/arxiv-classification'
NUM_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
TRAIN_MODEL = True
PREPROCESS_DATA = True

SEED = 42

# Use test flag to only use 3 examples for each split
TEST = True

ORIGINAL_LABELS = ['math.AC', 'cs.CV', 'cs.AI', 'cs.SY', 'math.GR', 'cs.DS', 'cs.CE', 'cs.PL', 'cs.IT', 'cs.NE', 'math.ST']

def fix_labels(instance):
    classConversion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    instance['labels'] = classConversion[instance['label']]
    return instance

# Tokenize random sentences
def tokenize_batch_random(batch):
    # Create list of sentences using spacy
    nlp = spacy.load("en_core_web_sm")

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

    print(train_data[0]['text'][:1000])
    print('\n\n\n------RANDOM---------\n\n\n')
    print(train_data_random[0]['text'][:1000])
    exit()

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

    for epoch in range(NUM_EPOCHS):
        for batch in train_loader:
            batch = {key: batch[key].to(device).squeeze() for key in batch}
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            p_bar.update(1)
            p_bar.set_postfix({'loss': loss})
    
    model.save_pretrained(f'./models/bert_random_epochs_{NUM_EPOCHS}_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}')

    return model

if __name__ == '__main__':
    random.seed(SEED)

    model = AutoModel.from_pretrained(MODEL_NAME)
    global tokenizer 
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #change to torch.device('mps') if running on mac
    model = model.to(device)

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

    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    if TRAIN_MODEL:
        model = train(train_loader, test_loader, val_loader, model, device)
    
    # evaluate on validation set
    model.eval()
    val_labels = []
    val_preds = []

    for batch in tqdm(test_loader):
        batch = {key: batch[key].to(device).squeeze() for key in batch}
        with torch.no_grad():
            outputs = model(**batch)
        val_labels.extend(batch['labels'].cpu().numpy().tolist())
        val_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist())

    accuracy = metrics.accuracy_score(val_labels, val_preds)
    precision = metrics.precision_score(val_labels, val_preds, average='macro')
    recall = metrics.recall_score(val_labels, val_preds, average='macro')
    f1 = metrics.f1_score(val_labels, val_preds, average='macro')

    print("Num samples", len(val_loader) * BATCH_SIZE)
    print("Num samples", len(val_labels))
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)