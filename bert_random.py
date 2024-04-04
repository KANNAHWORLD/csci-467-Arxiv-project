from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn import metrics
import spacy

## Note: expects directories called 'tokenized_random_data', 'tokenized_random_test_data', 'models'
# to exist and be in the same directory as this script

MODEL_NAME = 'bert-base-uncased' # something like ./models/bert_epochs_1_lr_1e-05_batch_16 when loading trained model
DATASET_NAME = 'ccdv/arxiv-classification'
NUM_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
TRAIN_MODEL = True
PREPROCESS_DATA = True

# Use test flag to only use 1% of each dataset
TEST = True

ORIGINAL_LABELS = ['math.AC', 'cs.CV', 'cs.AI', 'cs.SY', 'math.GR', 'cs.DS', 'cs.CE', 'cs.PL', 'cs.IT', 'cs.NE', 'math.ST']

def fix_labels(instance):
    classConversion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    instance['labels'] = classConversion[instance['label']]
    return instance

# Tokenize random sentences
def tokenize_batch_random(batch):
    # Create list of sentences
    sentences = batch['text'].split('.')
    print(sentences)
    exit()


    return tokenizer(batch['text'], max_length=512, truncation=True, return_tensors='pt')

def tokenize_batch(batch):
    return tokenizer(batch['text'], max_length=512, truncation=True, return_tensors='pt')

def load_process_data_from_hub():
    split = '[:1%]' if TEST else ''
    train_data = load_dataset(DATASET_NAME, "no_ref", split=f'train{split}')
    test_data = load_dataset(DATASET_NAME, "no_ref", split=f'test{split}')
    val_data = load_dataset(DATASET_NAME, "no_ref", split=f'validation{split}')

    train_data = train_data.map(fix_labels)
    test_data = test_data.map(fix_labels)
    val_data = val_data.map(fix_labels)

    train_data_random = train_data.map(tokenize_batch_random, batched=False)

    # Preprocessing
    train_data = train_data.map(tokenize_batch, batched=False)
    print(train_data_random)
    print('------------------------------\n\n\n')
    print(train_data)
    exit()
    test_data = test_data.map(tokenize_batch)
    val_data = val_data.map(tokenize_batch)

    ## Save the tokenized datasets
    test_str = 'test_' if TEST else ''
    train_data.save_to_disk(f'./tokenized_random_{test_str}data/train')
    test_data.save_to_disk(f'./tokenized_random_{test_str}data/test')
    val_data.save_to_disk(f'./tokenized_random_{test_str}data/val')

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