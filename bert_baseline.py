from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

## Note: expects directories called 'tokenized_original_data', 'tokenized_binary_data' and 'models'
# to exist and be in the same directory as this script

# MODEL_NAME = 'bert-base-uncased' # something like ./models/bert_epochs_1_lr_1e-05_batch_16 when loading trained model
MODEL_NAME = './models/bert_epochs_3_lr_1e-05_batch_4'
DATASET_NAME = 'ccdv/arxiv-classification'
NUM_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
TRAIN_MODEL = False
PREPROCESS_DATA = False

USE_ORIGINAL_LABELS = True 
ORIGINAL_LABELS = ['math.AC', 'cs.CV', 'cs.AI', 'cs.SY', 'math.GR', 'cs.DS', 'cs.CE', 'cs.PL', 'cs.IT', 'cs.NE', 'math.ST']

def fix_labels(instance):
    if USE_ORIGINAL_LABELS:
        classConversion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        classConversion = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
    
    instance['labels'] = classConversion[instance['label']]
    return instance

def tokenize_batch(batch):
    return tokenizer(batch['text'], max_length=512, truncation=True, return_tensors='pt')

def load_process_data_from_hub():
    train_data = load_dataset(DATASET_NAME, "no_ref", split='train')
    test_data = load_dataset(DATASET_NAME, "no_ref", split='test')
    val_data = load_dataset(DATASET_NAME, "no_ref", split='validation')

    train_data = train_data.map(fix_labels)
    test_data = test_data.map(fix_labels)
    val_data = val_data.map(fix_labels)

    # Preprocessing
    train_data = train_data.map(tokenize_batch, batched=False)
    test_data = test_data.map(tokenize_batch)
    val_data = val_data.map(tokenize_batch)

    ## Save the tokenized datasets
    label_str = 'original' if USE_ORIGINAL_LABELS else 'binary'
    train_data.save_to_disk(f'./tokenized_{label_str}_data/train')
    test_data.save_to_disk(f'./tokenized_{label_str}_data/test')
    val_data.save_to_disk(f'./tokenized_{label_str}_data/val')

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
    
    label_str = 'original' if USE_ORIGINAL_LABELS else 'binary'
    model.save_pretrained(f'./models/bert_{label_str}_epochs_{NUM_EPOCHS}_lr_{LEARNING_RATE}_batch_{BATCH_SIZE}')

    return model

if __name__ == '__main__':
    num_labels = 11 if USE_ORIGINAL_LABELS else 2
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    global tokenizer 
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #change to torch.device('mps') if running on mac
    model = model.to(device)

    if PREPROCESS_DATA:
        load_process_data_from_hub()

    ## Load the tokenized datasets from disk
    label_str = 'original' if USE_ORIGINAL_LABELS else 'binary'
    train_data = load_from_disk(f'./tokenized_{label_str}_data/train')
    test_data = load_from_disk(f'./tokenized_{label_str}_data/test')
    val_data = load_from_disk(f'./tokenized_{label_str}_data/val')

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

    cm = metrics.confusion_matrix(val_labels, val_preds)
    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(ORIGINAL_LABELS))
    plt.xticks(tick_marks, ORIGINAL_LABELS, rotation=45)
    plt.yticks(tick_marks, ORIGINAL_LABELS)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Save confusion matrix as a figure
    plt.savefig('confusion_matrix.png')

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