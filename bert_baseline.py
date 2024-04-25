from datasets import load_dataset, load_from_disk
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from torch.nn.functional import softmax

## Note: expects directories called 'tokenized_original_data', 'tokenized_binary_data' and 'models'
# to exist and be in the same directory as this script

# MODEL_NAME = 'bert-base-uncased' # something like ./models/bert_epochs_1_lr_1e-05_batch_16 when loading trained model
MODEL_NAME = './models/bert_epochs_3_lr_1e-05_batch_4'
DATASET_NAME = 'ccdv/arxiv-classification'
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
TRAIN_MODEL = False
PREPROCESS_DATA = False
DISPLAY_ERRORS = True

USE_ORIGINAL_LABELS = True 
ORIGINAL_LABELS = ['math.AC', 'cs.CV', 'cs.AI', 'cs.SY', 'math.GR', 'cs.DS', 'cs.CE', 'cs.PL', 'cs.IT', 'cs.NE', 'math.ST']

def fix_labels(instance):
    if USE_ORIGINAL_LABELS:
        classConversion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        classConversion = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
    
    instance['labels'] = classConversion[instance['label']]
    return instance

def predict_proba(texts):
    tokenized = tokenizer(texts, max_length=512, truncation=True, return_tensors='pt', padding='max_length')
    with torch.no_grad():
        inputs = {name: tensor.to(device) for name, tensor in tokenized.items()}
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
    return probs.squeeze().detach().cpu().numpy()

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

def display_errors(val_preds, val_labels):
    '''
    Print out original labels and text for incorrect predictions, one at a time
    Ask for user input to continue to next error
    '''
    errors = []
    for i, pred in enumerate(val_preds):
        if pred != val_labels[i]:
            errors.append(i)
    original_val_ds = load_dataset(DATASET_NAME, 'no_ref', split='validation')
    tokenized_val_ds = load_from_disk('./tokenized_data/val')

    explainer = LimeTextExplainer(class_names=ORIGINAL_LABELS)

    for i in errors:
        print("Original label: ", ORIGINAL_LABELS[original_val_ds[i]['label']])
        print("Predicted label: ", val_preds[i])
        print("Correct label: ", val_labels[i])
        print("Text: ", tokenizer.decode(tokenized_val_ds[i]['input_ids'][0]))
        print("Text from DS: ", original_val_ds[i]['text'][:1000])

        explanation = explainer.explain_instance(tokenizer.decode(tokenized_val_ds[i]['input_ids'][0]), predict_proba, num_features=10, num_samples=1000)

        fig = explanation.as_pyplot_figure()
        plt.savefig(f'bert_graphs/{LEARNING_RATE}_{NUM_EPOCHS}_{BATCH_SIZE}_explanation_{i}.png')

def analyze_errors(val_preds, val_labels):
    '''
    Generate plot of original labels for incorrect predictions
    '''

    errors = []
    for i, pred in enumerate(val_preds):
        if pred != val_labels[i]:
            errors.append(i)
    original_val_ds = load_dataset(DATASET_NAME, 'no_ref', split='validation')
    
    misclassifications = np.zeros(11)
    for i in errors:
        misclassifications[original_val_ds[i]['label']] += 1
    
    # plot histogram of misclassifications
    # X-axis: original label text
    # Y-axis: number of misclassifications
    plt.figure(figsize=(10, 5))
    plt.bar(ORIGINAL_LABELS, misclassifications)
    plt.xlabel('Original Label')
    plt.ylabel('Number of Misclassifications')
    plt.title('Misclassifications by Original Label')
    # save plot
    plt.savefig(f'bert_graphs/{LEARNING_RATE}_{NUM_EPOCHS}_{BATCH_SIZE}_misclassifications.png')

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

    cm = metrics.confusion_matrix(val_labels, val_preds, normalize="pred")
    plt.figure(figsize=(12, 12))
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ORIGINAL_LABELS)
    disp.plot()
    plt.savefig(f'confusion_matrix.png')

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

    analyze_errors(val_preds, val_labels)
    display_errors(val_preds, val_labels) if DISPLAY_ERRORS else None