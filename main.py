import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast, BertForTokenClassification, get_linear_schedule_with_warmup, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Initialize the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

# Function to read data from a given file
def read_data(file_path):
    sentences, tags = [], []
    sentence, tag = [], []
    with open(file_path, "r") as file:
        for line in file:
            stripped_line = line.strip().split(' ')
            # When a new sentence starts
            if stripped_line[0] == "":
                if len(sentence) > 0:
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence, tag = [], []
            else:
                sentence.append(stripped_line[0])
                tag.append(stripped_line[1])
    # Appending last sentence
    if len(sentence) > 0:
        sentences.append(sentence)
        tags.append(tag)
    return sentences, tags

# Function to encode tags
def encode_tags(tags, encodings, tag2id):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

# Read data from text file
sentences, tags = read_data("ner_data.txt")

# Define your tag2id dictionary based on your tags
tag2id = {"O": 0, "B-FirstName": 1, "I-FirstName": 2, "B-LastName": 3, "I-LastName": 4, "B-Email": 5, "I-Email": 6, "B-Mobile": 7, "I-Mobile": 8}

# Tokenize the texts and encode the tags
input_ids = []
attention_masks = []
encoded_tags = []

for sentence, sentence_tags in zip(sentences, tags):
    encoded_dict = tokenizer.encode_plus(
                        sentence,
                        add_special_tokens = True,
                        max_length = 128,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        is_split_into_words=True,
                        return_offsets_mapping=True,
                   )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

    labels = encode_tags([sentence_tags], encoded_dict, tag2id)
    encoded_tags.append(labels)

# Convert list to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
tags = torch.tensor(encoded_tags)

# Check if a GPU is available and if so, set the device to GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Initialize BERT model for token classification
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(tag2id),  # The number of output labels--in this case, the total number of tags.
    output_attentions=False,  # Whether the model returns attention weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)
model.to(device)

# Split each dataset into training and validation sets
input_ids_train, input_ids_val, tags_train, tags_val = train_test_split(input_ids, tags, test_size=0.1)
attention_masks_train, attention_masks_val, _, _ = train_test_split(attention_masks, input_ids, test_size=0.1)

# Create TensorDatasets
train_data = TensorDataset(input_ids_train, attention_masks_train, tags_train)
valid_data = TensorDataset(input_ids_val, attention_masks_val, tags_val)

# Specify batch size
batch_size = 32

# Create the DataLoaders
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), batch_size=batch_size)

# Define the optimizer
optimizer = AdamW(
    model.parameters(),
    lr=3e-5,  # You can tune the learning rate
    eps=1e-8  # a very small number to prevent any division by zero in the implementation
)

from transformers import get_linear_schedule_with_warmup
import torch
import numpy as np

max_grad_norm = 1.0
epochs = 5  # Starting point

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,  # Default value in run_glue.py
                                            num_training_steps = total_steps)

# Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

for _ in range(epochs):
    # Perform one full pass over the training set.
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        model.zero_grad()

        # forward pass
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()

        # backward pass
        loss.backward()

        # prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    # Validation loop
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    predictions , true_labels = [], []

    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        logit = outputs[1]

        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logit.detach().cpu().numpy(), axis=2)])
        true_labels.extend(b_labels)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))