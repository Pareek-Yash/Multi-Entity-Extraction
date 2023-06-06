# Import tokenizer and model from main.py and use here
id2tag = {id: tag for tag, id in tag2id.items()}





def extract_entities(sentence):
    id2tag = {id: tag for tag, id in tag2id.items()}
    tokenized_sentence = tokenizer.encode(sentence)
    input_ids = torch.tensor([tokenized_sentence]).to(device)
    
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    # join BPE tokens to words
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(id2tag[label_idx])
            new_tokens.append(token)

    # group tokens by entity type
    entities = [(token, label) for token, label in zip(new_tokens, new_labels) if label != "O"]
    return entities


def post_process_entities(entities):
    grouped_entities = {'Name': [], 'LastName': [], 'Phone': [], 'Email': []}
    for entity, label in entities:
        if label == 'B-FirstName' or label == 'I-FirstName':
            grouped_entities['Name'].append(entity)
        elif label == 'B-LastName' or label == 'I-LastName':
            grouped_entities['LastName'].append(entity)
        elif label == 'B-Mobile' or label == 'I-Mobile':
            grouped_entities['Phone'].append(entity)
        elif label == 'B-Email' or label == 'I-Email':
            grouped_entities['Email'].append(entity)
    
    grouped_entities['Email'].append('com')
    # Join words in each group
    for key, values in grouped_entities.items():
        grouped_entities[key] = ''.join(values)
    return grouped_entities

inp = input("Enter Sentence: ")
entities = extract_entities(inp)
print(entities)
processed_entities = post_process_entities(entities)
print(processed_entities)