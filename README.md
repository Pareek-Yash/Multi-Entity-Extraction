# Multi-Entity-Extraction

<p align="center">
<img src="https://github.com/Pareek-Yash/Multi-Entity-Extraction/blob/main/assets/images/Multi-Entity%20Extraction.png?raw=true" alt="image" width="400" height="400" />
</p>





## Task
There can be different Entities in a given sentence and this repo achives to target multiple entities together.
Not only we extract entities here. We also make synthetic data and convert our data to BIO format.

### The BIO Format
The BIO format (Beginning, Inside, Outside) is a common data representation method used in Named Entity Recognition (NER) tasks in Natural Language Processing (NLP). In this scheme, 'B' signifies the beginning of an entity, 'I' represents the inside of an entity, and 'O' denotes that the word is not part of an entity. 

For instance, consider the sentence: "John Doe works at Microsoft in Seattle." In BIO format, this would be represented as: \
John (B-PERSON) \
Doe (I-PERSON) \
works (O) \
at (O) \
Microsoft (B-ORGANIZATION) \
in (O) \
Seattle (B-LOCATION) 

The BIO tagging scheme was proposed as a part of the CoNLL (Conference on Natural Language Learning) shared tasks for chunking and named entity recognition.

## Getting Started

To start using this repository for your multi-entity extraction tasks, you will need to clone the repository and install the required packages. Instructions for both are included below.

### Prerequisites

This project requires Python 3.9+ and the following Python libraries installed:

- `numpy`
- `pandas`
- `sklearn`
- `torch`
- `transformers`

### Installation

1. Clone the repository to your local machine:
```
git clone https://github.com/Pareek-Yash/Multi-Entity-Extraction.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

After installing the prerequisites, you can run the script as follows:

```
python main.py
```

This will start the training process for the NER model on your synthetic data in BIO format.

## Synthetic Data Generation and Conversion to BIO Format

An innovative aspect of this project is the generation of synthetic data for training the NER model. We take use of `faker` library to generate queries, providing ample and diverse data for robust model training. We generate a total of 1000 queries.

`bio_data.ipynb` converts our csv of generated data to the BIO format.

Example dataset:

| Query ID | Query                                                                 | Email                    | Mobile               | FirstName   | LastName  |
|----------|-----------------------------------------------------------------------|--------------------------|----------------------|-------------|-----------|
| 0        | Please pull up contact details for Sherry Taylor...                   | allenmarissa@example.org | (733)806-4015x707   | Sherry      | Taylor    |
| 1        | Please pull up contact details for Christopher...                     | zmendez@example.com     | (518)319-0507       | Christopher | Walker    |
| 2        | Please pull up contact details for David Robinson...                  | stephanie52@example.org  | 001-197-751-2495x89864 | David       | Robinson  |
| 3        | Please pull up contact details for Jon Williams...                    | jaime53@example.org      | +1-172-648-1928x44276 | Jon         | Williams  |
| 4        | Please pull up contact details for Lisa Thompson...                   | aguilarandrew@example.net | 992-195-6796         | Lisa        | Thompson  |


## Model Training and Evaluation

The model training script uses the Transformer library, which allows the use of state-of-the-art transformer models like BERT and DistilBERT for NER tasks. The model's performance is evaluated using val_loss metric on a held-out test set.

Weights and biases is used to track the multiple iteration of ML model.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Future Work

We plan to extend the current model capabilities by adding more entity types and training on a wider range of data sources. We're also looking forward to developing a user-friendly application that can utilize this model for real-time entity extraction.

For any questions or feedback, feel free to open an issue on this repository. Connect with me on Linkedin @ Pareek-Yash.
