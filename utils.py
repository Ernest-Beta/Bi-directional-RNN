import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
from sklearn.model_selection import train_test_split

#vevaiosi oti yparxei to tokenizer tis NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize

#dataset paths
DATA_DIR = r"C:\Users\ernes\Desktop\PartA\Dataset\aclImdb"

#orismos twn diadromwn gia thetikes kai arnitikes kritikes
TRAIN_POS_DIR = os.path.join(DATA_DIR, "train", "pos")
TRAIN_NEG_DIR = os.path.join(DATA_DIR, "train", "neg")
TEST_POS_DIR = os.path.join(DATA_DIR, "test", "pos")
TEST_NEG_DIR = os.path.join(DATA_DIR, "test", "neg")

#path gia ta GloVe embeddings
GLOVE_PATH = r"C:\Users\ernes\Desktop\PartA\BiRNN\glove.6B.300d.txt"
EMBEDDING_DIM = 300  #xrisi 300-diastatwn embeddings

#synartisi gia fortosi keimenwn apo katalogous
def read_files_from_directory(directory, label):
    texts = []
    labels = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            texts.append(file.read().strip())  #anakthsi kai katharismos tou keimenou
            labels.append(label)  #orizoume ta labels
    return texts, labels

#fortosi twn ekpaideutikon dedomenwn
train_pos_texts, train_pos_labels = read_files_from_directory(TRAIN_POS_DIR, 1)
train_neg_texts, train_neg_labels = read_files_from_directory(TRAIN_NEG_DIR, 0)

#fortosi twn test dedomenwn
test_pos_texts, test_pos_labels = read_files_from_directory(TEST_POS_DIR, 1)
test_neg_texts, test_neg_labels = read_files_from_directory(TEST_NEG_DIR, 0)

#sygxoneusi twn ekpaideutikon kai test dedomenwn
train_texts = train_pos_texts + train_neg_texts
train_labels = train_pos_labels + train_neg_labels
test_texts = test_pos_texts + test_neg_texts
test_labels = test_pos_labels + test_neg_labels

#diaxorismos ekpaideusis se development set (80% train, 20% dev)
train_texts, dev_texts, train_labels, dev_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)
#synartisi tokenize
def tokenize(text):
    return word_tokenize(text.lower())  #metatropi se mikra kai tokenopoihsh

#tokenize olwn twn keimenwn
train_tokens = [tokenize(text) for text in train_texts]
dev_tokens = [tokenize(text) for text in dev_texts]
test_tokens = [tokenize(text) for text in test_texts]

#synartisi dhmiourgias leksilogiou
def build_vocab(tokenized_texts, vocab_size=20000):
    word_counts = Counter(word for tokens in tokenized_texts for word in tokens)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_counts.most_common(vocab_size))}
    vocab["<PAD>"] = 0  #padding token
    vocab["<UNK>"] = len(vocab)  #unknown token
    return vocab

#dhmiourgia tou leksilogiou apo ta ekpaideutika dedomena
vocab = build_vocab(train_tokens)
vocab_size = len(vocab)

#metatroph tokens se akeraious kwdikous
def tokens_to_ids(tokens, vocab, max_length=500):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_length]] + [vocab["<PAD>"]] * (max_length - len(tokens))

#metatroph twn keimenwn se akeraious kwdikous
train_sequences = [tokens_to_ids(tokens, vocab) for tokens in train_tokens]
dev_sequences = [tokens_to_ids(tokens, vocab) for tokens in dev_tokens]
test_sequences = [tokens_to_ids(tokens, vocab) for tokens in test_tokens]

#metatroph se PyTorch tensors
train_data = torch.tensor(train_sequences, dtype=torch.long)
dev_data = torch.tensor(dev_sequences, dtype=torch.long)
test_data = torch.tensor(test_sequences, dtype=torch.long)

train_labels = torch.tensor(train_labels, dtype=torch.long)
dev_labels = torch.tensor(dev_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

#synartisi fortosis twn GloVe embeddings
def load_glove_embeddings(glove_path, vocab, embedding_dim=300):
    embeddings = {}
    print('Loading GloVe embeddings...')

    #anagnosi tou GloVe arxeiou
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)

            #elegxos gia swsto dimension
            if vector.shape[0] == embedding_dim:
                embeddings[word] = vector 

    #arxikopoihsh tou embedding matrix me mikro tuxaies times
    embedding_matrix = np.random.uniform(-0.05, 0.05, (len(vocab), embedding_dim))

    for word, idx in vocab.items():
        if word in embeddings:
            embedding_matrix[idx] = embeddings[word]  # Replace with GloVe vector

    return torch.tensor(embedding_matrix, dtype=torch.float32)

#fortosi twn GloVe embeddings
pretrained_embedding_tensor = load_glove_embeddings(GLOVE_PATH, vocab, embedding_dim=EMBEDDING_DIM)

#dhmiourgia dataset class gia PyTorch
class IMDBDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

#dhmiourgia twn dataset objects
train_dataset = IMDBDataset(train_data, train_labels)
dev_dataset = IMDBDataset(dev_data, dev_labels)
test_dataset = IMDBDataset(test_data, test_labels)

#dhmiourgia twn DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#synartisi epistrofis twn DataLoaders gia train kai dev
def load_data():
    return train_loader, dev_loader, vocab_size, pretrained_embedding_tensor

#synartisi epistrofis tou DataLoader gia test
def load_test_data():
    return test_loader, vocab_size, pretrained_embedding_tensor
