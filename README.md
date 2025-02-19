BiRNN Sentiment Analysis

Overview 📖

This project implements a Bidirectional Recurrent Neural Network (BiRNN) for sentiment analysis, using pretrained word embeddings (GloVe) and LSTM layers for feature extraction. It predicts whether a given text has a positive or negative sentiment, making it ideal for analyzing customer feedback, reviews, and opinions.

Features ✨

BiRNN with LSTM: Captures sequential word dependencies in text.

GloVe Embeddings: Uses pre-trained word vectors for better generalization.

Tokenization & Vocabulary Generation: Converts raw text into a structured format.

Learning Curve Analysis: Tracks training performance over time.

Early Stopping: Prevents overfitting by monitoring validation loss.

Evaluation Metrics: Computes accuracy, precision, recall, and F1-score.

Project Structure 🏗️

├── BiRNN/

│   ├── train.py                 # Trains the BiRNN model with early stopping

│   ├── evaluate.py              # Evaluates the trained model on test data

│   ├── StackedBiNN.py           # Defines the BiRNN architecture

│   ├── utils.py                 # Data preprocessing, tokenization & embeddings

│   ├── dataset_loader.py        # Loads dataset into PyTorch DataLoader

│   ├── best_model.pth           # Saved best performing model

│   ├── add glove 300d embedings here ####

├── Dataset/

│   ├── aclImdb/                 # IMDB movie review dataset

├── README.md                    # This documentation

How It Works 🚀

1️⃣ Dataset Loading

Loads IMDB dataset with positive and negative movie reviews.

Tokenizes text and converts it into numerical sequences.

Splits data into training, validation, and test sets.

2️⃣ Model Training

Uses a Stacked BiRNN (Bi-LSTM) architecture.

Initializes with GloVe embeddings for better word representations.

Employs Adam optimizer and cross-entropy loss.

Early stopping prevents overfitting by tracking validation loss.

Saves the best model checkpoint based on development loss.

3️⃣ Evaluation & Prediction

Tests the model on unseen IMDB reviews.

Computes accuracy, precision, recall, and F1-score.

Predicts the sentiment of new text samples.


Usage 🛠️

1️⃣ Train the Model

python BiRNN/train.py

This will train the model and save the best version as best_model.pth.

2️⃣ Evaluate the Model

python BiRNN/evaluate.py

This script loads the trained model and evaluates its accuracy and performance metrics on the test set.

3️⃣ Predict Sentiment for New Text

Modify evaluate.py to provide a custom text input for prediction.

Learning Curve 📈

During training, we track the loss on both training and validation data. The loss curves help us understand the model’s generalization ability. The training script automatically smooths the learning curve using interpolation for better visualization.
