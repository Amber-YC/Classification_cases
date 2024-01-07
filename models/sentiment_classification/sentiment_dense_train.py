import preprocessing
from preprocessing import load_datasets, get_word2vec_embeddings, get_word2vec_sent_embeddings, word2vec_size
from sentiment_NNModel import FFNN, train_model, predict, get_classification_report, show_confusion_matrix
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import wandb


sentiment_train_path = '../../data/classification_data/data/sentiment/classification/classification_sentiment_train.jsonl'
sentiment_eval_path = '../../data/classification_data/data/sentiment/classification/classification_sentiment_eval.jsonl'

# get lines of the three task datasets

sentiment_train_data, sentiment_eval_data = load_datasets(sentiment_train_path, sentiment_eval_path)

# get dataframes:

sentiment_train_df = pd.DataFrame(sentiment_train_data)
sentiment_eval_df = pd.DataFrame(sentiment_eval_data)

# get the texts in train_dataset and eval_dataset

sentiment_train_texts = sentiment_train_df['text'].tolist()
sentiment_eval_texts = sentiment_eval_df['text'].tolist()


# get word2vec metrics
word2vec_model, word2vec_word2id = get_word2vec_embeddings(sentiment_train_texts)
train_word2vec_tensor = get_word2vec_sent_embeddings(sentiment_train_texts, word2vec_model, word2vec_word2id)
eval_word2vec_tensor = get_word2vec_sent_embeddings(sentiment_eval_texts, word2vec_model, word2vec_word2id)

"""get y"""
# encode the labels
y_train_sent = sentiment_train_df['sentiment'].tolist()
y_eval_sent = sentiment_eval_df['sentiment'].tolist()

label_encoder = LabelEncoder()
train_encoded = label_encoder.fit_transform(y_train_sent)
eval_encoded = label_encoder.transform(y_eval_sent)

Y_train = torch.tensor(train_encoded)
Y_eval = torch.tensor(eval_encoded)
# print(Y_train[:5], Y_eval[:5])
label_class_names = label_encoder.classes_

"""word2vec_train"""
input_size = word2vec_size
output_size = 2
# hyper-perameters for model
hidden_sizes = [128, 64]
# hyper-perameters for training
batch_size = 32
learning_rate = 0.05
num_epochs = 500

model = FFNN(input_size=input_size, output_size=output_size, hidden_sizes=hidden_sizes)

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

wandb.init(project="sentiment_classification_with_word2vec_embeddings")
train_model(train_word2vec_tensor, Y_train, model, optimizer, criterion, batch_size, num_epochs)
wandb.finish()
best_model_with_word2vec = model

"""check the accuracy on training and test data"""
Y_pred_train = predict(best_model_with_word2vec, train_word2vec_tensor)
Y_pred_eval = predict(best_model_with_word2vec, eval_word2vec_tensor)



train_report = get_classification_report(Y_train, Y_pred_train, label_class_names)
eval_report = get_classification_report(Y_eval, Y_pred_eval, label_class_names)
print(train_report)
print(eval_report)

show_confusion_matrix(Y_train, Y_pred_train, label_class_names, "Word2Vec_Train")
show_confusion_matrix(Y_eval, Y_pred_eval, label_class_names, "Word2Vec_Test")