import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import datetime
import os
import zipfile


def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.

    Parameters:
    - history: TensorFlow model History object.

    Returns:
    - Plot: Loss curves for training and validation.
    """
    loss, val_loss = history.history['loss'], history.history['val_loss']
    accuracy, val_accuracy = history.history['accuracy'], history.history['val_accuracy']
    epochs = range(len(history.history['loss']))

    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

import matplotlib.pyplot as plt

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.

    Parameters:
    - original_history: History object from the original model (before new_history).
    - new_history: History object from continued model training (after original_history).
    - initial_epochs: Number of epochs in original_history (new_history plot starts from here).

    Returns:
    - Plot: Comparison of training and validation metrics.
    """
    acc, loss = original_history.history["accuracy"], original_history.history["loss"]
    val_acc, val_loss = original_history.history["val_accuracy"], original_history.history["val_loss"]
    total_acc, total_loss = acc + new_history.history["accuracy"], loss + new_history.history["loss"]
    total_val_acc, total_val_loss = val_acc + new_history.history["val_accuracy"], val_loss + new_history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def unzip_data(filename):
    """
    Unzips a file into the current working directory.

    Parameters:
    - filename (str): Filepath of the target zip folder to be unzipped.
    """
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()


def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall, and f1 score of a binary classification model.

    Parameters:
    - y_true: True labels in the form of a 1D array.
    - y_pred: Predicted labels in the form of a 1D array.

    Returns:
    - Dictionary: Accuracy, precision, recall, and f1-score.
    """
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                    "precision": model_precision,
                    "recall": model_recall,
                    "f1": model_f1}
    return model_results


def split_chars(text):
    """
    Splits a given text into individual characters and returns a string
    where each character is separated by a space.

    Parameters:
    - text (str): The input text to be split into characters.

    Returns:
    str: A string containing individual characters from the input text,
         separated by a space.

    Example:
    >>> split_chars("Hello")
    'H e l l o'
    """
    return " ".join(list(text))
