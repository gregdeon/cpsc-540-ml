"""
Functions for evaluating model performance.

TODO: make command line interface.
"""

from models import PreferBackwardMoves, StockfishScoreModel
from dataset import get_dataloader

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

def accuracy(outputs, labels):
    """
    Compute the fraction of examples that the model gets correct.
    Predicted labels are max logit outputs, breaking ties randomly.
    """
    num_examples = len(outputs)

    # Choose randomly among the most likely moves
    predicted_labels = [(output == output.max()).float() for output in outputs]
    num_predicted = [sum(predicted) for predicted in predicted_labels]
    correct = [predicted_labels[i][labels[i]] / num_predicted[i] for i in range(num_examples)]
    return float(sum(correct)) / num_examples

def log_likelihood(outputs, labels):
    """
    Compute the log-likelihood (cross entropy) of the data, given the model.
    """
    num_examples = len(outputs)
    log_softmax = nn.LogSoftmax(dim=0)
    log_probabilities = [log_softmax(output) for output in outputs]
    log_p_class = [log_probabilities[i][labels[i]] for i in range(num_examples)]
    return sum(log_p_class).item()

def nll(outputs, labels):
    """
    Compute the negative log-likelihood.
    """
    return -log_likelihood(outputs, labels)

def evaluate(model, dataset, metrics):
    """
    Evaluate models on dataset and summarize with a number of metrics

    :param model: a model object from models.py
    :param data_generator: a DataLoader object
    :param metrics: dict of {metric_name: function}, where each function accepts a list of logit outputs and a list of correct labels
    :return: dict of results from evaluating metrics
    """

    # Put model into evaluate mode 
    model.eval()

    # Compute model outputs
    outputs = []
    labels = []
    with torch.no_grad():
        for inputs, label in tqdm(dataset):
        # TODO: this is hardcoded for batch size of 1...
            output = model(inputs)
            outputs.append(output)
            labels.append(label)

    # Summarize with metrics
    results = {metric: metrics[metric](outputs, labels) for metric in metrics}

    return results


if __name__ == "__main__":
    print('Loading...')
    from dataset import ChessDataset
    dataset = ChessDataset('../data/val.csv')
    feature_names = dataset.get_column_names()

    model = StockfishScoreModel()
    # model = PreferBackwardMoves()
    metrics = {
        'accuracy': accuracy,
        'log_likelihood': log_likelihood,
    }
    
    print('Evaluating...')
    results = evaluate(model, dataset, metrics)
    print(results)
