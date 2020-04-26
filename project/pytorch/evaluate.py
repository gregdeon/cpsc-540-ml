"""
Functions for evaluating model performance.

TODO: make command line interface.
"""

from models import PreferBackwardMoves, StockfishScoreModel
from dataset import get_dataloader

import numpy as np

import torch
import torch.nn as nn

def accuracy(outputs, labels):
    """
    Compute the fraction of examples that the model gets correct.
    Predicted labels are max logit outputs, breaking ties randomly.
    """
    num_examples = len(outputs)

    # Choose randomly among the most likely moves
    predicted_labels = [np.random.choice(np.flatnonzero(output == output.max())) for output in outputs]

    correct = [predicted_labels[i] == labels[i] for i in range(num_examples)]
    return float(sum(correct)) / num_examples

def log_likelihood(outputs, labels):
    """
    Compute the log-likelihood (cross entropy) of the data, given the model.
    """
    num_examples = len(outputs)
    log_softmax = nn.LogSoftmax(dim=1)
    log_probabilities = [log_softmax(output) for output in outputs]
    log_p_class = [log_probabilities[i][0][labels[i]] for i in range(num_examples)]
    return sum(log_p_class).item()

def nll(outputs, labels):
    """
    Compute the negative log-likelihood.
    """
    return -log_likelihood(outputs, labels)

def evaluate(model, data_generator, metrics):
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
        # TODO: this is hardcoded for batch size of 1...
        for batch_inputs, batch_labels in data_generator:
            batch_outputs = model(batch_inputs)
            outputs.append(batch_outputs)
            labels.append(batch_labels)

    # Summarize with metrics
    results = {metric: metrics[metric](outputs, labels) for metric in metrics}

    return results


if __name__ == "__main__":
    print('Loading...')
    data_generator = get_dataloader('../data/dataset_subset.csv')
    # model = StockfishScoreModel()
    model = PreferBackwardMoves()
    metrics = {
        'accuracy': accuracy,
        'log_likelihood': log_likelihood,
    }
    
    print('Evaluating...')
    results = evaluate(model, data_generator, metrics)
    print(results)
