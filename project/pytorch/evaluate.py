"""
Functions for evaluating model performance.
"""

import argparse
import numpy as np
from tqdm import tqdm
import functools

import torch
import torch.nn as nn

from dataset import ChessDataset
from models import build_model

def num_correct(outputs, labels):
    """
    Compute the fraction of examples that the model gets correct.
    Predicted labels are max logit outputs, breaking ties randomly.
    """
    num_examples = len(outputs)

    # Choose randomly among the most likely moves
    predicted_labels = [(output == output.max()).float() for output in outputs]
    num_predicted = [sum(predicted) for predicted in predicted_labels]
    correct = [predicted_labels[i][labels[i]] / num_predicted[i] for i in range(num_examples)]
    return float(sum(correct))

def top_k(outputs, labels, k):
    num_examples = len(outputs)
    prediction_orders = [output.argsort(descending=True).argsort() for output in outputs]
    correct = [prediction_orders[i][labels[i]] < k for i in range(num_examples)]
    return float(sum(correct))

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

def plot_distribution(outputs, labels):
    softmax = nn.Softmax(dim=0)
    probabilities = [softmax(output) for output in outputs]
    p_class = [probabilities[i][labels[i]] for i in range(len(outputs))]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(p_class, bins=np.geomspace(1e-5, 1e0, 16), rwidth=0.8, zorder=10)
    plt.xscale('log')
    plt.show()

    # Dummy value
    return 0

def list_probabilities(outputs, labels):
    softmax = nn.Softmax(dim=0)
    probabilities = [softmax(output) for output in outputs]

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
        for inputs, label in tqdm(dataset, desc='Examples'):
            output = model(inputs)
            outputs.append(output)
            labels.append(label)

    # Summarize with metrics
    results = {metric: metrics[metric](outputs, labels) for metric in metrics}

    return results

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',                                                                 help='(string) Path to model file')
parser.add_argument('--data_source', choices=['train', 'val', 'test'], default='val',               help='(string) Dataset to evaluate against')
parser.add_argument('--train_data',                                    default='../data/train.csv', help='(string) Path to training set CSV')
parser.add_argument('--validation_data',                               default='../data/val.csv',   help='(string) Path to validation set CSV')
parser.add_argument('--test_data',                                     default='../data/test.csv',  help='(string) Path to test set CSV')
parser.add_argument('--load_cached',                                   action='store_true',         help='Load cached versions of the training and validation datasets')

if __name__ == "__main__":
    args = parser.parse_args()

    print('Loading data...')
    if args.load_cached:
        # TODO: don't hard-code these
        data_sources = {'train': '../data/train_cached.pt', 'val': '../data/val_cached.pt', 'test': '../data/test_cached.pt'}
        dataset = torch.load(data_sources[args.data_source])
    else:
        data_sources = {'train': args.train_data, 'val': args.validation_data, 'test': args.test_data}
        dataset = ChessDataset(data_sources[args.data_source], data_sources['train'])

    print('Loading model...')
    model = torch.load(args.model_path)
    
    print('Evaluating...')
    metrics = {
        'num_correct': num_correct,
        'top_1' : functools.partial(top_k, k=1),
        'top_3' : functools.partial(top_k, k=3),
        'top_5' : functools.partial(top_k, k=5),
        'top_10': functools.partial(top_k, k=10),
        'top_20': functools.partial(top_k, k=20),
        'log_likelihood': log_likelihood,
        # 'plot': plot_distribution,
    }
    results = evaluate(model, dataset, metrics)
    print('Results:')
    for metric in results:
        print('- %s: %.4f' % (metric, results[metric]))
        print('  (%.4f / instance)' % (results[metric] / len(dataset)))
