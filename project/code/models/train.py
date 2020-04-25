import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange

from dataset import get_dataloader
from models import StockfishScoreModel
from evaluate import evaluate, accuracy, log_likelihood

def train(model, data_iterator, loss_fn, optimizer, num_steps):
    """
    TODO: add config for batch size, number of epochs, etc...
    """

    # Set model to training mode
    model.train()

    # Progress bar
    t = trange(num_steps)
    for i in t:
        # Forward pass
        batch_inputs, batch_labels = next(data_iterator)
        batch_outputs = model(batch_inputs)
        loss = loss_fn(batch_outputs, batch_labels)

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # TODO: save checkpoints

    # TODO: save best model



if __name__ == "__main__":

    print('Loading training data...')
    train_generator = get_dataloader('../../data/sample_dataset_subset.csv')
    
    print('Loading validation data...')
    validation_generator = get_dataloader('../../data/sample_dataset_subset.csv')
    validation_metrics = {
        'accuracy': accuracy,
        'log_likelihood': log_likelihood,
    }

    print('Setting up models...')
    model = StockfishScoreModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(20):
        train(model, iter(train_generator), loss_fn, optimizer, len(train_generator))
        print(evaluate(model, validation_generator, validation_metrics))
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)


# TODO: add command line interface
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', default='../../data/sample_dataset.csv',
#                     help="Path to dataset CSV")
# args = parser.parse_args()
# print(args.data_path)
