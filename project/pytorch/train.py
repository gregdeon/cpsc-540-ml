import argparse
from tqdm import tqdm
import numpy as np
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
 
from dataset import get_dataloader
from models import StockfishScoreModel, LinearMovesModel, NeuralNet
from evaluate import evaluate, accuracy, nll

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file.
    From https://github.com/cs230-stanford/cs230-code-examples/blob/478e747b1c8bf57c6e2ce6b7ffd8068fe0287056/pytorch/nlp/utils.py#L94
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def save_checkpoint(path, model, optimizer, best_loss):
    """
    Save the current training state in a checkpoint file
    """
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_loss': best_loss
    }, path)

def load_checkpoint(path, model, optimizer):
    """
    Load a previous checkpoint into the model and optimizer
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['best_loss']

def train_epoch(model, loss_fn, optimizer, data):
    """
    Train for a single epoch (pass through the entire training set).

    Note: can only use a batch size of 1 -- updates model weights after every example.
    TODO: add config for batch size, etc...

    :return: total loss across all training examples during the epoch
    """

    # Set model to training mode
    model.train()

    loss_total = 0

    # Loop through an entire epoch
    for batch_inputs, batch_labels in tqdm(data, total=len(data)):
        # Forward pass
        batch_outputs = model(batch_inputs)
        loss = loss_fn(batch_outputs, batch_labels)
        loss_total += loss

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_total

def train(model, loss_fn, optimizer, train_data, validation_data, num_epochs, model_dir, restore_file=None):
    """
    Train for many epochs, evaluating and saving models along the way

    :param model: model object
    :param loss_fn: training criterion; also used when evaluating
    :param optimizer: optimizer object
    :param train_data: generator for training data
    :param validation_data: generator for validation data
    :param num_epochs: number of passes to perform through the training data. TODO: put into params bbject?
    :param model_dir: path for saving intermediate models and checkpoints
    :param restore_file: optional file to restore training from

    TODO: add parameter like
    :param batch_size: number of datapoints to consider for each optimizer step
    """
    
    best_validation_loss = np.inf
    
    # Restart from existing checkpoint, if provided
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file)
        print("Restoring parameters from %s" % (restore_path))
        best_validation_loss = load_checkpoint(restore_path, model, optimizer)

    # Train
    for epoch in range(num_epochs):
        print('Starting epoch %d/%d' % (epoch+1, num_epochs))

        # Train for one epoch
        print('Training...')
        training_loss = train_epoch(model, loss_fn, optimizer, train_data)
        print('Training loss: %.4f' % training_loss)
        print()

        # Validate
        print('Validating...')
        validation_metrics = {
            'loss': nll,
            # 'accuracy': accuracy,
        }
        validation_results = evaluate(model, validation_data, validation_metrics)
        for metric in validation_results:
            print('- %s: %.4f' % (metric, validation_results[metric]))
        print()

        # Save model
        torch.save(model.state_dict(), os.path.join(model_dir, '%d.pt' % (epoch+1)))

        # Also save model as best if this beats validation record
        validation_loss = validation_results['loss']
        if validation_loss < best_validation_loss:
            print('Record validation loss: %.4f (beats %.4f)' % (validation_loss, best_validation_loss))
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best.pt'))

        # Save checkpoint
        save_checkpoint(os.path.join(model_dir, 'checkpoint-%d.pt' % (epoch+1)), model, optimizer, best_validation_loss )


if __name__ == "__main__":
    # TODO: don't hard-code locations of datasets
    # TODO: shuffle training data in loader?
    print('Loading training data...')
    train_generator = get_dataloader('../data/train.csv')
    # train_generator = get_dataloader('../data/dataset_subset.csv')
    
    print('Loading validation data...')
    validation_generator = get_dataloader('../data/val.csv')
    # validation_generator = get_dataloader('../data/dataset_subset.csv')

    print('Setting up models...')
    # model = LinearMovesModel(16)
    # model = StockfishScoreModel()
    model = NeuralNet(20, 16, 8, nn.ReLU())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # TODO: add command line interface for these
    num_epochs = 200
    batch_size = 1
    start_from_checkpoint = None
    # start_from_checkpoint = 'checkpoint-8.pt'
    train(model, loss_fn, optimizer, train_generator, validation_generator, num_epochs, 'models/nn', start_from_checkpoint)


# TODO: add command line interface
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', default='../../data/sample_dataset.csv',
#                     help="Path to dataset CSV")
# args = parser.parse_args()
# print(args.data_path)
