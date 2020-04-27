import argparse
from tqdm import tqdm
import numpy as np
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
 
from dataset import ChessDataset
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

def save_checkpoint(path, model, optimizer, best_loss, epochs):
    """
    Save the current training state in a checkpoint file
    """
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_loss': best_loss,
        'epochs': epochs,
    }, path)

def load_checkpoint(path, model, optimizer):
    """
    Load a previous checkpoint into the model and optimizer

    :return: tuple of (best validation loss, epochs completed)
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return (checkpoint['best_loss'], checkpoint['epochs'])

def train_epoch(model, loss_fn, optimizer, data, batch_size):
    """
    Train for a single epoch (pass through the entire training set).

    :return: total loss across all training examples during the epoch
    """

    # Set model to training mode
    model.train()

    # Shuffle data
    num_examples = len(data)
    permutation = torch.randperm(num_examples)

    # Loop through an entire epoch
    loss_total = 0
    for idx_start_batch in tqdm(range(0, num_examples, batch_size), desc='Batches'):
        batch_loss = 0
        batch_indices = permutation[idx_start_batch : min(idx_start_batch + batch_size, num_examples)]
        for i in batch_indices:
            inputs, label = data[i]
            output = model(inputs)
            batch_loss += loss_fn(output.unsqueeze(dim=0), label.unsqueeze(dim=0))

        # Update model
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        loss_total += batch_loss

    return loss_total

def train(model, loss_fn, optimizer, train_data, validation_data, num_epochs, batch_size, model_dir, restore_checkpoint=None):
    """
    Train for many epochs, evaluating and saving models along the way

    :param model: model object
    :param loss_fn: training criterion; also used when evaluating
    :param optimizer: optimizer object
    :param train_data: generator for training data
    :param validation_data: generator for validation data
    :param num_epochs: number of passes to perform through the training data. TODO: put into params bbject?
    :param batch_size: number of datapoints to consider for each optimizer step
    :param model_dir: path for saving intermediate models and checkpoints
    :param restore_checkpoint: optional file to restore training from
    """
    
    # Make model directory if it doesn't already exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_validation_loss = np.inf
    
    # Restart from existing checkpoint, if provided
    if restore_checkpoint is not None:
        restore_path = os.path.join(model_dir, 'checkpoint-%d.pt' % restore_checkpoint)
        print("Restoring parameters from %s..." % (restore_path))
        best_validation_loss, epoch = load_checkpoint(restore_path, model, optimizer)

    else:
        epoch = 0

    # Train
    while epoch < num_epochs:
        print('Starting epoch %d/%d' % (epoch+1, num_epochs))

        # Train for one epoch
        print('Training...')
        training_loss = train_epoch(model, loss_fn, optimizer, train_data, batch_size)
        print('Training loss: %.4f' % training_loss)
        print()

        # Validate
        print('Validating...')
        validation_metrics = {
            'loss': nll,
            'accuracy': accuracy,
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
            print()
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), os.path.join(model_dir, 'best.pt'))

        # Save checkpoint
        save_checkpoint(os.path.join(model_dir, 'checkpoint-%d.pt' % (epoch+1)), model, optimizer, best_validation_loss, epoch+1)

        epoch += 1


def build_model(model_type, feature_names):
    """
    Create a model object.
    """

    # TODO: build model of model_type
    num_board_features = len(feature_names['board'])
    num_move_features  = len(feature_names['move'])

    if model_type == 'random':
        model = 'TODO'

    elif model_type == 'stockfish_score':
        stockfish_score_index = feature_names['move'].index('move_stockfish_eval')
        model = StockfishScoreModel(stockfish_score_idx=stockfish_score_index)

    elif model_type == 'linear_moves':
        model = LinearMovesModel(num_move_features)

    elif model_type == 'nn_board':
        # TODO: read hidden layer size from 
        model = NeuralNet(num_board_features, num_move_features, 8, nn.ReLU())

    return model

parser = argparse.ArgumentParser()
parser.add_argument('--train_data',                default='../data/train.csv', help='(string) Path to training set CSV')
parser.add_argument('--validation_data',           default='../data/val.csv',   help='(string) Path to validation set CSV')
parser.add_argument('--load_cached',   action='store_true',                     help='Load cached versions of the training and validation')
parser.add_argument('--model_type',                                             help='(string) Name of model to train')
parser.add_argument('--model_dir',                 default='models/test',       help='(string) Path to store models and checkpoints')
parser.add_argument('--epochs',        type=int,   default=10,                  help='(int) Number of passes to make through the training set')
parser.add_argument('--batch_size',    type=int,   default=16,                  help='(int) Number of examples to evaluate for each optimizer step')
parser.add_argument('--learning_rate', type=float, default=1e-4,                help='(float) Optimizer learning rate')
parser.add_argument('--checkpoint',    type=int,   default=None,                help='(int) Checkpoint number to restart training from')

if __name__ == "__main__":
    args = parser.parse_args()

    print('Loading data...')
    print('- Training...')
    if args.load_cached:
        train_data = torch.load('../data/train_cached.pt')
    else:
        train_data = ChessDataset(args.train_data)
    
    print('- Validation...')
    if args.load_cached:
        validation_data = torch.load('../data/val_cached.pt')
    else:
        validation_data = ChessDataset(args.validation_data)

    print('Setting up models...')    
    feature_names = train_data.get_column_names()
    model = build_model(args.model_type, feature_names)
    model_dir = 'models/%s' % args.model_type

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    train(
        model, 
        loss_fn, 
        optimizer, 
        train_data, 
        validation_data, 
        args.epochs, 
        args.batch_size, 
        model_dir, 
        args.checkpoint,
    )
