import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
 
from dataset import get_dataloader
from models import StockfishScoreModel, LinearMovesModel
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

def train(model, loss_fn, optimizer, train_data, validation_data, num_epochs):
    """

    """
    # TODO: make it possible to restore training from checkpoint
    # See https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/nlp/train.py#L104:
    # if restore_file is not None:
    #     restore_path = os.path.join(
    #         args.model_dir, args.restore_file + '.pth.tar')
    #     logging.info("Restoring parameters from {}".format(restore_path))
    #     utils.load_checkpoint(restore_path, model, optimizer)



    for epoch in range(num_epochs):
        print('Starting epoch %d/%d' % (epoch+1, num_epochs))

        # Train for one epoch
        print('Training...')
        training_loss = train_epoch(model, loss_fn, optimizer, train_data)
        print('Training loss: %.4f' % training_loss)

        # Validate
        print('Validating...')
        validation_metrics = {
            'loss': nll,
            'accuracy': accuracy,
        }
        validation_results = evaluate(model, validation_data, validation_metrics)

        for metric in validation_results:
            print('- %s: %.4f' % (metric, validation_results[metric]))

        # TODO: check if this is the best model and save if it is
        torch.save(model.state_dict(), 'models/%d.pt' % (epoch+1))

        # TODO: save checkpoint here



if __name__ == "__main__":
    # TODO: load proper training/validation data
    print('Loading training data...')
    train_generator = get_dataloader('../data/dataset_subset.csv')
    
    print('Loading validation data...')
    validation_generator = get_dataloader('../data/dataset_subset.csv')

    print('Setting up models...')
    # model = LinearMovesModel(16)
    model = StockfishScoreModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    train(model, loss_fn, optimizer, train_generator, validation_generator, 200)


# TODO: add command line interface
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', default='../../data/sample_dataset.csv',
#                     help="Path to dataset CSV")
# args = parser.parse_args()
# print(args.data_path)
