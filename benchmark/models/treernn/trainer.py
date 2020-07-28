from tqdm import tqdm
from numpy.random import shuffle
import torch
import numpy as np 

class Trainer:
    
    def __init__(self, args, model, criterion, optimizer, device):
        
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):

        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = [i for i in range(len(dataset.trees))]
        shuffle(indices)
        length = 0
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, inputs, target = dataset[indices[idx]]
            inputs = inputs.to(self.device)
            target = torch.tensor([[target]])
            target = target.to(self.device)
            output = self.model(tree, inputs)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            length +=1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.epoch += 1
        return total_loss / length 

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            indices = [i for i in range(len(dataset.trees))] #torch.randperm(len(dataset), dtype=torch.long, device='cpu')
            predictions, targets = [], []
            length = 0
            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                tree, inputs, target = dataset[indices[idx]]
                if target != 0:
                    targets.append(target)
                    inputs = inputs.to(self.device)
                    target = torch.tensor([[target]])
                    target = target.to(self.device)
                    output = self.model(tree, inputs)
                    length +=1
                    predictions.append(output.item())
                    loss = self.criterion(output, target)
                    total_loss += loss.item()
        
        targets = [dataset.min_y + k*(dataset.max_y-dataset.min_y) for k in targets]
        targets = [np.exp(k) for k in targets]
        predictions = [dataset.min_y + k*(dataset.max_y-dataset.min_y) for k in predictions]
        predictions = [np.exp(k) for k in predictions]
        return total_loss / length, targets, predictions
