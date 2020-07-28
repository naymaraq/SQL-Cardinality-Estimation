import torch
import torch.optim as optim
from tqdm import tqdm 
import numpy as np

from benchmark.models.treernn.treernn import QueryRNN
from benchmark.models.treernn.featurize import index
from benchmark.scripts.query import GaugeQuery
from benchmark.scripts.tree import *

class TreeRNNInference:

    def __init__(self, model_path, state):

        self.min_y = 0
        self.max_y = 12.524984453533651
        self.load_model(model_path, state)

    def decode(self, k):
        return np.exp(self.min_y + k*(self.max_y-self.min_y))

    def load_model(self, model_path, state):

        self.input_dim = state["input_dim"]
        self.mem_dim = state["mem_dim"]
        self.hidden_dim = state["hidden_dim"]

        #PATH = 'models/treernn_50K_best_0_bias_on_test.pt'
        self.cardinality_model = QueryRNN(self.input_dim, self.mem_dim, self.hidden_dim)
        self.cardinality_model.load_state_dict(torch.load(model_path))
        

    def estimate(self, q):

        self.cardinality_model.eval()
        q.build_exp_array()
        q_tree = constructTree(exp2Prefix(q.exp_array))
        q_tree.set_idx(0)
        traverse = q_tree.traverse_by_values()
        inputs = [torch.tensor([0.]*self.input_dim) if p_ in ['|',"&"] else torch.tensor(index.featurize(p_)) for p_ in traverse]
        inputs = torch.stack(inputs)
        output = self.cardinality_model(q_tree, inputs).item()
        res = self.decode(output)
        if res<=1:
            return 0
        return int(res)

