import torch
import torch.nn as nn
import torch.nn.functional as F


class ChildSumTreeRNN(nn.Module):

    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeRNN, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.and_ = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
        self.or_  = nn.Linear(self.mem_dim, self.mem_dim, bias=False)

    def and_op(self, child_h):

        outs = self.and_(child_h)
        outs = torch.tanh(outs)
        out = torch.mean(outs, dim=0, keepdim=True)#**3
        return out 

    def or_op(self, child_h):

        outs = self.or_(child_h)
        outs = torch.tanh(outs)
        out = torch.mean(outs, dim=0, keepdim=True)#**3
        return out 

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        
        if tree.num_children == 0:
            child_h = inputs[tree.idx]
            tree.state = child_h.view(1,-1)
        else:
            child_h = tuple(map(lambda x: x.state.view(1,-1), tree.children))
            child_h = torch.cat(child_h, dim=0)
        
        if tree.value == '&':
            tree.state = self.and_op(child_h)
        elif tree.value == '|':
            tree.state = self.or_op(child_h)

        return tree.state


class RegressorNN(nn.Module):
    def __init__(self, mem_dim, hidden_dim):

        super(RegressorNN, self).__init__()

        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.wh_1 = nn.Linear(self.mem_dim, self.hidden_dim, bias=True)
        self.wh_2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.wh_3 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.wp = nn.Linear(self.hidden_dim, 1, bias=True)

    def forward(self, h_state):

        out = torch.relu(self.wh_1(h_state))
        out = torch.relu(self.wh_2(out))
        out = torch.relu(self.wh_3(out))
        out = self.wp(out)
        return out



class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):

        super(MLP, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.wh_1 = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        self.wh_2 = nn.Linear(self.hidden_dim, 3*self.hidden_dim, bias=False)
        self.wh_3 = nn.Linear(3*self.hidden_dim, 2*self.hidden_dim, bias=False)
        self.wp = nn.Linear(2*self.hidden_dim, hidden_dim, bias=False)

    def forward(self, inp):
        out = torch.tanh(self.wh_1(inp))
        out = torch.tanh(self.wh_2(out))
        out = torch.tanh(self.wh_3(out))

        out = torch.tanh(self.wp(out)) #todo after normalization use these
        return out


class QueryRNN(nn.Module):
    
    def __init__(self, input_dim,
                       mem_dim, 
                       hidden_dim):

        super(QueryRNN, self).__init__()
        self.mlp = MLP(input_dim, mem_dim)
        self.treelstm = ChildSumTreeRNN(mem_dim, mem_dim)
        self.regressor = RegressorNN(mem_dim, hidden_dim)

    def forward(self, tree, inputs):
        input_embs = self.mlp(inputs)
        h = self.treelstm(tree, input_embs)
        output = self.regressor(h)
        return output


