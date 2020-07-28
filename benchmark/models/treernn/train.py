import torch
import json
import torch.optim as optim

from benchmark.models.treernn.dataset import GaugeDataset
from benchmark.models.treernn.trainer import Trainer
from benchmark.models.treernn.treernn import QueryRNN
from benchmark.scripts.metrics import Summary
from benchmark.scripts.parser import Benchmark
from benchmark.scripts.tree import *


def read_benchmark(data_dir):
    benchmark = Benchmark(data_dir)
    trees = {}
    for q_name, q in benchmark.queries.items():
        q.build_exp_array()
        q_tree = constructTree(exp2Prefix(q.exp_array))
        q_tree.set_idx(0)
        trees[q_name] = q_tree
    return benchmark, trees


def train_test_val_split(benchmark, trees):
    train_q = [trees[q] for q in benchmark.train]
    train_y = [benchmark.labels[q] for q in benchmark.train]
    val_q = train_q[-10000:]
    val_y = train_y[-10000:]
    train_q = train_q[:-10000]
    train_y = train_y[:-10000]
    test_q = [trees[q] for q in benchmark.test]
    test_y = [benchmark.labels[q] for q in benchmark.test]

    return train_q, train_y, val_q, val_y, test_q, test_y


def q_loss(output, target):
    output += 1
    target += 1
    loss = torch.mean(output / target + target / output) / 2
    return loss


def mse_loss(output, target):
    return ((output - target) ** 2).mean()


def loss_alpha(alpha):
    def loss(output, target):
        return alpha * q_loss(output, target) + (1 - alpha) * mse_loss(output, target)
    return loss


class Args:
    def __str__(self):
        return json.dumps(self.__dict__)


def fill_args(lr, bs, epochs, cuda, input_dim, mem_dim, hidden_dim, alpha):
    args = Args()
    args.lr = lr
    # args.wd = 0.005#1e-4
    args.batchsize = bs
    args.n_epoch = epochs
    args.cuda = cuda
    args.input_dim = input_dim
    args.mem_dim = mem_dim
    args.hidden_dim = hidden_dim
    args.alpha = alpha
    return args


def build_training_enviroment(args):
    a = args
    model = QueryRNN(a.input_dim, a.mem_dim, a.hidden_dim)
    device = torch.device("cuda:0" if a.cuda else "cpu")
    model.to(device)
    criterion = loss_alpha(a.alpha)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()), lr=a.lr
                           )  # weight_decay=args.wd)
    return model, criterion, optimizer, device

class EarlyStoppingCallback:

    def __init__(self, stop_at_k, monitoring_metric):
        self.stop_at_k = stop_at_k
        self.monitoring_metric = monitoring_metric
        self.i = 0
        self.current_metrics = None
        self.previous_metrics = None

    def swap(self):
        self.current_metrics,self.previous_metrics = self.previous_metrics, self.current_metrics

    def need_early_stopping(self):

        if self.previous_metrics:
            current_val_loss = self.current_metrics['val_'+self.monitoring_metric]
            prev_val_loss = self.previous_metrics['val_'+self.monitoring_metric]

            current_train_loss = self.current_metrics['train_' + self.monitoring_metric]
            prev_train_loss = self.previous_metrics['train_' + self.monitoring_metric]
            return current_val_loss>prev_val_loss and current_train_loss<prev_train_loss

    def on_epoch_end(self, last_metrics):

        self.swap()
        self.current_metrics = last_metrics
        if self.need_early_stopping():
            self.i+=1
        else:
            self.i = 0
        return self.i == self.stop_at_k

from torch.utils.tensorboard import SummaryWriter

def main():
    DATA_DIR = 'data'
    print("Read benchmark...")
    print('''-----------------/\-----------------''')
    benchmark, trees = read_benchmark(DATA_DIR)
    print("Train/Val/Test split...")
    print('''-----------------/\-----------------''')
    train_q, train_y, val_q, val_y, test_q, test_y = train_test_val_split(benchmark, trees)
    # calculate min_y, max_y only for training dataset
    print("Initialize Pytorch datasets...")
    print('''-----------------/\-----------------''')

    n_ops = 6
    n_cols = 7
    n_values = 263
    input_dim = n_ops + n_cols + n_values

    train_dataset = GaugeDataset(train_q, train_y, input_dim, min_y=None, max_y=None)
    tr_min_y, tr_max_y = train_dataset.get_min_max_y()
    val_dataset = GaugeDataset(val_q, val_y, input_dim, tr_min_y, tr_max_y)
    test_dataset = GaugeDataset(test_q, test_y, input_dim, tr_min_y, tr_max_y)

    args = fill_args(lr=1e-4, bs=1, epochs=100, cuda=False, input_dim=input_dim, mem_dim=32, hidden_dim=80, alpha=0.9999)
    print("Hyperparameters...")
    print('''-----------------/\-----------------''')
    print(args)

    print("Creating Pytorch Model, Optimizer, Criterion...")
    print('''-----------------/\-----------------''')
    model, criterion, optimizer, device = build_training_enviroment(args)
    trainer = Trainer(args, model, criterion, optimizer, device)

    early_stopping = EarlyStoppingCallback(3, 'loss')

    PATH_TO_SAVE = 'models/treernn_best.pt'
    min_q_error = float("Inf")
    history = {}
    writer = SummaryWriter('best_result')

    for i in range(args.n_epoch):
        train_loss = trainer.train(train_dataset)
        val_loss, targets, predictions = trainer.test(val_dataset)
        val_summary = Summary(targets, predictions)
        history[i] = {'train_loss': train_loss, 'val_loss': val_loss, **val_summary.summary}
        writer.add_scalars('best m{}h{}'.format(args.mem_dim, args.hidden_dim), {'train_loss': train_loss, 'val_loss': val_loss, **val_summary.summary})

        q_error = val_summary.summary['interval_mae_err']
        if q_error < min_q_error:
           print('Saving models: interval error {} ...'.format(q_error))
           min_q_error = q_error
           torch.save(trainer.model.state_dict(), PATH_TO_SAVE)

        if early_stopping.on_epoch_end(history[i]):
            print('''-----------------/\-----------------''')
            print("Attention: Early Stopped")
            break

    json.dump(history, open("best_result.json", 'w'))
    writer.close()

if __name__ == "__main__":
    main()