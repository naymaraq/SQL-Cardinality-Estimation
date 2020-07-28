import time
import numpy as np
from benchmark.scripts.parser import Benchmark
from benchmark.scripts.metrics import Summary
from benchmark.scripts.query import GaugeQuery


def acc(o, r, op_type):
    acc = 0
    if op_type == "||":
        for i in range(len(o)):
            if o[i] >= r[i]:
                acc += 1

    elif op_type == "&&":
        for i in range(len(o)):
            if o[i] <= r[i]:
                acc += 1

    acc /= len(o)
    acc *= 100
    return acc


class Score:
    
    def __init__(self, benchmark_path):

        self.benchmark = Benchmark(benchmark_path)
        self.summary = {}

    def eval(self, model, test_or_train):

        y_pred, y_true = [],[]
        time_history = []
        queries = getattr(self.benchmark, test_or_train)

        short_qs, middle_qs, long_qs  = [],[],[]
        for q in queries:
            if len(self.benchmark.queries[q])<=5:
                short_qs.append(q)
            elif len(self.benchmark.queries[q])<=12:
                middle_qs.append(q)
            else:
                long_qs.append(q)

        overall_summary = {}
        global_y_true = []
        global_y_pred = []
        for q_bag, q_name in zip([short_qs, middle_qs, long_qs], ['short', 'middle', 'long']):
            l = 0
            y_pred, y_true = [],[]
            for q in q_bag:
                start = time.time()
                if self.benchmark.labels[q] == 0:
                    continue

                pred = model.estimate(self.benchmark.queries[q])
                time_history.append(time.time()-start)
                l+=1
                y_pred.append(pred)
                y_true.append(self.benchmark.labels[q])
                global_y_true.append(self.benchmark.labels[q])
                global_y_pred.append(pred)

            avg_inference_time = sum(time_history)/len(time_history)
            res = Summary(y_pred,y_true)
            res.summary["model"] = type(model).__name__
            res.summary['n_samples'] = l#len(queries)
            res.summary["avg_inference_time"] = avg_inference_time
            overall_summary[q_name] = res

        global_summary = Summary(global_y_pred,global_y_true)
        global_summary.summary["model"] = type(model).__name__
        global_summary.summary['n_samples'] = len(global_y_pred)
        #global_summary.summary["avg_inference_time"] = avg_inference_time
        overall_summary['avg'] = global_summary
        return overall_summary

    def eval_train(self, model):
        return self.eval(model, 'train')

    def eval_test(self, model):
        return self.eval(model, 'test')
        
    def __call__(self, model):

        test_res = self.eval_test(model)
        return test_res
