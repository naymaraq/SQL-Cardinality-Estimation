from os.path import join
from benchmark.scripts.query import GaugeQuery


class Benchmark:
    def __init__(self, benchmark_folder):
        self.b_f = benchmark_folder
        self._read_benchmark()

    def _read_benchmark(self):
        self.train = open(join(self.b_f, "train.txt")).read().rstrip().split('\n')
        self.test = open(join(self.b_f, "test.txt")).read().rstrip().split('\n')
        self.queries = open(join(self.b_f, "queries.txt")).read().rstrip().split('\n')
        self.queries = {row.split('\t')[0]: GaugeQuery.from_string(row.split('\t')[1]) for row in self.queries}
        self.labels = open(join(self.b_f, "labels.txt")).read().rstrip().split('\n')
        self.labels = {row.split('\t')[0]: int(row.split('\t')[1]) for row in self.labels}
