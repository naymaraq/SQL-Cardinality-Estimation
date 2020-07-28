from benchmark.models.histogram.histogram import HistogramEstimator
from benchmark.models.hybrid.ensemble import Ensemble
from benchmark.models.treernn.treernn_inference import TreeRNNInference
from benchmark.scripts.score import Score

# random_model = Random()

state = {"input_dim": 6 + 7 + 263, "mem_dim": 32, "hidden_dim": 80}
treernn_model = TreeRNNInference('models/treernn_best.pt', state=state)
hist_est = HistogramEstimator('data/stats.json')
ensemble = Ensemble([treernn_model, hist_est])

if __name__ == "__main__":
    sc = Score('data')
    print(sc(treernn_model))
    print(sc(hist_est))
    #print(sc(ensemble))
