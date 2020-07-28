import numpy as np


class Summary(object):
    def __init__(self, y_true, y_pred):

        rmse = self.rmse(y_true, y_pred)
        mae = self.mae(y_true, y_pred)
        q_err = self.q_error(y_pred, y_true)
        interval_mae_err = self.interval_mae_error(y_pred, y_true)
        interval_acc = self.interval_acc(y_pred, y_true)
        self.summary = {"rmse": rmse,
                        "mae": mae,
                        "q_error": q_err,
                        "interval_mae_err": interval_mae_err,
                        "interval_acc": interval_acc}

    def rmse(self, y_pred, y_true):
        return np.sqrt(np.mean(np.square([a - b for a, b in zip(y_pred, y_true)])))

    def mae(self, y_pred, y_true):
        return np.mean(np.abs([a - b for a, b in zip(y_pred, y_true)]))

    def q_error(self, y_pred, y_true):
        return np.mean([(max(a, b) + 1) / (min(a, b) + 1) for a, b in zip(y_pred, y_true)])

    def interval_mae_error(self, y_pred, y_true):
        bins = np.array([50, 500, 1000, 2500, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 250000, 400000, 600000])
        pred_inds = np.digitize(y_pred, bins)
        true_inds = np.digitize(y_true, bins)
        return self.mae(pred_inds, true_inds)

    def interval_acc(self, y_pred, y_true):
        bins = np.array([50, 500, 1000, 2500, 5000, 10000, 25000, 50000, 75000, 100000, 150000, 250000, 400000, 600000])
        pred_inds = np.digitize(y_pred, bins)
        true_inds = np.digitize(y_true, bins)
        s = sum([int(i == j) for i, j in zip(pred_inds, true_inds)])
        return s / len(pred_inds)

    def __str__(self):
        to_print = "Model {} Summary: (\n".format(self.summary.get('model', 'temp'))
        for metric, value in self.summary.items():
            if metric != 'model':
                to_print += "\t{}: ({:.5f})\n".format(metric, value)
        to_print += ")"
        return to_print

    def __repr__(self):
        return self.__str__()
