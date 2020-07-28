
class Ensemble:
	def __init__(self, models):
		self.models = models

	def estimate(self, q):

		s = sum([m.estimate(q) for m in self.models])
		return s/len(self.models)