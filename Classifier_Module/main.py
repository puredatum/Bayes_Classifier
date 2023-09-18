from dataclasses import dataclass
import numpy as np
from .functions import DataWrangler, BayesClassifier


@dataclass
class BayesClass:
	dataset_file_name: str
	X_labels: list
	y_label: str
	delim: str = ","

	# Loads and manages the dataset, then sets up the bayes classifier class
	def __post_init__(self):
		self.dataset = DataWrangler(
			self.dataset_file_name, self.X_labels, self.y_label, self.delim)
		self.bayes_classifier = BayesClassifier()

	# Trains the algorithm
	def train_algorithm(self):
		self.bayes_classifier.train_dataset(*self.dataset.return_train_data())

	# Run prediction, usues the supplied test data if none specified
	def predict(self, test_data=None):
		if test_data is None:
			self.pred = self.bayes_classifier.predict(test_data)
		else:
			self.pred = self.bayes_classifier.predict(self.dataset.return_test_data())

		return self.pred

	# Testing accuracy of the predictions
	def test_accuracy(self, y_test, y_pred):
		accuracy = np.sum(y_test == y_pred) / len(y_test)

		return accuracy