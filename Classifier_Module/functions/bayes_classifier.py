import numpy as np
from dataclasses import dataclass, field


@dataclass
class BayesClassifier:

    # Import dataset
    def _import_dataset(self, X_train, y_train):
        # Bring in dataset
        self.X_train = X_train
        self.y_train = y_train

    # Setup the datasets
    def _setup_dataset(self):
        # Build dataset information
        self._n_samples, self._n_features = self.X_train.shape
        self._classes = np.unique(self.y_train)
        self._n_classes = len(self._classes)

        # Build array for the statistics
        self._mean = np.zeros((self._n_classes, self._n_features), dtype=np.float64)
        self._var = np.zeros((self._n_classes, self._n_features), dtype=np.float64)
        self._priors = np.zeros(self._n_classes, dtype=np.float64)

    # Training the bayes classifier
    def train_dataset(self, X_train, y_train):
        self._import_dataset(X_train, y_train)
        self._setup_dataset()

        # Training the dataset
        for c in self._classes:
            self._mean[c] = self.X_train[self.y_train == c].mean().values
            self._var[c] = self.X_train[self.y_train == c].var().values
            self._priors[c] = \
                self.X_train[self.y_train == c].shape[0] / float(self._n_samples)

    # Prediction for an array X of vectors x and converting to a numpy array
    def predict(self, X):
        y_pred = np.zeros((len(X)), dtype=np.float64)

        for ind, x in enumerate(X.to_numpy()):
            y_pred[ind] = self._predict_helper(x)

        return y_pred

    # Helper function to run prediction for each x in X
    def _predict_helper(self, x):
        posteriors = []

        for c in self._classes:
            prior = np.log(self._priors[c])
            class_conditional = np.sum(np.log(self._class_pdf(c, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    # PDF to model data using a gaussian distribution
    def _class_pdf(self, class_id, x):
        class_mean = self._mean[class_id]
        class_var = self._var[class_id]
        pdf_numerator = np.exp(-1 * (x-class_mean)**2 / (2 * class_var**2))
        pdf_denominator = np.sqrt(2 * np.pi * class_var)

        return pdf_numerator / pdf_denominator
