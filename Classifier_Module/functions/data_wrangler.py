import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field


@dataclass
class DataWrangler:
    dataset_name: str = ""
    x_labels: list = field(default_factory = list)
    y_label: str = ""
    delim: str = ","

    # Load the data
    def __post_init__(self):
        self._data_in = pd.read_csv(self.dataset_name, delimiter=self.delim).dropna()
        self.data_out_x = self._data_in[self.x_labels]
        self.data_out_y = self._data_in[self.y_label]
        self._split_data()

    # Train test split
    def _split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.data_out_x, self.data_out_y, test_size=0.2)

    def return_train_data(self):
        return self.X_train, self.y_train

    def return_test_data(self):
        return self.X_test

    def return_test_y(self):
        return self.y_test
