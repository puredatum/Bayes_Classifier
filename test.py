from Classifier_Module import BayesClass
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Selecting labels from dataset
data_file_name = "diabetes_prediction_dataset.csv"
X_labels = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
y_label = "diabetes"


# Instantiate the classes
bayes_class_1 = BayesClass(data_file_name, X_labels, y_label)

# Train the algorithm
bayes_class_1.train_algorithm()

# Display the testing results
y_preds = bayes_class_1.predict(bayes_class_1.dataset.return_test_data())
print(bayes_class_1.test_accuracy(bayes_class_1.dataset.y_test, y_preds))

"""-----------------------"""

# Sklearn bayes classifier
gnb = GaussianNB()  # Instantiate the classifier

# Fit the model
gnb.fit(bayes_class_1.dataset.X_train, bayes_class_1.dataset.y_train)  # Train dataset

# Run predictions and test
y_pred = gnb.predict(bayes_class_1.dataset.return_test_data())  # Generate predictions
print(accuracy_score(bayes_class_1.dataset.y_test, y_pred))

"""-----------------------"""

# Selecting labels from dataset
data_file_name = "cardio_train.csv"
X_labels = ["age", "gender", "height", "weight", "ap_hi", 
			"ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
y_label = "cardio"
delimiter_csv = ";"


# Instantiate the classes
bayes_class_2 = BayesClass(data_file_name, X_labels, y_label, delimiter_csv)

# Train the algorithm
bayes_class_2.train_algorithm()

# Display the testing results
y_preds = bayes_class_2.predict(bayes_class_2.dataset.return_test_data())
print(bayes_class_2.test_accuracy(bayes_class_2.dataset.y_test, y_preds))

"""-----------------------"""

# Sklearn bayes classifier
gnb = GaussianNB()  # Instantiate the classifier

# Fit the model
gnb.fit(bayes_class_2.dataset.X_train, bayes_class_2.dataset.y_train)  # Train dataset

# Run predictions and test
y_pred = gnb.predict(bayes_class_2.dataset.return_test_data())  # Generate predictions
print(accuracy_score(bayes_class_2.dataset.y_test, y_pred))
