# Bayes Classifier in Python
## Idea
I wanted to make a Bayes Classifier in Python from scratch. The goal of this was simply to do it and to practice my coding. Always practice all the time.

## Usage
The included test file works with the attached datasets and outlines how to use the module.

<b>Load the classifier with the data file information</b>

<code>data_file_name = "diabetes_prediction_dataset.csv"</code>
<code>X_labels = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]</code>
<code>y_label = "diabetes"</code>


<b>Instrantiate the classifier with the data information</b>

<code>from Classifier_Module import BayesClass</code>
<code>bayes_class_1 = BayesClass(data_file_name, X_labels, y_label)</code>



<b>Train the classifier</b>

<code>bayes_class_1.train_algorithm()</code>


<b>Make predictions and test them</b>

<code>y_preds = bayes_class_1.predict(bayes_class_1.dataset.return_test_data())</code>
<code>print(bayes_class_1.test_accuracy(bayes_class_1.dataset.y_test, y_preds))</code>

