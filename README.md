# Bayes Classifier in Python
## Idea
I wanted to make a Bayes Classifier in Python from scratch. The goal of this was simply to do it and to practice my coding. Always practice all the time.

## Usage
The included test file works with the attached datasets and outlines how to use the module.

<b>Load the classifier with the data file information</b>
<code>
# Selecting labels from dataset
data_file_name = "diabetes_prediction_dataset.csv"
X_labels = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
y_label = "diabetes"


# Instantiate the classes
bayes_class_1 = BayesClass(data_file_name, X_labels, y_label)
</code>

   
3. Train the classifier
4. Make predictions
5. Test the results
