# **Machine Learning Model for Diabetes Prediction**

This repository contains a machine learning project for predicting diabetes based on various health-related attributes. The models included in this project are trained on a dataset with attributes such as **Glucose**, **BloodPressure**, **BMI**, **Age**, and others to predict the presence of diabetes.

### **Table of Contents**
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Performance Evaluation](#performance-evaluation)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

---

## **Introduction**

The goal of this project is to build a machine learning model that predicts whether a person has diabetes based on a set of features. This project utilizes several machine learning algorithms to classify the data:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**

The model is evaluated based on several metrics such as accuracy, precision, recall, and AUC-ROC.

---

## **Dataset**

The dataset used in this project contains the following features:

- **Pregnancies**: Number of pregnancies
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg / height in m²)
- **DiabetesPedigreeFunction**: A function that scores likelihood of diabetes based on family history
- **Age**: Age in years
- **Outcome**: 1 if the person has diabetes, 0 otherwise

The dataset is split into training and testing sets.

---

## **Models Used**

### **1. Logistic Regression**
- A logistic regression model is used to predict the probability of the target variable, **Outcome** (diabetes or not).
  
### **2. Decision Tree**
- A decision tree model classifies data by creating decision nodes based on input features.

### **3. Random Forest**
- A random forest model is an ensemble learning technique that uses multiple decision trees to improve prediction accuracy.

### **4. K-Nearest Neighbors (KNN)**
- KNN is a non-parametric method used for classification by finding the majority class among the nearest neighbors.

### **5. Support Vector Machine (SVM)**
- SVM is a powerful classifier that creates hyperplanes to separate different classes in a high-dimensional space.

---

## **Performance Evaluation**

The models are evaluated based on several metrics:

1. **Accuracy**
2. **Precision**
3. **Recall**
4. **F1-Score**
5. **AUC-ROC**

Each model's confusion matrix and classification report are generated to assess performance.

---

## **Screenshots**

Below are some screenshots of the model's output and performance evaluation.

### **Model Training:**
![Model Training](screenshots/model_training.png)

### **Confusion Matrix:**
![Confusion Matrix](screenshots/confusion_matrix.png)

### **ROC Curve:**
![ROC Curve](screenshots/roc_curve.png)

### **Cross-validation results:**
![Cross-validation](screenshots/cross_validation.png)

---

## **Installation**

To run this project on your local machine, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Install dependencies:**

   Make sure you have Python 3.x installed. You can install the required libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Python script:**

   After installing the required dependencies, you can run the script to train and evaluate the models.

   ```bash
   python diabetes_model.py
   ```

---

## **Usage**

After training the models, you can test them with your own input data by modifying the `input_data` in the script or by providing new data in the appropriate format.

```python
input_data = [/* Enter data in the format */]
prediction = model.predict(input_data)
print(prediction)
```

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### **Adding Screenshots to the Repository**

1. Create a `screenshots` directory in the root of your project.
2. Place all your relevant screenshots (e.g., **model_training.png**, **confusion_matrix.png**, **roc_curve.png**) in that directory.
3. In the README file, refer to these images using markdown like so:

```markdown
![Model Training](screenshots/model_training_1.png)
![Model Training](screenshots/model_training_2.png)
![Model Training](screenshots/model_training_3.png)
```

