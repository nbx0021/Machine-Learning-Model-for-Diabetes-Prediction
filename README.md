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
![model_training_1](https://github.com/user-attachments/assets/34869a31-37a9-4460-a233-ed5b6a910e2d)
![model_training_2](https://github.com/user-attachments/assets/f045d086-5bfd-461f-8b47-59a1725e097b)
![model_training_3](https://github.com/user-attachments/assets/d637e7b2-307b-4e2e-afa5-a824808af1c8)

### **Confusion Matrix:**
![confusion_matrix](https://github.com/user-attachments/assets/41096d81-5ee6-4394-8237-3989ad890d6d)

### **ROC Curve:**
![ROC curves](https://github.com/user-attachments/assets/5ff7e50c-f60a-406e-a032-33d445391752)


### **Cross-validation results:**
![cross_validation](https://github.com/user-attachments/assets/53f927eb-4800-4db8-8084-21338085bd3f)


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


