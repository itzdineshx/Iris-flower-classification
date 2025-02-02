# Iris Flower Classification: A Comparison of KNN, SVM, and Logistic Regression

## **Overview**
This project demonstrates the classification of the **Iris Flower dataset** using three different machine learning algorithms: **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **Logistic Regression**. The dataset contains measurements of four features (sepal length, sepal width, petal length, petal width) for three species of Iris flowers: **Setosa**, **Versicolor**, and **Virginica**. The goal of this project is to explore and compare the performance of these models on a classification task.

## **Dataset**
- **Name:** Iris dataset
- **Source:** [Scikit-learn](https://scikit-learn.org/)
- **Number of Samples:** 150
- **Features:** 
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target Classes:**
  - **Setosa**
  - **Versicolor**
  - **Virginica**

## **Project Goals**
- Compare the performance of **KNN**, **SVM**, and **Logistic Regression** models on the Iris dataset.
- Evaluate the models using classification metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.
- Perform data preprocessing, including **standardization** to scale the features.

## **Model Comparison**
1. **K-Nearest Neighbors (KNN):** A non-parametric, instance-based learning algorithm that classifies a new sample based on the majority class of its nearest neighbors.
2. **Support Vector Machine (SVM):** A supervised machine learning model that constructs hyperplanes to classify data points.
3. **Logistic Regression:** A linear model used for binary and multiclass classification, applying the logistic function to predict class probabilities.

## **Project Structure**
```
Iris-Flower-Classification/
├── data/
│   └── iris.csv (if you choose to include a CSV of the dataset)
├── notebooks/
│   └── iris_classification.ipynb (Main Jupyter notebook with code and analysis)
├── requirements.txt (List of Python dependencies)
└── README.md (This file)
```

## **Installation**
Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/Iris-Flower-Classification.git
cd Iris-Flower-Classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## **Dependencies**
- **Scikit-learn:** A library for machine learning in Python.
- **Pandas:** Data manipulation and analysis library.
- **NumPy:** A library for numerical computations.
- **Matplotlib:** A plotting library for creating static, animated, and interactive visualizations.
- **Seaborn:** A data visualization library based on Matplotlib that provides a high-level interface for drawing attractive statistical graphics.

To install these libraries, run:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## **Usage**
1. Open the Jupyter notebook `iris_classification.ipynb`.
2. Follow the steps in the notebook to load the dataset, perform data preprocessing, and train the models.
3. Evaluate the models using various classification metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.
4. View the performance comparison between **KNN**, **SVM**, and **Logistic Regression**.

## **Results**
The models were evaluated on the Iris dataset with the following classification reports:

### **KNN Classification Report**
```
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

### **SVM Classification Report**
```
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.89      0.94         9
   virginica       0.92      1.00      0.96        11

    accuracy                           0.97        30
   macro avg       0.97      0.96      0.97        30
weighted avg       0.97      0.97      0.97        30
```

### **Logistic Regression Classification Report**
```
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

## **Conclusion**
- **KNN** and **Logistic Regression** achieved perfect accuracy of 100%.
- **SVM** showed excellent performance with an accuracy of 97%, though its recall for the "Versicolor" class was slightly lower.
- All models performed well on the small and simple Iris dataset, but **KNN** and **Logistic Regression** were slightly better in this case.

## **Future Work**
- **Hyperparameter Tuning:** Further tuning of model parameters, such as the number of neighbors for KNN or the kernel type for SVM, could improve model performance.
- **Cross-validation:** Implementing **cross-validation** for more reliable model evaluation.
- **Advanced Models:** Exploring more complex models like **Random Forests** or **Gradient Boosting**.

## **Author**
DINESH S
[Your LinkedIn Profile]([your-linkedin-link](https://www.linkedin.com/in/dinesh-x/)) | [GitHub Profile]([your-github-link](https://github.com/itzdineshx/)

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

