## Iris Flower Classification

This project focuses on developing a machine learning model to classify Iris flowers into three species based on sepal and petal measurements. The task is to accurately predict the species (Iris-setosa, Iris-versicolor, or Iris-virginica) using the famous Iris dataset.

## Task Objectives:
1.Preprocess the dataset, including feature scaling and normalization.
2.Train multiple machine learning models (Random Forest, Decision Tree, and SVC).
3.Evaluate model performance using accuracy, precision, recall, and F1-score.
4.Identify significant features that influence flower classification.
5.Achieve high species classification accuracy and present results.

## Dataset:
The dataset contains 150 samples with the following features:

1.Sepal length (cm)

2.Sepal width (cm)

3.Petal length (cm)

4.Petal width (cm)

5.Species: Target variable (Iris-setosa, Iris-versicolor, Iris-virginica).

## Running the Project:
1. Clone the repository:
          git clone https://github.com/yourusername/iris-flower-classification.git

2. Install dependencies:
You will need Python 3 and the following Python libraries to run the project:

          1.pandas

          2.numpy
   
          3.matplotlib
   
          4.seaborn
   
          5.scikit-learn
   
          6.jupyter

You can install them using the requirements.txt file (if provided), or manually:

          pip install pandas numpy matplotlib seaborn scikit-learn jupyter
          
3. Run the Jupyter Notebook:

          jupyter notebook code/iris_classification.ipynb

This notebook contains the entire project code, including data preprocessing, model training, evaluation, and feature importance analysis.

4. Dataset Location:

Make sure that the IRIS.csv dataset is located in the dataset/ folder. If necessary, update the dataset path in the notebook as:

          df = pd.read_csv("../dataset/IRIS.csv")
   
## Model Evaluation:
Several machine learning models have been trained, and the best-performing model is reported with high accuracy for classifying Iris species. The evaluation metrics include:

          1.Accuracy
          
          2.Precision
          
          3.Recall
          
          4.F1-Score
          
These metrics are discussed in detail within the notebook.

## Feature Importance:
The project also includes analysis of the most significant features that influence the Iris species classification, providing insights into which sepal and petal measurements are most relevant.

## Conclusion:
This project demonstrates a simple yet effective application of machine learning to classify flower species using a well-known dataset. The models perform with high accuracy, and the results are thoroughly evaluated and explained.
