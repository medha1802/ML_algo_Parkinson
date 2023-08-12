# ML_algo_Parkinson

Cascade Forest Classifier for Early Detection of Parkinson's Disease

This repository contains a Python implementation of a Cascade Forest Classifier enhanced with Multi-Grained Data Processing. The code preprocesses audio data from Parkinson's disease patients, performs multi-grained scanning, and trains a cascade forest ensemble for early detection of the disease.

**Introduction**
The Cascade Forest Classifier is a powerful ensemble learning technique that combines the strengths of random forests and cascading classifiers. In this project, the classifier is applied to the early detection of Parkinson's disease. Parkinson's disease is a brain disorder that causes unintended or uncontrollable movements, such as shaking, stiffness, and difficulty with balance and coordination.

**Installation**
To use the Cascade Forest Classifier for Early Detection of Parkinson's Disease, follow these steps:

Clone the repository:

git clone https://github.com/medha1802/ML_algo_Parkinson.git

Navigate to the repository:

cd your-repo

Install required packages using pip:

pip install -r requirements.txt

**Usage**
To use the provided code, follow these steps:

Prepare your audio dataset in a suitable format.

Update the file paths and parameters in the code to match your data.

Run the code using Python:

python main.py

Review the printed results, which include train and test accuracies at each cascade level.

**Dependencies**
The code depends on the following libraries:

Python 3. x
NumPy
pandas
sci-kit-learn

Install these dependencies using the provided requirements.txt file.

**How It Works**
The code performs the following steps:

1. Reading and preprocessing audio data from CSV files.
2. Label encoding categorical features.
3. Multi-grained scanning of input audio data using various sliding ratios.
4. Training a cascade forest ensemble using multi-grained data for early Parkinson's disease detection.
5. Evaluating the model on test audio data and calculating accuracies at each cascade level.
6. For a detailed overview of the code's functionality, refer to the How It Works section in the repository.

**Results**
The repository's code demonstrates the effectiveness of the Cascade Forest Classifier with Multi-Grained Data Processing for early detection of Parkinson's disease. Detailed results, including train and test accuracies, can be observed in the printed output after running the code.

**Future Updates**
This project is an ongoing effort to enhance the accuracy of Parkinson's disease detection. Future updates may include implementing additional algorithms to achieve even better results.
