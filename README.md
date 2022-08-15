# KNN realization

Educational project where I will predict the photographed numbers using the KNN method.
1. Implementation of K-nearest neighbor classifier by hand.
2. Hyperparameter selection with cross-validation.
## Installation
1. Run the `download_data.sh` file to download the data we will use for training.

2. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install -r requirements.txt
```

## Description
1. The image is converted to a vector and the Manhattan distance is calculated.
In the knn.py  3 functions for distance calculation are implemented,, when applying, you can choose any of them.

2. Separate functions for prediction of binary classification (only for 1 or 0) and multiclass classification are implemented

3. Functions for calculating multiclass and binary metrics

## Results

binary: F1 score = 0.3644

multiclass: Accuracy = 0.34

For the method applied to such a problem, the results are quite high :)