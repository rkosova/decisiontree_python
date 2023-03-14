# Decision Tree Classifier

This is a Python implementation of a decision tree classifier. The implementation uses binary tree data structure to build and traverse the decision tree.

## Installation

Clone the repository to your local machine:

`$ git clone https://github.com/rkosova/decisiontree_python.git`

Install required packages:

`$ pip install -r requirements.txt`

## Usage

This decision tree model is trained when instantiated via the Tree class. It takes, as parameters, its hyperparameters:

`maxDepth` - Maximum depth of the tree.

`minLabels` - Minimum number of label members for a leaf split.

`maxImpurity` - Maximum acceptable impurity for a leaf split.  
  
```Python
from decisiontree import Tree

tree = Tree(X=X_train, y=y_train, maxDepth=5, minLabels=5, maxImpurity=0.1)

y_pred = tree.predict(X=X_test)
```

## Example

```Python
from decisiontree import Tree
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# generate toy dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create decision tree instance which also trains it)
dt = Tree(X_train=X_train, y_train=y_train, maxDepth=12, minLabels=35, maxImpurity=0.1)

# predict class labels for test data
y_pred = dt.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Note

This project is not meant to be a better tool than what's already out there. This project was a way for me to learn more about machine learning models in the back-end
