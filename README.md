# Decision Tree Classifier & Random Forest Classifier

This is a Python implementation of a decision tree classifier. The implementation uses binary tree data structure to build and traverse the decision tree.

The same Tree objects are then used to create Random Forests. The forests are trained parallely.

## Installation

Clone the repository to your local machine:

`$ git clone https://github.com/rkosova/decisiontree_python.git`

Install required packages:

`$ pip install -r requirements.txt`

## Usage

### Decision Tree

This decision tree model is trained when instantiated via the Tree class. It takes, as parameters, its hyperparameters:

`maxDepth` - Maximum depth of the tree.

`minLabels` - Minimum number of label members for a leaf split.

`maxImpurity` - Maximum acceptable impurity for a leaf split.  
  
```Python
from decisiontree import Tree

tree = Tree(X=X_train, y=y_train, maxDepth=5, minLabels=5, maxImpurity=0.1)

y_pred = tree.predict(X=X_test)
```

### Decision Tree Example

```Python
from decisiontree import Tree
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# generate toy dataset
X, y = make_classification(n_samples=250, n_features=10, n_classes=2, random_state=42)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert to pandas Dataframe (currently, the only type decisiontree.Tree accepts)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train, name='Label')
y_test = pd.Series(y_test, name='Label')

# create decision tree instance which also trains it)
dt = Tree(X_train, y_train, maxDepth=10, minLabels=30, maxImpurity=0.15 )

# # predict class labels for test data
y_pred = dt.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

Tree generated by this sample code:

```Python
8 >=1.1634157882836547
----predict 0
----8 <=-0.9493959589379786
--------predict 1
--------3 <=-0.7575629815516245
------------predict 0
------------8 <=-0.26061444278444235
----------------0 <=1.9172914148627807
--------------------predict 1
--------------------predict 1
----------------3 >=-0.3008292027124013
--------------------5 >=-1.5763921569842494
------------------------predict 1
------------------------predict 0
--------------------predict 0
Accuracy: 0.82
```

(left neighbour on top, accessed through parent condition)

#### Saving Trees to JSON Files

Trees can also be saved by passing the output of the `toDict` Tree method to the `toJSON` Tree method:

```Python
dt.toJSON()
```

This will dump the JSON version of the tree into a file. The name of the file is the data and time of creation unless a `filename` parameter is passed to `toJSON`.

This tree generated (beautified):

```JSON
{
    "feature": 8,
    "value": 1.1634157882836547,
    "optimalSplitDirection": "right",
    "left": {
        "value": 0
    },
    "right": {
        "feature": 8,
        "value": -0.9493959589379786,
        "optimalSplitDirection": "left",
        "left": {
            "value": 1
        },
        "right": {
            "feature": 3,
            "value": -0.7575629815516245,
            "optimalSplitDirection": "left",
            "left": {
                "value": 0
            },
            "right": {
                "feature": 8,
                "value": -0.26061444278444235,
                "optimalSplitDirection": "left",
                "left": {
                    "feature": 0,
                    "value": 1.9172914148627807,
                    "optimalSplitDirection": "left",
                    "left": {
                        "value": 1
                    },
                    "right": {
                        "value": 1
                    }
                },
                "right": {
                    "feature": 3,
                    "value": -0.3008292027124013,
                    "optimalSplitDirection": "right",
                    "left": {
                        "feature": 5,
                        "value": -1.5763921569842494,
                        "optimalSplitDirection": "right",
                        "left": {
                            "value": 1
                        },
                        "right": {
                            "value": 0
                        }
                    },
                    "right": {
                        "value": 0
                    }
                }
            }
        }
    }
} 
```

#### Reading Trees from JSON Files

To read trees from previously generated JSON files:

```Python
from decisiontree import Tree

dt = Tree.fromJSON("filename.json")
```

### Random Forest

Takes the same parameters as the Decision Tree only with the addition of `n_trees` which dictates how many trees will be in the forest.

### Random Forest Example

```Python
from decisiontree import Tree
from randomforest import Forest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# generate toy dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert to pandas Dataframe (currently, the only type decisiontree.Tree accepts)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.Series(y_train, name='Label')
y_test = pd.Series(y_test, name='Label')

# create random forest instance (which also trains it)
forest = Forest(X_train, y_train, n_trees=15, maxDepth=20, minLabels=17, maxImpurity=0.15)

# predict class labels for test data
y_pred = forest.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Note

This project is not meant to be a better tool than what's already out there. This project was a way for me to learn more about machine learning models in the back-end
