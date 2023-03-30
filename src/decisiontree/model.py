import pandas as pd
from datetime import datetime
import json
import numpy as np


class Tree:
    def __init__(self, 
                 X_train: pd.DataFrame = None, 
                 y_train: pd.Series = None, 
                 maxDepth: int = 35, 
                 minLabels: int = 10, 
                 maxImpurity: float = 0.1, 
                 ensembled = False,
                 read = False) -> None:
        
        #data
        if not read:
            self.X_train = X_train
            self.y_train = y_train
            self.data = pd.concat((X_train, y_train), axis=1)
            # hyperparameters
            self.maxDepth = maxDepth
            self.minLabels = minLabels
            self.maxImpurity = maxImpurity

            self.depth = 0
            self.ensembled = ensembled

            if not ensembled:
                print("Attempting to train tree with hyperparameters:\n",
                    f"\tmaxDepth={self.maxDepth}\n",
                    f"\tminLabels={self.minLabels}\n",
                    f"\tmaxImpurity={self.maxImpurity}\n")

            self.root = self._train(self.data)

            if not ensembled: 
                if self.depth < self.maxDepth:
                    print("\nNOTE: The training terminated 'prematurely' (i.e., before reaching set max depth) due to one of the other stopping criterion \n",
                        "being met. Probably due to there not being any splits that meet the max acceptable impurity while still satisfying the min label members.")
        

    def _train(self, data, depth = 0):
        self.depth = depth if depth > self.depth else self.depth + 0
        if not self.ensembled:
            print(f"\rTraining \t{ (self.depth/self.maxDepth) * 100:.1f}%",
                f" |{ (int(50 * ((self.depth/self.maxDepth)))) * '='}{ (50 - int(50 * (self.depth/self.maxDepth))) * '.' }|",
                f"\tDepth { self.depth }/{ self.maxDepth }", end="")

        dataImpurity = gini(getClassDistribution(data.iloc[:, -1]))
        split = self._findSplit(data)

        # Stopping Criterion:
        # Max depth has ben reached
        # Split impurity is below max
        # Data cannot be split anymore and still respect max impurity and min label members
        if depth == self.maxDepth or dataImpurity <= self.maxImpurity or split[list(split.keys())[0]]["feature"] == None:
            return Node(None, None, None, data.iloc[:, -1].mode()[0])
        

        leftData, rightData = self._splitData(split, data)

        node = Node(feature=split[list(split.keys())[0]]["feature"], value=split[list(split.keys())[0]]["value"], optimalSplitDirection=list(split.keys())[0])

        node.left = self._train(leftData, depth + 1)
        node.right = self._train(rightData, depth + 1)

        return node


    def _findSplit(self, data) -> dict:
        features = data.columns[:-1]


        giniSmallest = {
                "left": {"gini": 1, "feature": None, "value": None, "labels": 0},
                "right": {"gini": 1, "feature": None, "value": None, "labels": 0}
        }

        for feature in features:
            
            uniqueFeatureValues = data[feature].unique()
            uniqueFeatureValues.sort()

            for i in range(1, len(uniqueFeatureValues) - 1):

                 # values of labels of the data points with feature values to the left of the i-th unique feature value
                labelsLeft = data[data[feature] <= uniqueFeatureValues[i]].iloc[:, -1]

                # values of labels of the data points with feature values to the right of the i-th unique feature value
                labelsRight = data[data[feature] >= uniqueFeatureValues[i]].iloc[:, -1]


                # gini values of the i-th unique feature value sliced labels sets 
                currentGiniLeft = gini(getClassDistribution(labelsLeft))
                currentGiniRight = gini(getClassDistribution(labelsRight))


                if currentGiniLeft < giniSmallest["left"]["gini"]:
                    if labelsLeft.count() >= self.minLabels: # labelsLeft.count() >= giniSmallest["left"]["labels"]: find way to implement
                        giniSmallest["left"]["gini"] = currentGiniLeft
                        giniSmallest["left"]["feature"] = feature
                        giniSmallest["left"]["value"] = uniqueFeatureValues[i]
                        giniSmallest["left"]["labels"] = labelsLeft.count()


                if currentGiniRight < giniSmallest["right"]["gini"]:
                    if labelsRight.count() >= self.minLabels: # and labelsRight.count() >= giniSmallest["right"]["labels"]: find way to implement
                        giniSmallest["right"]["gini"] = currentGiniRight
                        giniSmallest["right"]["feature"] = feature
                        giniSmallest["right"]["value"] = uniqueFeatureValues[i]
                        giniSmallest["right"]["labels"] = labelsRight.count()

   
        if giniSmallest["left"]["gini"] < giniSmallest["right"]["gini"]:
            return {"left": giniSmallest["left"]}
        else:
            return {"right": giniSmallest["right"]}



    def predict(self, X_test: pd.DataFrame):
        # TODO
        # - Refactor so it can also use  single unpacked itertuple datapoint so forest doesnt need its own copy of _getPred lol
        y_pred = pd.Series(dtype=float)

        for idx, *dataPoint in X_test.itertuples():
            prediction = self._getPred(dataPoint, self.root)
            y_pred.at[idx] = prediction
        
        return y_pred
    

    def _getPred(self, dataPoint, node):
        
        if node.left == None and node.right == None:
            return node.value
        
        direction = node.optimalSplitDirection

        if direction == "left":
            if dataPoint[node.feature] <= node.value:
                pred = self._getPred(dataPoint, node.left)
            else:
                pred =self._getPred(dataPoint, node.right)
        else:
            if dataPoint[node.feature] >= node.value:
                pred = self._getPred(dataPoint, node.left)
            else:
                pred = self._getPred(dataPoint, node.right)

        return pred
        

    def _splitData(self, split, data):
        if list(split.keys())[0] == "left":
            leftData = data[data[split["left"]["feature"]] <= split["left"]["value"]]
            rightData = data[data[split["left"]["feature"]] > split["left"]["value"]]
        else:
            leftData = data[data[split["right"]["feature"]] >= split["right"]["value"]] # this way the complement is always on the right
            rightData = data[data[split["right"]["feature"]] < split["right"]["value"]]

        return leftData, rightData
    

    def printTree(self, node, indent = '') -> None:
        """ Prints tree, depth is determined by 4 '-' characters. Left node is on top and is the one which the parent condition leads too
        
        ## Arguments
        node: node to traverse and print tree from
        indent: *by default ''* indent of root node
        """
        if node.left == None and node.right == None:
            print(f"{indent}predict {node.value}")
            return
        
        print(f"{indent}{node.feature} {'<=' if node.optimalSplitDirection == 'left' else '>='}{node.value}")
        self.printTree(node.left, indent + '----')
        self.printTree(node.right, indent + '----')

    def _toDict(self, node):

        if node.left == None and node.right == None:
            return {"value": node.value}

        d = {"feature": node.feature, "value": node.value, "optimalSplitDirection": node.optimalSplitDirection}

        d["left"] = self._toDict(node.left)
        d["right"] = self._toDict(node.right)

        return d
    

    def toJSON(self, fileName = datetime.today().strftime('%Y%m%d_%H%M%S') + ".json"): 
        d = self._toDict(self.root)
        with open(fileName, "w") as outfile:
            json.dump(d, outfile, cls=NpEncoder)


    @staticmethod
    def fromDict(nodeDict):
        if "left" not in nodeDict.keys() and "right" not in nodeDict.keys():
            return Node(None, None, None, nodeDict["value"])
        
        node = Node(feature=nodeDict["feature"], value=nodeDict["value"], optimalSplitDirection=nodeDict["optimalSplitDirection"])

        node.left = Tree.fromDict(nodeDict["left"])
        node.right = Tree.fromDict(nodeDict["right"])

        return node
    

    @staticmethod
    def fromJSON(fileName):
        with open(fileName, "r") as infile:
            tree_dict = json.load(infile) 

        tree = Tree(read=True)
        tree.root = Tree.fromDict(tree_dict)
        return tree
        

class Node:
    def __init__(self, left = None, right = None, feature = None, value = None, optimalSplitDirection = None) -> None:
        self.left = left
        self.right = right
        self.feature = feature
        self.value = value
        self.optimalSplitDirection = optimalSplitDirection


# https://stackoverflow.com/a/57915246
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def gini(classDistribution: dict[str, float]) -> float:
    """ Returns the Gini impurity of set

    Arguments:
    classDistribution: dictionary that associates class name and it's rational distribution within the set
    """
    sumSquared: float = 0

    for _class in classDistribution.keys():
        sumSquared += (classDistribution[_class] ** 2)

    return 1 - sumSquared


def getClassDistribution(labels: pd.Series) -> dict[str, float]:
    """ ## Returns
    Dictionary with the unique labels as keys and their rational distribution as the value
    
    ## Arguments
    labels: pandas Series, label column from data
    """
    classDistribution: dict[str, float] = {}
    uniqueClassCount: pd.Series = labels.value_counts()

    for i in range(len(uniqueClassCount)):
        classDistribution.update({
            str(uniqueClassCount.index[i]): uniqueClassCount[uniqueClassCount.index[i]] / len(labels)
        })
    
    return classDistribution
