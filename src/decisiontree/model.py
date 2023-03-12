import pandas as pd

class Tree:
    def __init__(self, data: pd.DataFrame, maxDepth: int = 35, minLabels: int = 10, maxImpurity: float = 0.1) -> None:

        #data
        self.data = data
        self.classDistribution = getClassDistribution(self.data)

        # hyperparameters
        self.maxDepth = maxDepth
        self.minLabels = minLabels
        self.maxImpurity = maxImpurity

        self.root = self.train(self.data)


    def train(self, data, depth = 0):
        dataImpurity = gini(getClassDistribution(data.iloc[:, -1]))
        # print(dataImpurity)
        if depth == self.maxDepth or dataImpurity <= self.maxImpurity:
            return Node(None, None, None, data.iloc[:, -1].mode()[0])
        
        split = self.findSplit(data)

        leftData, rightData = self.splitData(split, data)

        node = Node(feature=split[list(split.keys())[0]]["feature"], value=split[list(split.keys())[0]]["value"], optimalSplitDirection=list(split.keys())[0])

        node.left = self.train(leftData, depth + 1)
        node.right = self.train(rightData, depth + 1)

        return node


    def findSplit(self, data) -> dict:
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


    def splitData(self, split, data):
        if list(split.keys())[0] == "left":
            leftData = data[data[split["left"]["feature"]] <= split["left"]["value"]]
            rightData = data[data[split["left"]["feature"]] > split["left"]["value"]]
        else:
            leftData = data[data[split["right"]["feature"]] >= split["right"]["value"]] # this way the complement is always on the right
            rightData = data[data[split["right"]["feature"]] < split["right"]["value"]]

        return leftData, rightData
    

    def printTree(self, node, indent = ''):
        if node.left == None and node.right == None:
            print(f"{indent}predict {node.value}")
            return
        
        print(f"{indent}{node.feature} {'<=' if node.optimalSplitDirection == 'left' else '>='}{node.value}")
        self.printTree(node.left, indent + '----')
        self.printTree(node.right, indent + '----')
    
           

class Node:
    def __init__(self, left = None, right = None, feature = None, value = None, optimalSplitDirection = None) -> None:
        self.left = left
        self.right = right
        self.feature = feature
        self.value = value
        self.optimalSplitDirection = optimalSplitDirection
        

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


