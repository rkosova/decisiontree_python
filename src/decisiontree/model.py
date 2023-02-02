import pandas as pd
import numpy as np

class Tree:
    # figure way out to store layers
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.classDistribution = getClassDistribution(self.data)
    
    
    def train(self, maxDepth: int = None, maxImpurity: float = 0) -> dict:
        features = self.data.columns[:-1]
        for feature in features:
            giniLeftSmallest = 1
            giniRightSmallest = 1
            uniqueFeatureValues = self.data[feature].unique()
            uniqueFeatureValues.sort()

            for i in range(1, len(uniqueFeatureValues) - 1):

                # values of labels of the data points with feature values to the left of the i-th unique feature value
                labelsLeft = self.data[self.data[feature] < uniqueFeatureValues[i]].iloc[:, -1]

                # values of labels of the data points with feature values to the right of the i-th unique feature value
                labelsRight = self.data[self.data[feature] > uniqueFeatureValues[i]].iloc[:, -1]
                
                # gini values of the i-th unique feature value sliced labels sets 
                currentGiniLeft = gini(getClassDistribution(labelsLeft))
                currentGiniRight = gini(getClassDistribution(labelsRight))

                giniLeftSmallest = currentGiniLeft if currentGiniLeft < giniLeftSmallest else giniLeftSmallest
                giniRightSmallest = currentGiniRight if currentGiniRight < giniRightSmallest else giniRightSmallest

                if giniLeftSmallest <= maxImpurity or giniRightSmallest <= maxImpurity:
                    return (
                        {
                            "feature": feature,
                            "filter": lambda x: x < uniqueFeatureValues[i],
                            "value": uniqueFeatureValues[i]
                        }
                    ) if giniLeftSmallest < giniRightSmallest else (
                        {
                            "feature": feature,
                            "filter": lambda x: x > uniqueFeatureValues[i],
                            "value": uniqueFeatureValues[i]
                        }
                    )

class Node:
    def __init__(self, nodeDescriptor: dict) -> None:
        pass


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

