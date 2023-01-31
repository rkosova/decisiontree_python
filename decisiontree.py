import pandas as pd
import numpy as np

class Tree:
    # figure way out to store layers
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.classDistribution = getClassDistribution(self.data)
    
    
    def train(maxDepth: int, maxImpurity: float = 0) -> None:
        pass


class Node:
    def __init__(self, nodeDescriptor: dict) -> None:
        # maybe a lambda function to store the node condition
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
    """ Returns a dictionary with the unique labels as keys and their rational distribution as the value
    
    Arguments:
    labels: pandas Series, label column from data
    """
    classDistribution: dict[str, float] = {}
    uniqueClassCount: pd.Series = labels.value_counts()

    for i in range(len(uniqueClassCount)):
        classDistribution.update({
            str(uniqueClassCount.index[i]): uniqueClassCount[uniqueClassCount.index[i]] / len(labels)
        })
    
    return classDistribution

