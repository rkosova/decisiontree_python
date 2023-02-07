import pandas as pd
import numpy as np

class Tree:

    
    # figure way out to store layers
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.classDistribution = getClassDistribution(self.data)
    
    
    def train(self, maxDepth: int = None, maxImpurity: float = 0) -> dict:
        featuresAndImpurity = {}
        [featuresAndImpurity.update({key: None}) for key in self.data.columns[:-1]]
        features = featuresAndImpurity.keys()
        for feature in features:
            giniLeftSmallest = (
                            {
                                "gini": 1,
                                "feature": None,
                                "filter": None,
                                "value": None,
                                "direction": "left"
                            }
            )

            giniRightSmallest = (
                            {
                                "gini": 1,
                                "feature": None,
                                "filter": None,
                                "value": None,
                                "direction": "right"
                            }
            )
            uniqueFeatureValues = self.data[feature].unique()
            uniqueFeatureValues.sort()

            for i in range(1, len(uniqueFeatureValues) - 1):

                # values of labels of the data points with feature values to the left of the i-th unique feature value
                labelsLeft = self.data[self.data[feature] <= uniqueFeatureValues[i]].iloc[:, -1]

                # values of labels of the data points with feature values to the right of the i-th unique feature value
                labelsRight = self.data[self.data[feature] >= uniqueFeatureValues[i]].iloc[:, -1]
                
                # gini values of the i-th unique feature value sliced labels sets 
                currentGiniLeft = gini(getClassDistribution(labelsLeft))
                currentGiniRight = gini(getClassDistribution(labelsRight))

                if currentGiniLeft < giniLeftSmallest['gini']:
                    giniLeftSmallest['gini'] = currentGiniLeft
                    giniLeftSmallest['feature'] = feature
                    giniLeftSmallest['filter'] =  makeClosure(uniqueFeatureValues[i], giniLeftSmallest['direction'])
                    giniLeftSmallest['value'] = uniqueFeatureValues[i]

                
                if currentGiniRight <  giniRightSmallest['gini']:
                    giniRightSmallest['gini'] = currentGiniRight
                    giniRightSmallest['feature'] = feature
                    giniRightSmallest['filter'] = makeClosure(uniqueFeatureValues[i], giniRightSmallest['direction']) 
                    giniRightSmallest['value'] = uniqueFeatureValues[i]


                if giniLeftSmallest['gini'] <= maxImpurity or giniRightSmallest['gini'] <= maxImpurity:
                    return giniLeftSmallest if (giniLeftSmallest['gini'] < giniRightSmallest['gini']) else giniRightSmallest
    
                #     return (
                #         {
                #             "feature": feature,
                #             "filter": lambda x: x < uniqueFeatureValues[i],
                #             "value": uniqueFeatureValues[i]
                #         }
                #     ) if giniLeftSmallest < giniRightSmallest else (
                #         {
                #             "feature": feature,
                #             "filter": lambda x: x > uniqueFeatureValues[i],
                #             "value": uniqueFeatureValues[i]
                #         }
                #     )
                
            featuresAndImpurity[feature] = (giniLeftSmallest if giniLeftSmallest['gini'] < giniRightSmallest['gini'] else giniRightSmallest)

        return featuresAndImpurity 

class Node:
    def __init__(self, nodeDescriptor: dict) -> None:
        self.feature = nodeDescriptor['feature']
        self.filter = nodeDescriptor['filter']
        self.value = nodeDescriptor['value']


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


def makeClosure(uniqueFeatureValue, direction):
    return (lambda x: x <= uniqueFeatureValue) if direction == "left" else (lambda x: x >= uniqueFeatureValue)
