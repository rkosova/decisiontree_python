import pandas as pd
import numpy as np

class Tree:

    
    # figure way out to store layers
    def __init__(self, data: pd.DataFrame, maxDepth: int = 35) -> None:
        self.data = data
        self.classDistribution = getClassDistribution(self.data)

        self.firstLayer = [] 

        layers = 0

        while layers < maxDepth:
            # so it all stays within one loop maybe provide a function for traversing down a branch until 
            # the n-th node is reached and a function to append to the n-th node of a branch
            layers += 1
    
    def constructLayer(self, maxImpurity: float = 0):

        #todo: only internal nodes
        featuresAndImpurity = {}
        [featuresAndImpurity.update({key: None}) for key in self.data.columns[:-1]]

        leaves = [] 

        features = featuresAndImpurity.keys()
        for feature in features:
            giniLeftSmallest = Node(None, 1, None, None) 

            giniRightSmallest = Node(None, 1, None, None)


            uniqueFeatureValues = self.data[feature].unique()
            uniqueFeatureValues.sort()

            # print(feature, uniqueFeatureValues)

            for i in range(1, len(uniqueFeatureValues) - 1):

                # values of labels of the data points with feature values to the left of the i-th unique feature value
                labelsLeft = self.data[self.data[feature] <= uniqueFeatureValues[i]].iloc[:, -1]

                # values of labels of the data points with feature values to the right of the i-th unique feature value
                labelsRight = self.data[self.data[feature] >= uniqueFeatureValues[i]].iloc[:, -1]
                
                # gini values of the i-th unique feature value sliced labels sets 
                currentGiniLeft = gini(getClassDistribution(labelsLeft))
                currentGiniRight = gini(getClassDistribution(labelsRight))


                # used to find the smallest gini value of feature for internal (non-leaf node)
                if currentGiniLeft <= giniLeftSmallest.impurity:
                    giniLeftSmallest.impurity = currentGiniLeft
                    giniLeftSmallest.feature = feature
                    giniLeftSmallest.direction =  'left'
                    giniLeftSmallest.value = uniqueFeatureValues[i]

                
                if currentGiniRight <=  giniRightSmallest.impurity:
                    giniRightSmallest.impurity = currentGiniRight
                    giniRightSmallest.feature = feature
                    giniRightSmallest.direction = 'right'
                    giniRightSmallest.value = uniqueFeatureValues[i]

                # used to construct the leaves of the feature 
                if currentGiniLeft <= maxImpurity or currentGiniRight <= maxImpurity:
                    # print(uniqueFeatureValues[i])

                    leaf = (Node(feature, currentGiniLeft, uniqueFeatureValues[i], 'left') 
                    if (currentGiniLeft < currentGiniRight)
                    else Node(feature, currentGiniRight, uniqueFeatureValues[i], 'right'))
                    
                    leaves.append(leaf) 

            # todo: only internal nodes   
            featuresAndImpurity[feature] = (giniLeftSmallest if giniLeftSmallest.impurity < giniRightSmallest.impurity else giniRightSmallest)

        # _smallest = None
        # for i in featuresAndImpurity:
        #     if not _smallest:
        #         _smallest = featuresAndImpurity[i]
        #     else:
        #         _smallest = _smallest if _smallest['gini'] < featuresAndImpurity[i]['gini'] else featuresAndImpurity[i]

        # for i in featuresAndImpurity:
        #     print(featuresAndImpurity[i])
        #     print(f"For feature {featuresAndImpurity[i]['feature']}, the best available Gini is {featuresAndImpurity[i]['gini']} split at {featuresAndImpurity[i]['value']}: ")
        #     print("====" * 10)

        leaves_ = []

        # https://stackoverflow.com/questions/10305762/best-method-for-changing-a-list-while-iterating-over-it
        for i in range(len(leaves)):
            for j in range(len(leaves)):
                if leaves[i] != leaves[j] and leaves[i] and leaves[j] and leaves[i].feature == leaves[j].feature and leaves[i].direction == leaves[j].direction:
                    if leaves[i].direction == 'left':
                        if leaves[i].value > leaves[j].value:
                            leaves[j] = None
                        else:
                            leaves[i] = None
                    else:
                        if leaves[i].value > leaves[j].value:
                            leaves[i] = None
                        else:
                            leaves[j] = None
                        
            
        # if not self.firstLayer:
        return [leaf for leaf in leaves if leaf]
        # train could be called for each child node for every layer  

        

class Node:
    # implement Node as such that it has list of children nodes
    # think up how to traverse tree and add to each branch, i.e. parallel or one branch at a time, make pros and cons
    def __init__(self, feature: str | int, impurity: float, featureValue: int | float, direction: str) -> None:
        self.feature = feature
        self.impurity = impurity
        self.direction = direction
        self.value = featureValue


    def evaluate(self, data):
        if self.direction == 'right':
            return data >= self.value
        else:
            return data <= self.value


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


