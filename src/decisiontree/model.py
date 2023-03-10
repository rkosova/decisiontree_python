import pandas as pd

class Tree:
    def __init__(self, data: pd.DataFrame, maxDepth: int = 35) -> None:
        self.data = data
        self.classDistribution = getClassDistribution(self.data)

        # maxImpurity: float = 0 add to function that determines whether a node is a leaf or not 


    def findSplit(self, minLabels: float = 0):
        features = self.data.columns[:-1]

        giniSmallest = {
                "left": {"gini": 1, "feature": None, "value": None},
                "right": {"gini": 1, "feature": None, "value": None}
        }

        for feature in features:
            
            uniqueFeatureValues = self.data[feature].unique()
            uniqueFeatureValues.sort()

            for i in range(1, len(uniqueFeatureValues) - 1):

                 # values of labels of the data points with feature values to the left of the i-th unique feature value
                labelsLeft = self.data[self.data[feature] <= uniqueFeatureValues[i]].iloc[:, -1]

                # values of labels of the data points with feature values to the right of the i-th unique feature value
                labelsRight = self.data[self.data[feature] >= uniqueFeatureValues[i]].iloc[:, -1]


                if feature == 6 and i == 2:
                    pass 

                # gini values of the i-th unique feature value sliced labels sets 
                currentGiniLeft = gini(getClassDistribution(labelsLeft))
                currentGiniRight = gini(getClassDistribution(labelsRight))


                if currentGiniLeft < giniSmallest["left"]["gini"]:
                    if labelsLeft.count() >= minLabels:
                        giniSmallest["left"]["gini"] = currentGiniLeft
                        giniSmallest["left"]["feature"] = feature
                        giniSmallest["left"]["value"] = uniqueFeatureValues[i]


                if currentGiniRight < giniSmallest["right"]["gini"]:
                    if labelsRight.count() >= minLabels:
                        giniSmallest["right"]["gini"] = currentGiniRight
                        giniSmallest["right"]["feature"] = feature
                        giniSmallest["right"]["value"] = uniqueFeatureValues[i]



        # print()
        return giniSmallest
                
class Node:
    def __init__(self) -> None:
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


