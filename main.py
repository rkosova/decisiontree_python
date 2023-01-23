import pandas as pd
import numpy as np

 
def gini(classDistribution: dict[str, float]):
    """ Returns the Gini impurity of set

    Arguments:
    classDistribution: dictionary that associates class name and it's rational distribution within the set
    """

    sum_squared: float = 0

    for _class in classDistribution.keys():
        sum_squared += (classDistribution[_class] ** 2)

    return 1 - sum_squared



