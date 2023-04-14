import pandas as pd
from datetime import datetime
import json
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Tree:
    def __init__(self, 
                 X_train: pd.DataFrame = None, 
                 y_train: pd.Series = None, 
                 maxDepth: int = 35, 
                 minLabels: int = 10, 
                 maxImpurity: float = 0.1, 
                 ensembled = False,
                 read = False,
                 hyperTuned = False) -> None:
        
        #data
        if not read:
            self.X_train = X_train # remove header from here for _getPred, store it in some variable for printTree
            self.y_train = y_train
            self.data = pd.concat((X_train, y_train), axis=1)
            # hyperparameters
            self.maxDepth = maxDepth
            self.minLabels = minLabels
            self.maxImpurity = maxImpurity

            self.depth = 0
            self.ensembled = ensembled

            if not ensembled and not hyperTuned:
                print("\n\nAttempting to train tree with hyperparameters:\n",
                    f"\tmaxDepth={self.maxDepth}\n",
                    f"\tminLabels={self.minLabels}\n",
                    f"\tmaxImpurity={self.maxImpurity}\n")

            self.root = self._train(self.data)

            if hyperTuned:
                print() # to escape previous tree's progress bar

            if not ensembled and not hyperTuned: 
                if self.depth < self.maxDepth:
                    print("\nNOTE: The training terminated 'prematurely' (i.e., before reaching set max depth) due to one of the other stopping criterion \n",
                        "being met. Probably due to there not being any splits that meet the max acceptable impurity while still satisfying the min label members.")
        

    def _train(self, data, depth = 0):
        self.depth = depth if depth > self.depth else self.depth + 0
        if not self.ensembled and self.depth > 0:
            print(f"\rTraining \t{ (self.depth/self.maxDepth) * 100:.1f}%",
                f" |{ (int(50 * ((self.depth/self.maxDepth)))) * '='}{ (50 - int(50 * (self.depth/self.maxDepth))) * '.' }|",
                f"\tDepth { self.depth }/{ self.maxDepth }  minLabels: {self.minLabels}  maxImpurity: {self.maxImpurity}", end="")

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
    

    def toJSON(self, fileName = datetime.today().strftime('%Y%m%d%H%M%S') + ".json"): 
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


class HyperTuner:
    def __init__(self, X_train: pd.DataFrame = None,
                y_train: pd.Series = None, 
                maxDepth: int = 50, 
                minLabels: int = 10, 
                maxImpurity: int = 0.1, 
                nIndividuals: int = 10,
                targetFitness: float = 0.9,
                maxGenerations: int = 25,
                crossoverOperation: str = 'genomic',
                mutationChance: float = .15) -> None:
        # split X_train and y_train into X_train, y_train, X_validate, y_validate
        # validation data is used for genetic algorithm fitness
        self.maxDepth = maxDepth 
        self.minLabels = minLabels
        self.maxImpurity = maxImpurity
        self.nIndividuals = nIndividuals
        self.targetFitness = targetFitness
        self.maxGenerations = maxGenerations
        self.crossoverOperation = crossoverOperation
        self.mutationChance = mutationChance

        self.generations = 1
        self.population = []

        self._getBinStr = lambda x: format(x, 'b')

        self.maxDepthGenomeLength = len(self._getBinStr(self.maxDepth))
        self.minLabelsGenomeLength = len(self._getBinStr(self.minLabels))

        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(X_train, y_train, test_size=0.3)

        self._createFirstGeneration()
        self.fittest = self.population[0]

        for ind in self.population:
            if ind[1] > self.fittest[1]:
                self.fittest = ind

        while self.generations <= self.maxGenerations and self.fittest[1] < self.targetFitness:
            print(f"\n\nGeneration {self.generations} / {self.maxGenerations}\n\n")
            mates = self._tournamentSelection(self.population) 
            self.population = self._mate(mates)
            self._mutatePopulation(self.population)

            for ind in self.population:
                ind.append(self._fitness(ind[0]))

            for ind in self.population:
                if ind[1] > self.fittest[1]:
                    self.fittest = ind
            
            self.generations += 1

        fittestMaxDepth = int(self.fittest[0][:self.maxDepthGenomeLength], 2)
        fittestMinLabels = int(self.fittest[0][self.maxDepthGenomeLength:self.maxDepthGenomeLength + self.minLabelsGenomeLength], 2)
        fittestMaxImpurity = self._decodeTwelveBitImpurity(self.fittest[0][self.maxDepthGenomeLength + self.minLabelsGenomeLength:])

        s = f"Fittest individual from the {self.generations - 1} generations has an accuracy of {ind[1]}."

        print("*" * (len(s) + 2))
        print("*" + s + "*")
        print("*" * (len(s) + 2))

        tree = Tree(self.X_train, self.y_train, fittestMaxDepth, fittestMinLabels, fittestMaxImpurity)

        tree.toJSON()



    def _getTwelveBitImpurity(self, minImpurity: float):
        twelveBitImp = ''

        for i in str(minImpurity)[2:5]:
            twelveBitImp += self._getBinStr(int(i)).zfill(4)
        
        return twelveBitImp
    

    def _decodeTwelveBitImpurity(self, twelveBitImpurity: str):
        t = 0
        frac = 1
        for i in range(0, 12, 4):
            frac *= 10
            t += int(twelveBitImpurity[i:i + 4], 2)/frac

        return t
    

    def _createFirstGeneration(self):
        print(f"\n\nGeneration {self.generations} / {self.maxGenerations}\n\n")
        for i in range(self.nIndividuals):
            individual = []
            depthGenome = self._getBinStr(random.randint(1, self.maxDepth)).zfill(self.maxDepthGenomeLength)
            labelsGenome = self._getBinStr(random.randrange(0, self.minLabels)).zfill(self.minLabelsGenomeLength)
            impurityGenome = self._getTwelveBitImpurity(random.uniform(0, self.maxImpurity))
            
            individual.append((depthGenome + labelsGenome + impurityGenome))
            individual.append(self._fitness(individual[0]))
            self.population.append(individual)
        
        self.generations += 1


    def _fitness(self, individual):
        maxDepth = int(individual[:self.maxDepthGenomeLength], 2)
        minLabels = int(individual[self.maxDepthGenomeLength:self.maxDepthGenomeLength + self.minLabelsGenomeLength], 2)
        maxImpurity = self._decodeTwelveBitImpurity(individual[self.maxDepthGenomeLength + self.minLabelsGenomeLength:])

        tree = Tree(self.X_train, self.y_train, maxDepth, minLabels, maxImpurity, hyperTuned=True)
        y_pred = tree.predict(self.X_validate)

        accuracy = accuracy_score(self.y_validate, y_pred)

        return accuracy


    def _tournamentSelection(self, population):
        # SELECTION WITH REPLACEMENT. LEADS TO LESS DIVERSE SMALL POPULATIONS (smaller populations more easily converge to certain solutions)
        mates = []

        while len(mates) < len(population):
            tournament = random.sample(population, 2)
            tournament.sort(key=lambda x: x[1], reverse=True)
            mates.append(tournament[0])
        
        return mates
    

    def _crossover(self, individualA, individualB):
        childA = []
        childB = []
        if self.crossoverOperation == 'genomic':
            # maxDepth genome crossover
            maxDepthCPoint = random.randrange(0, self.maxDepthGenomeLength)

            individualAMaxDepthGenome = individualA[0][:self.maxDepthGenomeLength]
            individualBMaxDepthGenome = individualB[0][:self.maxDepthGenomeLength]

            childAMaxDepthGenome = individualAMaxDepthGenome[:maxDepthCPoint] + individualBMaxDepthGenome[maxDepthCPoint:]
            childBMaxDepthGenome = individualBMaxDepthGenome[:maxDepthCPoint] + individualAMaxDepthGenome[maxDepthCPoint:]

            # minLabels genome crossover
            minLabelsCPoint = random.randrange(0, self.minLabelsGenomeLength)

            individualAMinLabelsGenome = individualA[0][self.maxDepthGenomeLength:self.maxDepthGenomeLength + self.minLabelsGenomeLength]
            individualBMinLabelsGenome = individualB[0][self.maxDepthGenomeLength:self.maxDepthGenomeLength + self.minLabelsGenomeLength]

            childAMinLabelsGenome = individualAMinLabelsGenome[:minLabelsCPoint] + individualBMinLabelsGenome[minLabelsCPoint:]
            childBMinLabelsGenome = individualBMinLabelsGenome[:minLabelsCPoint] + individualAMinLabelsGenome[minLabelsCPoint:]

            # maxImpurity genome crossover
            maxImpurityCPoint = random.randrange(0, 12)

            individualAMaxImpurityGenome = individualA[0][self.maxDepthGenomeLength + self.minLabelsGenomeLength:]
            individualBMaxImpurityGenome = individualB[0][self.maxDepthGenomeLength + self.minLabelsGenomeLength:]

            childAMaxImpurityGenome = individualAMaxImpurityGenome[:maxImpurityCPoint] + individualBMaxImpurityGenome[maxImpurityCPoint:]
            childBMaxImpurityGenome = individualBMaxImpurityGenome[:maxImpurityCPoint] + individualAMaxImpurityGenome[maxImpurityCPoint:]


            childA.append(childAMaxDepthGenome + childAMinLabelsGenome + childAMaxImpurityGenome)
            childB.append(childBMaxDepthGenome + childBMinLabelsGenome + childBMaxImpurityGenome)
        elif self.crossoverOperation == 'holistic':
            individualAGeneticMaterial = individualA[0]
            individualBGeneticMaterial = individualB[0]

            crossoverPoint = random.randrange(len(individualAGeneticMaterial))

            _childA = individualAGeneticMaterial[:crossoverPoint] + individualBGeneticMaterial[crossoverPoint:]
            _childB = individualBGeneticMaterial[:crossoverPoint] + individualAGeneticMaterial[crossoverPoint:]

            childA.append(_childA)
            childB.append(_childB)

        return childA, childB



            

    def _mate(self, mates):
        children = []
        for i in range(0, len(mates), 2):
            children_ = list(self._crossover(mates[i], mates[i+1])) 
            [children.append(child) for child in children_]
            

        return children


    def _mutate(self, individual):
        ind_list = list(individual[0])
        index = random.randint(0, self.maxDepthGenomeLength + self.minLabelsGenomeLength + 12 - 1)
        ind_list[index] = str(1 - int(ind_list[index]))
        individual[0] = ''.join(ind_list)


    def _mutatePopulation(self, population):
        for ind in population:
            if random.random() <= self.mutationChance:
                self._mutate(ind)

    # genome crossover via k-point crossover. k-points may be positioned between genomes for genome-specific (genomic) crossover
    # or randomly for individual (holistic) crossover


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
