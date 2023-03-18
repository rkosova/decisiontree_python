import pandas as pd
from decisiontree import Tree
from multiprocessing import Pool
import random
from collections import Counter

class Forest:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, n_trees: int, maxDepth: int = 10, minLabels: int = 10, maxImpurity: float = 0.1):
        self.n_trees = n_trees

        self.completed = 0

        print(f"Stand by as {n_trees} trees are being trained. Depending on the complexity of the data and depth of the tree, this may take a while.")
        
        with Pool(processes=n_trees) as pool:
            self.trees = pool.starmap(self._train_tree, [(X_train, y_train, maxDepth, minLabels, maxImpurity) for _ in range(n_trees)])


    def predict(self, X: pd.DataFrame):
        y_pred = pd.Series(dtype=float)

        for idx, *dataPoint in X.itertuples():
            preds = []
            for tree in self.trees:
                preds.append(self._getPred(dataPoint, tree.root))
            
            y_pred.at[idx] = Counter(preds).most_common(1)[0][0]
        
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


    def _train_tree(self, X_train: pd.DataFrame, y_train: pd.Series, maxDepth, minLabels, maxImpurity, ensembled = True):
        bootstrap = X_train.sample(frac=1, replace=True).sample(n=random.randint(1, len(X_train.columns)), axis=1)
        bootstrap_y_train = y_train.loc[bootstrap.index] 
        tree = Tree(bootstrap, bootstrap_y_train, maxDepth, minLabels, maxImpurity, ensembled)

        return tree
    
