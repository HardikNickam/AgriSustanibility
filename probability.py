import numpy as np

class RandomForestClassifier:
    def __init__(self, n_estimators=22):
        self.n_estimators = n_estimators
        self.trees = []
    
    def predict_proba(self, X):
        class_counts = np.zeros((len(X), 2))  
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier()
            probabilities = tree.predict_proba(X)
            class_counts += probabilities
            self.trees.append(tree)
        probabilities = class_counts / self.n_estimators
        return probabilities

class DecisionTreeClassifier:
    def predict_proba(self, X):
        probabilities = np.zeros((len(X), 2))  
        for i, x in enumerate(X):
            if x[0] > 0.5:
                probabilities[i][0] = 0.8
                probabilities[i][1] = 0.2
            else:
                probabilities[i][0] = 0.2
                probabilities[i][1] = 0.8
        return probabilities
    
    def predict_proba(self, X_test):
        class_counts = np.zeros((len(X_test), 2))  
        probabilities = np.zeros((len(X_test), 2))
        prob_class0 = 0.7
        prob_class1 = 1 - prob_class0
        probabilities[:, 0] = prob_class0
        probabilities[:, 1] = prob_class1
        return probabilities
