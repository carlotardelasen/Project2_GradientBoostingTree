import numpy as np
import pandas as pd

# Class for a single Decision Tree
class DecisionTree:
    def __init__(self, max_depth=5):
        """
        Initialize the Decision Tree.
        Parameters:
        - max_depth: Maximum depth of the tree to control overfitting.
        """
        self.max_depth = max_depth
        self.feature = None          # Feature to split on
        self.threshold = None        # Threshold value for splitting
        self.left_value = None       # Value for the left leaf node
        self.right_value = None      # Value for the right leaf node
        self.left_tree = None        # Left subtree
        self.right_tree = None       # Right subtree

    def fit(self, X, y):
        """
        Fit the Decision Tree to the data.
        Parameters:
        - X: Features (input data).
        - y: Target values (output data).
        """
        self._fit(X, y, depth=0)

    def _fit(self, X, y, depth):
        """
        Recursive function to build the tree.
        Parameters:
        - X: Features at the current node.
        - y: Target values at the current node.
        - depth: Current depth of the tree.
        """
        # Stop if maximum depth is reached or no further splitting is possible
        if depth >= self.max_depth or len(X) <= 1:
            self.left_value = self.right_value = np.mean(y)  # Assign mean value
            return

        best_feature, best_threshold, best_loss = None, None, float("inf")
        best_left_idx, best_right_idx = None, None

        # Iterate over all features and their unique values
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx

                # Skip thresholds that create empty splits
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                # Compute losses only if there are enough elements
                left_loss = np.var(y[left_idx]) * np.sum(left_idx) if np.sum(left_idx) > 1 else 0
                right_loss = np.var(y[right_idx]) * np.sum(right_idx) if np.sum(right_idx) > 1 else 0
                total_loss = left_loss + right_loss

                # Update the best split if this split improves the loss
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_feature = feature
                    best_threshold = threshold
                    best_left_idx = left_idx
                    best_right_idx = right_idx
                    
        # Store the best feature and threshold for this node (best split)
        self.feature = best_feature
        self.threshold = best_threshold

        # Recursively build left and right subtrees
        if best_left_idx is not None:
            self.left_tree = DecisionTree(self.max_depth - 1)
            self.left_tree._fit(X[best_left_idx], y[best_left_idx], depth + 1)
            self.right_tree = DecisionTree(self.max_depth - 1)
            self.right_tree._fit(X[best_right_idx], y[best_right_idx], depth + 1)

    def predict(self, X):
        """
        Predict the target values for the given input data.
        Parameters:
        - X: Features to predict.
        Returns:
        - Predictions for each row in X.
        """
        # If this is a leaf node, return the stored value
        if self.feature is None:
            return np.full(X.shape[0], self.left_value)
        
        # Otherwise, divide data based on the threshold and recurse
        left_idx = X[:, self.feature] <= self.threshold
        right_idx = ~left_idx
        predictions = np.zeros(X.shape[0])

        if self.left_tree:
            predictions[left_idx] = self.left_tree.predict(X[left_idx])
        if self.right_tree:
            predictions[right_idx] = self.right_tree.predict(X[right_idx])

        return predictions


# Class for Gradient Boosting using Decision Trees
class GradientBoostingTree:
    def __init__(self, M=100, max_depth=5, learning_rate=0.1):
        """
        Initialize the Gradient Boosting Tree model.
        Parameters:
        - M: Number of boosting iterations (trees).
        - max_depth: Maximum depth of each tree.
        - learning_rate: Step size for each tree's contribution.
        """
        self.M = M
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.models = []  
        self.f_0 = None  

    def fit(self, X, y):
        """
        Fit the Gradient Boosting model to the data.
        Parameters:
        - X: Features (input data).
        - y: Target values (output data).
        """
        N = len(y)
        # Loss function: Mean Squared Error
        self.f_0 = np.mean(y) 
        f = np.full(N, self.f_0) 

        for m in range(self.M):
            # Compute the gradient (residuals)
            residuals = y - f  
            tree = DecisionTree(max_depth=self.max_depth) 
            tree.fit(X, residuals)  
            self.models.append(tree) 

            predictions = tree.predict(X) 
            f += self.learning_rate * predictions  

    def predict(self, X):
        """
        Predict the target values for the given input data.
        Parameters:
        - X: Features to predict.
        Returns:
        - Predictions for each row in X.
        """
        # Start with the initial prediction
        f = np.full(X.shape[0], self.f_0)  
        for tree in self.models:
            # Add contributions from each tree
            f += self.learning_rate * tree.predict(X)  
        return f


# Preprocessing function for the dataset
def preprocess_data(df):
    """
    Preprocess the dataset.
    - Encodes categorical variables using one-hot encoding.
    - Splits the dataset into features (X) and target variable (y).
    Parameters:
    - df: DataFrame containing the dataset.
    Returns:
    - X: Processed feature matrix.
    - y: Target variable.
    """
    # 1. Encode categorical variables using one-hot encoding
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

    # 2. Split the dataset into features and target variable
    X = df.drop(columns=['charges'], errors='ignore') 
    y = df['charges'] if 'charges' in df.columns else None  

    return X.values, y.values

