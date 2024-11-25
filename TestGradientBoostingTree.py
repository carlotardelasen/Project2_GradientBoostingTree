import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from GradientBoostingTree import GradientBoostingTree, preprocess_data

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('insurance.csv',encoding='latin1')

    # Preprocess the data
    X, y = preprocess_data(df)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    gb_tree = GradientBoostingTree(M=200, max_depth=5, learning_rate=0.05)
    gb_tree.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = gb_tree.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    # Print the results
    print("Predictions:", predictions)
    print("R2 Score:", r2)
    print("Mean Absolute Error (MAE):", mae)


