# Project 2: Gradient-Boosting Trees Implementation

## **Authors**
- Clara Aparicio Mendez (A20599326)
- Juan Cantarero Angulo (A20598593)
- Raquel Gimenez Pascual (A20599725)
- Carlota Ruiz de Conejo de la Sen (A20600262)

This project implements the gradient-boosting tree algorithm, following the methodology described in Sections 10.9-10.10 of *Elements of Statistical Learning* (2nd Edition). The implementation provides a fit-predict interface for training and predicting with boosted trees, along with configurable parameters for optimization. The project is organized into two main files:

- **`GradientBoostingTree.py`**: Contains the implementation of the Gradient Boosting Tree model and its underlying Decision Tree class.
- **`TestGradientBoostingTree.py`**: Contains the test script for evaluating the model using the `insurance.csv` dataset. It also allows users to adjust hyperparameters to test the model's performance.

The dataset, `insurance.csv`, is also placed in the same directory as the Python files.

---

## **Overview**

### **What does the model do and when should it be used?**

The gradient-boosting tree model is a powerful ensemble learning method that builds a series of decision trees sequentially. Each subsequent tree corrects the errors of the previous ones by minimizing a loss function. The algorithm is widely used for regression and classification tasks due to its ability to model complex relationships and handle mixed data types effectively.

Gradient boosting is particularly useful in scenarios where:

- The dataset exhibits complex, non-linear relationships.
- High predictive accuracy is required, especially for regression or classification tasks.
- Interpretability is not the primary concern, and performance takes precedence.
- The data is structured or tabular, and the dataset size ranges from small to moderately large.

For this project, we implemented the model specifically for regression tasks, using a medical insurance dataset to predict individual charges based on personal and lifestyle factors such as age, sex, bmi, children, smoker, region and charges.

---

### **Features**
- Regression support.
- Customizable hyperparameters:
  - `learning_rate` to control the contribution of each tree.
  - `max_depth` to limit tree complexity.
  - `M` (number of trees) to balance bias-variance tradeoff.
- Handles both numerical and categorical data seamlessly with preprocessing.

---

## **Dataset Description**

The model was trained and evaluated using the `insurance.csv` dataset, which contains the following features:

1. **age:** Age of the individual.
2. **sex:** Gender (`male` or `female`).
3. **bmi:** Body Mass Index.
4. **children:** Number of children or dependents.
5. **smoker:** Whether the individual is a smoker (`yes` or `no`).
6. **region:** Geographical region (`northeast`, `southeast`, `southwest`, `northwest`).
7. **charges:** Medical insurance charges (target variable).

The dataset has 1,338 rows with no missing values, making it a clean and practical dataset for regression.

---

## **Training and Testing:**

How did you test your model to determine if it is working reasonably correctly?

### **Preprocessing**

The preprocessing steps include:

1. **Categorical Encoding:** 
   - Used one-hot encoding for `sex`, `smoker`, and `region`.
   - Dropped one category per feature to avoid multicollinearity.
2. **Feature Selection:**
   - Split `charges` as the target variable (`y`) and all other columns as features (`X`).

### **Model Training**

- The dataset was split into training (80%) and testing (20%) subsets using `train_test_split`.
- A Gradient Boosting Tree model was trained with the following parameters:
  - `M=200`: The number of boosting stages.
  - `max_depth=5`: Maximum depth of each decision tree.
  - `learning_rate=0.05`: Step size for each tree's contribution.
- Training involved sequentially fitting trees to minimize the residuals of the previous stage.

---

### **Testing and Validation**

The model was tested on unseen data (test subset) using the following metrics:

1. **R² Score:**
   - Measures the proportion of variance explained by the model.
   - Higher values indicate better performance.
2. **Mean Absolute Error (MAE):**
   - The average magnitude of errors in predictions, measured in the same units as the target variable.

---

## **Results**

On the insurance dataset:

- **R² Score:** Achieved a value of ~0.88, indicating a reasonable fit to the data.
- **Mean Absolute Error (MAE):** Approximately $2400, depending on the specific hyperparameters.

---

## **Model Parameters**

What parameters have you exposed to users of your implementation in order to tune performance?

The following parameters can be customized to tune the performance:

- **`M` (number of trees):**
  - Default: 100. Increasing this can reduce bias but may lead to overfitting.
- **`max_depth`:**
  - Default: 5. Controls the depth of each tree. Higher values allow for more complex splits.
- **`learning_rate`:**
  - Default: 0.1. Lower values improve accuracy but require more boosting stages.


---

## **Limitations**

Are there specific inputs that your implementation has trouble with?

Before arriving at the final version of our model, we experimented with a real estate dataset. This process highlighted some important limitations of our implementation:

- **Large datasets:**  
  The real estate dataset was considerably large and high-dimensional, leading to extended training times and increased memory usage. This made it challenging to efficiently fit our model within a reasonable timeframe.
  
- **Outliers:**  
  The dataset contained numerous outliers, especially in property prices, which heavily influenced the predictions. These outliers resulted in a lower \( R^2 \) score, as the model struggled to generalize well across the full range of values.

- **Missing values:**  
  Many features in the real estate dataset had missing values, requiring extensive preprocessing to handle imputations. Despite these efforts, the presence of missing data degraded overall model performance and increased complexity.

These challenges with the real estate dataset helped us identify areas for improvement in our implementation, such as handling outliers more robustly and optimizing performance for larger datasets. For the final implementation, we selected the insurance dataset, which is cleaner and smaller, allowing us to focus on model performance without being hindered by these issues.

---
## **Usage Instructions**

1. Ensure the files `GradientBoostingTree.py`, `TestGradientBoostingTree.py`, and `insurance.csv` are in the same directory.
2. Open `TestGradientBoostingTree.py` to modify hyperparameters as needed (`M`, `max_depth`, `learning_rate`).
3. Run the test script to train and evaluate the model:

```bash
python TestGradientBoostingTree.py
```

## **Usage Example**

```python

# Step 1: Load the dataset
df = pd.read_csv('insurance.csv')

# Step 2: Preprocess the data
X, y = preprocess_data(df)

# Step 3: Split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize the Gradient Boosting Tree model with the tuning parameters
gb_tree = GradientBoostingTree(M=200, max_depth=5, learning_rate=0.05)

# Step 5: Train the model
gb_tree.fit(X_train, y_train)

# Step 6: Make predictions on the test set
predictions = gb_tree.predict(X_test)

# Step 7: Evaluate the model
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# Step 8: Print results
print("Predictions:", predictions)
print("R2 Score:", r2)  # Closer to 1 is better
print("Mean Absolute Error (MAE):", mae)  # Lower values indicate better predictions
```
