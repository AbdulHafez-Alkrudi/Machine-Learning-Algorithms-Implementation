import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Step 1: Generate a synthetic dataset with two features
X, y = make_classification(n_samples=10000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# Step 2: Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)


# Step 3: Create a function to plot the decision boundary
def plot_decision_boundary(X, y, model):
    # Define the grid to plot
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap=plt.cm.Paired)
    plt.title('Decision Boundary and Data Points')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# Step 4: Plot the decision boundary and points
plot_decision_boundary(X, y, model)
