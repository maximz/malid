def linear_model_supervised_embedding(clf, X):
    """A multi-class linear model like lasso has a coefficient matrix `coef_` of shape (n_classes, n_features).
    For input data X of shape (n_samples, n_features), the linear model can give us a "supervised embedding" of X as follows:
        `embedded = X * coef_.T`.

    This is multiplying the data (n_samples x n_features) by the coefficients (n_features x n_classes) of the linear model,
    producing a (n_samples x n_classes) matrix -- i.e. each data point is now a vector of n_classes dimensions.

    This is just a linear transformation that applies the coefficients as weights to the input features.
    The value at position [i, j] of this new matrix is the dot product of
    the i-th feature vector of the input data X and the coefficients vector of the "jth class vs all other classes" linear model.
    In other words, it is the sum of beta_i * x_i for all i from 1 to n_features.

    When you add the intercept beta_0 to that sum, it becomes the logit (returned by `decision_function()`), which is then converted to the class probability.

    You can then plot the pairwise scatterplots for all pairs of n_classes, and each axis will have a meaning (the log odds of each class vs all classes).

    Or you can do a UMAP or PCA to convert this n_classes-dimensional embedding into a 2d visualization.

    This is a replacement for training a neuralnet just to get the hidden state at a bottleneck layer.

    Returns:
        - a matrix of shape (n_samples, n_classes) where each row is the linear model's embedding of a single sample.
    """
    # get the logits
    # under the hood this is: (dot product of X and clf.coef_.T) + clf.intercept_
    return clf.decision_function(X)
