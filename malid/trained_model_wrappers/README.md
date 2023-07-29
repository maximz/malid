Exposes pre-trained repertoire and sequence classifiers for use on new data.

The convention is that these wrappers leave decision_function and predict_proba of the underlying classifiers untouched,
but provide helper functions that featurize an anndata into the right format for the underlying classifier and pass to those methods.
