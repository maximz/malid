# %%

# %% [markdown]
# # Likelihood ratio test when removing one demographic covariate at a time from the metamodel with sequence features and demographic features

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy.stats.distributions import chi2

# %%
from malid.trained_model_wrappers import BlendingMetamodel
from malid import io
from malid.datamodels import GeneLocus, TargetObsColumnEnum
import crosseval

# %%

# %%
clf = BlendingMetamodel.from_disk(
    fold_id=-1,
    # Load elastic net model because this had the highest cross validation test set performance in our evaluations for this metamodel flavor
    metamodel_name="elasticnet_cv",
    base_model_train_fold_name="train_smaller",
    metamodel_fold_label_train="validation",
    gene_locus=GeneLocus.BCR | GeneLocus.TCR,
    target_obs_column=TargetObsColumnEnum.disease_all_demographics_present,
    metamodel_flavor="with_demographics_columns",
)
clf

# %%
type(clf)

# %%
clf._inner

# %%

# %%

# %%
# Load data and make features
adata_bcr = io.load_fold_embeddings(
    fold_id=-1,
    fold_label="validation",
    gene_locus=GeneLocus.BCR,
    target_obs_column=TargetObsColumnEnum.disease_all_demographics_present,
)
adata_tcr = io.load_fold_embeddings(
    fold_id=-1,
    fold_label="validation",
    gene_locus=GeneLocus.TCR,
    target_obs_column=TargetObsColumnEnum.disease_all_demographics_present,
)
features = clf.featurize({GeneLocus.BCR: adata_bcr, GeneLocus.TCR: adata_tcr})

# %%
features.X

# %%
X = features.X.copy()
y = features.y.copy()

# %%
X

# %%
y

# %%
groups = features.metadata["participant_label"].values.copy()
groups


# %%

# %%

# %%
def clone_and_refit_with_same_lambda(clf, X, y, groups, feature_to_drop=None):
    """Clone the clf glmnet classifier and refit it having dropped feature_to_drop. Keep the regularization parameter lambda identical."""
    if feature_to_drop is not None:
        X = X.drop(feature_to_drop, axis="columns")

    # Clone
    clf_new = sklearn.base.clone(clf)

    # Set the lambda
    desired_lambda = clf.steps[-1][1]._lambda_for_prediction_
    clf_new.steps[-1][1]._inner.lambda_path = np.array([desired_lambda])
    clf_new.steps[-1][1].internal_cv = None
    clf_new.steps[-1][1]._inner.n_splits = 0
    clf_new.steps[-1][1]._inner.n_lambda = 1

    # Fit
    # Use crosseval to properly pass groups to the final step of the pipeline
    clf_new, _ = crosseval.train_classifier(
        clf=clf_new, X_train=X, y_train=y, train_groups=groups
    )

    assert len(clf_new.steps[-1][1].lambda_path_) == 1
    assert np.allclose(desired_lambda, clf_new.steps[-1][1].lambda_path_)

    return clf_new, desired_lambda, X


# %%
def log_likelihood(model, X, y, chosen_lambda):
    """Calculate the log-likelihood of a fitted multinomial model."""
    return -1 * sklearn.metrics.log_loss(
        y_true=y,
        y_pred=model.predict_proba(X, lamb=chosen_lambda),
        labels=model.classes_,
        normalize=False,
        sample_weight=None,
    )


# %%

# %%
# sanity check: should be identical outputs if refit with same data
clf_tmp, chosen_lambda, _ = clone_and_refit_with_same_lambda(
    clf._inner, X, y, groups, feature_to_drop=None
)
assert np.allclose(
    clf_tmp.predict_proba(X, lamb=chosen_lambda), clf.predict_proba(X), atol=1e-3
)

# %%

# %%
# Remove interaction terms between sequence features and demographic features.
X_trim = X[X.columns[~X.columns.str.startswith("interaction")]].copy()
X_trim.columns

# %%

# %%
X_trim.shape

# %%
X_trim

# %%
pd.Series(y).value_counts()

# %%

# %%

# %%
# Train original full model, without interaction terms
clf_orig, _ = crosseval.train_classifier(
    clf=sklearn.base.clone(clf._inner), X_train=X_trim, y_train=y, train_groups=groups
)
clf_orig

# %%

# %%

# %%

# %%
# Drop age, sex, and ancestry (which is a set of dummy variables), refitting the model each time and comparing log likelihoods
results = []
for to_drop in [
    ["demographics:age"],
    ["demographics:sex_F"],
    list(
        X_trim.columns[
            X_trim.columns.str.startswith("demographics:ethnicity_condensed")
        ]
    ),
]:
    reduced_model, chosen_lambda, X_reduced = clone_and_refit_with_same_lambda(
        clf_orig, X_trim, y, groups, feature_to_drop=to_drop
    )
    for f in to_drop:
        assert f not in X_reduced.columns
    ll_full = log_likelihood(clf_orig, X_trim, y, chosen_lambda)
    ll_reduced = log_likelihood(reduced_model, X_reduced, y, chosen_lambda)

    # Perform the likelihood ratio test
    likelihood_ratio = 2 * (ll_full - ll_reduced)
    # the full model has n_removed_features more Degrees of Freedom than the reduced model
    dof = len(to_drop)
    p_value = chi2.sf(
        likelihood_ratio,
        df=dof,
    )
    results.append(
        dict(
            p_value=p_value,
            likelihood_ratio=likelihood_ratio,
            to_drop=to_drop,
            ll_full=ll_full,
            ll_reduced=ll_reduced,
            dof=dof,
        )
    )


pd.DataFrame(results)

# %%

# %%

# %%
# Try again, but now keep the interaction terms. Drop each demographic feature along with all the associated interaction terms.
results_with_interaction_terms = []
for to_drop in [
    # Include all interaction terms
    list(X.columns[X.columns.str.contains("demographics:age", regex=False)]),
    list(X.columns[X.columns.str.contains("demographics:sex_F", regex=False)]),
    list(
        X.columns[
            X.columns.str.contains("demographics:ethnicity_condensed", regex=False)
        ]
    ),
]:
    reduced_model, chosen_lambda, X_reduced = clone_and_refit_with_same_lambda(
        clf._inner, X, y, groups, feature_to_drop=to_drop
    )
    for f in to_drop:
        assert f not in X_reduced.columns
    ll_full = log_likelihood(clf, X, y, chosen_lambda)
    ll_reduced = log_likelihood(reduced_model, X_reduced, y, chosen_lambda)

    # Perform the likelihood ratio test
    likelihood_ratio = 2 * (ll_full - ll_reduced)
    # the full model has n_removed_features more Degrees of Freedom than the reduced model
    dof = len(to_drop)
    p_value = chi2.sf(
        likelihood_ratio,
        df=dof,
    )
    results_with_interaction_terms.append(
        dict(
            p_value=p_value,
            likelihood_ratio=likelihood_ratio,
            to_drop=to_drop,
            ll_full=ll_full,
            ll_reduced=ll_reduced,
            dof=dof,
        )
    )


pd.DataFrame(results_with_interaction_terms)

# %%
