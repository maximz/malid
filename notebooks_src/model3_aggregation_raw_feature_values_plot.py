# %%

# %%
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns

import genetools
import shap
import joblib

from malid.trained_model_wrappers import VJGeneSpecificSequenceModelRollupClassifier
from malid import io, config
from malid.datamodels import GeneLocus, TargetObsColumnEnum

# %%

# %% [markdown]
# SHAP analysis highlighted IGHV3-7, IgG for T1D. Let's look at the raw feature values in one fold.

# %%

# %%
fold_id = 1
fold_label = "test"

# %%
adata_bcr = io.load_fold_embeddings(
    fold_id=fold_id,
    fold_label=fold_label,
    gene_locus=GeneLocus.BCR,
    target_obs_column=TargetObsColumnEnum.disease,
)

# %%

# %%
clf_bcr = VJGeneSpecificSequenceModelRollupClassifier(
    # First, all the usual parameters, like fold ID, sequencing locus, and classification target:
    fold_id=fold_id,
    gene_locus=GeneLocus.BCR,
    target_obs_column=TargetObsColumnEnum.disease,
    #
    # Model 3 includes a seqeunce stage and an aggregation stage.
    # The aggregation stage is trained on top of the sequence stage, so to speak.
    # First, provide the sequence stage model name:
    base_sequence_model_name=config.metamodel_base_model_names.base_sequence_model_name[
        GeneLocus.BCR
    ],
    # The sequence stage was trained on train_smaller1:
    base_model_train_fold_label="train_smaller1",
    #
    # Next, provide the aggregation stage model name here:
    rollup_model_name=config.metamodel_base_model_names.aggregation_sequence_model_name[
        GeneLocus.BCR
    ],
    # The aggregation stage was trained on train_smaller2:
    fold_label_train="train_smaller2",
)
clf_bcr

# %%
clf_bcr._inner

# %%

# %%
featurized_data_bcr = clf_bcr.featurize(adata_bcr)

# %%
featurized_data_bcr.X

# %%
# we will plot the feature values that go into the model after final scaling
transformed_X_bcr = clf_bcr._inner[:-1].transform(featurized_data_bcr.X)
transformed_X_bcr

# %%
fig, ax = plt.subplots(figsize=(6, 4))
feat = "T1D_IGHV3-7_IGHG"
friendly_name = feat.replace("T1D_IGHV3-7_IGHG", "IGHV3-7, IgG")
plot_df = pd.concat(
    [
        transformed_X_bcr[feat],
        (featurized_data_bcr.y == "T1D")
        .map({True: "T1D", False: "Not T1D"})
        .rename("Disease"),
    ],
    axis=1,
)
order = ["Not T1D", "T1D"]
ax = sns.boxplot(
    data=plot_df,
    x=feat,
    y="Disease",
    ax=ax,
    order=order,
    # Disable outlier markers:
    fliersize=0,
    palette=sns.color_palette("Paired")[:2],
    zorder=1,
)
for patch in ax.patches:
    # Set boxplot alpha transparency: https://github.com/mwaskom/seaborn/issues/979#issuecomment-1144615001
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.3))

np.random.seed(123)  # seed for stripplot jitter
sns.stripplot(
    data=plot_df,
    x=feat,
    y="Disease",
    order=order,
    hue="Disease",
    legend=None,
    linewidth=1,
    edgecolor="gray",
    palette=sns.color_palette("Paired")[:2],
    ax=ax,
    zorder=20,
    size=4,
    jitter=0.1,
    alpha=0.7,
)
plt.axvline(
    x=plot_df[plot_df["Disease"] == "Not T1D"][feat].describe().loc["75%"],
    linestyle="dashed",
    zorder=10,
    linewidth=1,
    # color=sns.color_palette("Paired")[0],
    color="k",
)
plt.xlabel(f"{friendly_name} feature value")
ax.set_yticklabels(
    genetools.plots.add_sample_size_to_labels(
        labels=ax.get_yticklabels(),
        data=plot_df,
        hue_key="Disease",
    )
)
plt.ylabel("Disease", rotation=0)
sns.despine(ax=ax)
genetools.plots.savefig(
    fig,
    clf_bcr.output_base_dir
    / f"{clf_bcr.fold_label_train}.feature_values.{clf_bcr.rollup_model_name}.fold_{clf_bcr.fold_id}.{fold_label}.png",
    dpi=300,
)

# %%
