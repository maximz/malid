# %% [markdown]
# ## Emulate `analyze_metamodels.ipynb` to follow up on `in_house_peak_disease_leave_one_cohort_out`: exclude the samples where one replicate failed sequencing
#
# We found these samples in `paired_sample_batch_effects.ipynb` by applying our QC min-clone-count filters to the individual replicates.

# %%
import os

os.environ["MALID_CV_SPLIT"] = "in_house_peak_disease_leave_one_cohort_out"

# %%
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools
import genetools

# %matplotlib inline
import seaborn as sns
from IPython.display import display, Markdown

from malid import config, logger, helpers
from wrap_glmnet import GlmnetLogitNetWrapper
from malid.train import train_metamodel
import crosseval
from malid.datamodels import (
    DataSource,
    TargetObsColumnEnum,
    GeneLocus,
    healthy_label,
    map_cross_validation_split_strategy_to_default_target_obs_column,
)
from malid.trained_model_wrappers import BlendingMetamodel


# %%
base_model_train_fold_name = "train_smaller"
metamodel_fold_label_train = "validation"
gene_locus = config.gene_loci_used
target_obs_column = TargetObsColumnEnum.disease

# %%
flavors = train_metamodel.get_metamodel_flavors(
    gene_locus=gene_locus,
    target_obs_column=target_obs_column,
    fold_id=config.all_fold_ids[0],
    base_model_train_fold_name=base_model_train_fold_name,
    use_stubs_instead_of_submodels=True,
)

metamodel_flavor = "default"
metamodel_config = flavors["default"]

# %%
# should already exist:
metamodels_base_dir = BlendingMetamodel._get_metamodel_base_dir(
    gene_locus=gene_locus,
    target_obs_column=target_obs_column,
    metamodel_flavor=metamodel_flavor,
)

_output_suffix = Path(gene_locus.name) / target_obs_column.name / metamodel_flavor
# might not exist yet:
output_base_dir = (
    config.paths.second_stage_blending_metamodel_output_dir / _output_suffix
)
highres_output_base_dir = (
    config.paths.high_res_outputs_dir / "metamodel" / _output_suffix
)
output_base_dir.mkdir(parents=True, exist_ok=True)
highres_output_base_dir.mkdir(parents=True, exist_ok=True)

fname_prefix = (
    f"{base_model_train_fold_name}_applied_to_{metamodel_fold_label_train}_model"
)
model_prefix = metamodels_base_dir / fname_prefix
results_output_prefix = output_base_dir / fname_prefix
highres_results_output_prefix = highres_output_base_dir / fname_prefix

computed_abstentions = None

# Load and summarize
experiment_set = crosseval.ExperimentSet.load_from_disk(output_prefix=model_prefix)

# Note that default y_true from BlendingMetamodel._featurize() is target_obs_column.value.blended_evaluation_column_name
# Use DROP_INCOMPLETE_FOLDS setting because alternate classification targets might not be well-split in the small validation set of the cross-validation folds that were designed to stratify disease.
# In the cases of some classification targets, we might need to automatically drop folds that have only a single class in the metamodel training data (i.e. in the validation set).
experiment_set_global_performance = experiment_set.summarize(
    remove_incomplete_strategy=crosseval.RemoveIncompleteStrategy.DROP_INCOMPLETE_FOLDS
)

# %%

# %%
model_global_performance = experiment_set_global_performance.model_global_performances[
    "ridge_cv"
]

# %%
# review classification for each specimen
individual_classifications = model_global_performance.get_all_entries()

# %%
individual_classifications

# %%

# %%
# Load the list of rejected specimens (where at least one replicate failed QC)
bad_specimens = pd.read_csv(
    config.paths.base_output_dir_for_selected_cross_validation_strategy
    / "rejected_specimens_because_some_replicates_failed_qc.txt",
    header=None,
)[0].values
bad_specimens

# %%
# Filter out healthy samples that underwent replicate sequencing but failed
# See the other notebook mentioned above for details.
print(individual_classifications.shape)

individual_classifications_filtered = individual_classifications[
    (~individual_classifications["specimen_label"].isin(bad_specimens))
]
individual_classifications_filtered.shape

# %%
# Of the remainder: what was the accuracy?
for y_true, grp in individual_classifications_filtered.groupby("y_true"):
    print(f"For y_true={y_true}, predictions are:")
    print(grp["y_pred"].value_counts())
    print()

# %%
