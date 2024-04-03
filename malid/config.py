#!/usr/bin/env python

import logging
import os
from pathlib import Path
import pandas as pd
from typing import Dict, Callable, Optional, Type, Union

from matplotlib import rc

from malid.trained_model_wrappers.vj_gene_specific_sequence_classifier import (
    SequenceSubsetStrategy,
)


logger = logging.getLogger(__name__)

# Struct-like: namedtuple (https://stackoverflow.com/a/45426493/130164) or simplenamespace (https://dbader.org/blog/records-structs-and-data-transfer-objects-in-python)
from types import SimpleNamespace

from malid import embedders
from malid.embedders.base_embedder import BaseEmbedder, BaseFineTunedEmbedder
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
    CrossValidationSplitStrategy,
    map_cross_validation_split_strategy_to_default_target_obs_column,
)
import choosegpu
from environs import Env

env = Env()

## Feature flags for modeling

# Active dataset version
_default_dataset_version = "20231027"
dataset_version = os.getenv("MALID_DATASET_VERSION", _default_dataset_version)

# Active cross-validation split strategy
_default_cross_validation_split_strategy = (
    CrossValidationSplitStrategy.in_house_peak_disease_timepoints
)
# cross_validation_split_strategy = CrossValidationSplitStrategy[
#     os.getenv("MALID_CV_SPLIT", _default_cross_validation_split_strategy.name)
# ]
cross_validation_split_strategy: CrossValidationSplitStrategy = env.enum(
    "MALID_CV_SPLIT",
    type=CrossValidationSplitStrategy,
    ignore_case=True,
    # Pass .name as default here, because matching happens on string name:
    # The internal "if enum_value.name.lower() == value.lower()" will fail unless value is the .name. The enum object itself doesn't have a .lower()
    default=_default_cross_validation_split_strategy.name,
)

if not cross_validation_split_strategy.value.is_single_fold_only:
    # Default case
    n_folds = 3  # number of cross validation folds
    # list of all cross validation fold IDs
    cross_validation_fold_ids = list(range(n_folds))
    # also add the "global fold"
    use_global_fold = True
else:
    # Special case: single fold with a pre-determined held-out test set (see study_names_for_held_out_set docs).
    n_folds = 1
    cross_validation_fold_ids = [0]
    # no "global fold"
    use_global_fold = False

all_fold_ids = cross_validation_fold_ids.copy()
if use_global_fold:
    all_fold_ids += [-1]
# Record total number of folds, which may or may not include a global fold
n_folds_including_global_fold = len(all_fold_ids)

# Which sample weight strategy to use? Can select multiple, e.g. SampleWeightStrategy.ISOTYPE_USAGE | SampleWeightStrategy.CLONE_SIZE
sample_weight_strategy: SampleWeightStrategy = SampleWeightStrategy.ISOTYPE_USAGE

metamodel_base_model_names = SimpleNamespace()
# Base model names used in metamodels.
# This is set based on validation set performance of the base models.
metamodel_base_model_names.model_name_overall_repertoire_composition = {
    GeneLocus.BCR: "elasticnet_cv0.25",
    GeneLocus.TCR: "lasso_cv",
}
metamodel_base_model_names.model_name_convergent_clustering = {
    GeneLocus.BCR: "ridge_cv",
    GeneLocus.TCR: "lasso_cv",
}
metamodel_base_model_names.base_sequence_model_name = {
    GeneLocus.BCR: "rf_multiclass",
    GeneLocus.TCR: "ridge_cv_ovr",
}
metamodel_base_model_names.base_sequence_model_subset_strategy = (
    SequenceSubsetStrategy.split_Vgene_and_isotype
)
metamodel_base_model_names.aggregation_sequence_model_name = {
    GeneLocus.BCR: "rf_multiclass_mean_aggregated_as_binary_ovr_reweighed_by_subset_frequencies",
    GeneLocus.TCR: "rf_multiclass_entropy_twenty_percent_cutoff_aggregated_as_binary_ovr_reweighed_by_subset_frequencies",
}

# Default set of model names to train in many places, suitable for use cases that have ~hundreds of rows
model_names_to_train = [
    # Training is fast because the inputs are small,
    # so we can afford to train more complex models.
    "dummy_most_frequent",
    "dummy_stratified",
    #
    "lasso_cv",
    "elasticnet_cv0.75",
    "elasticnet_cv",
    "elasticnet_cv0.25",
    "ridge_cv",
    "logisticregression_unregularized",  # for comparison
    #
    "rf_multiclass",
    "linearsvm_ovr",
]
model_names_to_analyze_extra = [
    # Some analysis is time-consuming
    # This is a model name whitelist for analysis notebooks
    "ridge_cv",
]

sequence_identity_thresholds = SimpleNamespace()
# e.g. if 0.85: at least 85% sequence identity, means maximum normalized hamming distance of 15% allowed in a cluster
# we should be lenient for BCR due to somatic hypermutation, while stricter for TCR
sequence_identity_thresholds.call_nucleotide_clones_with_patient = {
    # These are the standards used in Boydlab pipeline run notes:
    GeneLocus.BCR: 0.9,
    GeneLocus.TCR: 0.95,
}
sequence_identity_thresholds.cluster_amino_acids_across_patients = {
    GeneLocus.BCR: 0.85,
    GeneLocus.TCR: 0.90,
}
sequence_identity_thresholds.assign_test_sequences_to_clusters = {
    GeneLocus.BCR: 0.85,
    GeneLocus.TCR: 0.90,
}

# Choose from _EMBEDDERS in embedders/__init__.py
# TODO: support configuring different embedders for each gene locus
_default_embedder = "esm2_cdr3"


def choose_embedder():
    return embedders.get_embedder_by_name(os.getenv("EMBEDDER_TYPE", _default_embedder))


embedder = choose_embedder()

choosegpu.preferred_gpu_ids = [3]

## Embedder-specific feature flags:
# Which data to include? e.g. GeneLocus.BCR | GeneLocus.TCR
# gene_loci_used will be iterable even if we only specify a single flag.
# To compute this, we will take the intersection of what is available in the dataset, and what is supported by the embedder.
_gene_loci_available_in_dataset = (
    cross_validation_split_strategy.value.gene_loci_supported
)
_gene_loci_supported_by_embedder = embedder.gene_loci_supported
# Because GeneLocus is a NotNullableFlag enum, we will get a ValueError if this AND operation results in null.
# That's what we want - we don't want gene_loci_used to be empty
gene_loci_used = _gene_loci_available_in_dataset & _gene_loci_supported_by_embedder


# Default classification targets: enable all
_default_classification_targets = set(TargetObsColumnEnum)
# Or, temporarily subset to certain classification targets only
if env.bool("MALID_DEFAULT_TARGET_ONLY", False):
    _default_classification_targets = {
        # Include only the default TargetObsColumn for the current active cross validation split strategy.
        # e.g. this might be TargetObsColumnEnum.disease
        map_cross_validation_split_strategy_to_default_target_obs_column[
            cross_validation_split_strategy
        ],
    }

# Subset the classification targets:
# TODO(Refactor): this should be a computed property so we can change the cross validation split strategy as we go. But config would need to be a class to have a property.
classification_targets = {
    target_obs_column
    for target_obs_column in _default_classification_targets
    # 1) Must be available for the active cross validation split strategy
    if cross_validation_split_strategy
    in target_obs_column.value.available_for_cross_validation_split_strategies
    # 2) Must be available for any of the active gene loci (but remember this TargetObsColumn filter is optional)
    and (
        target_obs_column.value.limited_to_gene_locus is None
        or
        # we can't do: len(gene_loci_used & target_obs_column.value.limited_to_gene_locus) > 0)
        # because, for example, len(GeneLocus.BCR & GeneLocus.TCR) throws ValueError: Null / zero-value is not allowed in NotNullableFlag GeneLocus enum.
        # instead we use something like this: any(g in GeneLocus.TCR for g in GeneLocus.BCR)
        any(g in gene_loci_used for g in target_obs_column.value.limited_to_gene_locus)
    )
}

# Special casing for Adaptive data:
# Change metamodel base model names based on validation set performance
if (
    cross_validation_split_strategy
    == CrossValidationSplitStrategy.adaptive_peak_disease_timepoints
):
    metamodel_base_model_names.model_name_overall_repertoire_composition = {
        GeneLocus.TCR: "lasso_cv",  # No change
    }
    metamodel_base_model_names.model_name_convergent_clustering = {
        GeneLocus.TCR: "elasticnet_cv0.25",
    }
    if embedder.name == "esm2_cdr3":
        metamodel_base_model_names.base_sequence_model_name = {
            GeneLocus.TCR: "ridge_cv_ovr",  # No change
        }
        metamodel_base_model_names.aggregation_sequence_model_name = {
            # TODO: Update:
            GeneLocus.TCR: "rf_multiclass_mean_aggregated_as_binary_ovr",
        }
    else:
        raise ValueError("Unsupported")
elif (
    cross_validation_split_strategy
    == CrossValidationSplitStrategy.adaptive_peak_disease_timepoints_leave_some_cohorts_out
):
    metamodel_base_model_names.model_name_overall_repertoire_composition = {
        GeneLocus.TCR: "ridge_cv",
    }
    metamodel_base_model_names.model_name_convergent_clustering = {
        GeneLocus.TCR: "ridge_cv",
    }
    if embedder.name == "esm2_cdr3":
        metamodel_base_model_names.base_sequence_model_name = {
            GeneLocus.TCR: "ridge_cv_ovr",  # No change
        }
        metamodel_base_model_names.aggregation_sequence_model_name = {
            GeneLocus.TCR: "ridge_cv_mean_aggregated_as_binary_ovr",
        }
    else:
        raise ValueError("Unsupported")


#######


# Configure paths here.
# Then create all necessary directories with: `python scripts/make_dirs.py`
def make_paths(
    embedder: Union[Type[BaseEmbedder], Type[BaseFineTunedEmbedder]],
    cross_validation_split_strategy: CrossValidationSplitStrategy,
    dataset_version: str,
    base_data_dir="data",
    base_output_dir="out",
    base_scratch_dir="/srv/scratch/$USER/",
    relative_to_path=None,
):
    """
    Configure file paths.
    Pass in current embedder type, dataset version, and cross-validation split strategy to generate configuration-specific paths.

    Hierarchy of paths:
    - `paths.base_data_dir`
        -> divide by data version: `paths.data_versioned_root`
            -> full dataset, before applying a particular cross validation split strategy
            -> divide by cross validation split strategy (e.g. peak disease timepoints only): `paths.data_versioned_for_selected_cross_validation_strategy`
                -> general non-embedder-specific models and metadata, e.g. models 1 and 2
                -> divide by embedder type: `paths.base_embedder_data_dir` and derived paths
                    -> embedder-specific models and metadata
    - `paths.base_output_dir` (not versioned to data; applies to latest data version at a given time; assumes that outputs are tracked in git along with dataset version config changes)
        -> general outputs describing the full dataset, before applying a particular cross validation split strategy
        -> divide by cross validation split strategy (e.g. peak disease timepoints only): `paths.base_output_dir_for_selected_cross_validation_strategy`
            -> general non-embedder-specific outputs, e.g. from models 1 and 2
            -> divide by embedder type: `paths.output_dir` and derived paths
                -> embedder-specific outputs
    - `paths.base_scratch_dir`: used for temporary files only
    """

    paths = SimpleNamespace()
    # Root directories
    paths.base_data_dir = Path(base_data_dir)
    paths.base_output_dir = Path(base_output_dir)
    paths.base_scratch_dir = Path(base_scratch_dir)

    # ********************************************************************************************************************

    ### General:
    paths.metadata_dir = "metadata"

    # Set logdir to something Github Actions CI can access,
    # and adjust ci.yaml accordingly to ensure this directory exists
    paths.log_dir = paths.base_data_dir / "logs"

    paths.external_raw_data = paths.base_data_dir / "external_cohorts" / "raw_data"

    paths.tests_snapshot_dir = "tests/snapshot"
    paths.transformers_cache = paths.base_data_dir / "transformers_cache/"

    # Local cache of network drive files
    paths.local_machine_cache_dir = paths.base_scratch_dir / "cache"

    # ********************************************************************************************************************

    ### Outputs:
    # Our pattern here is that outputs are not versioned to the data (i.e. data version name is not stored in the path),
    # because outputs are tracked in git along with dataset version config changes.

    ## Not associated with a particular embedder:
    paths.base_output_dir_for_selected_cross_validation_strategy = (
        paths.base_output_dir / cross_validation_split_strategy.name
    )
    # Model 1
    paths.repertoire_stats_classifier_output_dir = (
        paths.base_output_dir_for_selected_cross_validation_strategy
        / "repertoire_stats"
    )
    # Model 2
    paths.convergent_clusters_output_dir = (
        paths.base_output_dir_for_selected_cross_validation_strategy
        / "convergent_clusters"
    )
    paths.exact_matches_output_dir = (
        paths.base_output_dir_for_selected_cross_validation_strategy / "exact_matches"
    )
    # interpretations, not associated with a particular embedder
    paths.model_interpretations_for_selected_cross_validation_strategy_output_dir = (
        paths.base_output_dir_for_selected_cross_validation_strategy / "interpretations"
    )

    ## Associated with a particular embedder:
    paths.output_dir = (
        paths.base_output_dir_for_selected_cross_validation_strategy / embedder.name
    )
    # Model 3
    paths.sequence_models_output_dir = paths.output_dir / "sequence_models"
    # Metamodel
    paths.second_stage_blending_metamodel_output_dir = (
        paths.output_dir / "blending_metamodel"
    )
    # interpretations
    # TODO: Rename to be clear this is specific to an embedder
    paths.model_interpretations_output_dir = paths.output_dir / "interpretations"
    # external cohort evaluations
    paths.external_cohort_evaluation_output_dir = (
        paths.output_dir / "external_cohort_evaluation"
    )

    # ********************************************************************************************************************

    ### Raw data and non-embedder models:
    ## Versioned with the data, but not associated with a particular cross validation split strategy or embedder:
    paths.data_versioned_root = paths.base_data_dir / f"data_v_{dataset_version}"
    paths.scratch_versioned_root = paths.base_scratch_dir / f"data_v_{dataset_version}"

    # all sequences
    paths.sequences = paths.data_versioned_root / "sequences.parquet"
    # sampled subset: sample a sequence from each clone (essentially)
    paths.sequences_sampled = paths.data_versioned_root / "sequences.sampled.parquet"

    paths.dataset_specific_metadata = paths.data_versioned_root / "metadata"

    # Store oversized outputs here
    paths.high_res_outputs_dir_base = paths.data_versioned_root / "high_res_outputs"

    # ********************************************************************************************************************

    ## Specific to a cross validation split strategy, but not yet to an embedder:
    paths.data_versioned_for_selected_cross_validation_strategy = (
        paths.data_versioned_root / cross_validation_split_strategy.name
    )
    paths.scratch_versioned_for_selected_cross_validation_strategy = (
        paths.scratch_versioned_root / cross_validation_split_strategy.name
    )

    # Model 1
    paths.repertoire_stats_classifier_models_dir = (
        paths.data_versioned_for_selected_cross_validation_strategy / "repertoire_stats"
    )
    # Model 2
    paths.convergent_clusters_models_dir = (
        paths.data_versioned_for_selected_cross_validation_strategy
        / "convergent_clusters"
    )
    paths.exact_matches_models_dir = (
        paths.data_versioned_for_selected_cross_validation_strategy / "exact_matches"
    )

    # dataset EDA
    paths.dotplots_input = (
        paths.data_versioned_for_selected_cross_validation_strategy
        / "dotplots"
        / "input"
    )
    paths.dotplots_output = (
        paths.data_versioned_for_selected_cross_validation_strategy
        / "dotplots"
        / "output"
    )

    paths.dataset_specific_metadata_for_selected_cross_validation_strategy = (
        paths.data_versioned_for_selected_cross_validation_strategy / "metadata"
    )

    # Store oversized outputs here
    paths.high_res_outputs_dir_for_cross_validation_strategy = (
        paths.data_versioned_for_selected_cross_validation_strategy / "high_res_outputs"
    )

    # ********************************************************************************************************************

    ### Embedder-specific data and models:
    ## Versioned with the data, with a cross validation split strategy applied, and specific to an embeddder:
    paths.anndatas_scratch_dir = (
        paths.scratch_versioned_for_selected_cross_validation_strategy
        / f"{embedder.name}/anndatas"
    )
    paths.base_embedder_data_dir = (
        paths.data_versioned_for_selected_cross_validation_strategy
        / f"embedded/{embedder.name}"
    )

    paths.fine_tuned_embedding_dir = (
        paths.base_embedder_data_dir / "fine_tuned_embedding"
    )

    paths.anndatas_dir = paths.base_embedder_data_dir / "anndatas_temp"
    paths.scaled_anndatas_dir = paths.base_embedder_data_dir / "anndatas_scaled"

    paths.simulated_data_dir = paths.base_embedder_data_dir / "simulated_data"

    # Model 3
    paths.sequence_models_dir = paths.base_embedder_data_dir / "sequence_models"
    # Metamodel
    paths.second_stage_blending_metamodel_models_dir = (
        paths.base_embedder_data_dir / "blending_metamodel"
    )

    # Store oversized outputs here
    paths.high_res_outputs_dir = paths.base_embedder_data_dir / "high_res_outputs"

    # ********************************************************************************************************************

    # convert all to absolute paths - consider the above relative to where this script lives, not where it's called from
    for key, relative_path in paths.__dict__.items():
        # wrong way: this would be relative to where config.py is imported from!
        # full_path = os.path.abspath(relative_path)

        ## get absolute path by starting from where this config.py lives,
        # going up two levels to root project directory,
        # then appending relative path.
        if relative_to_path is None:
            # go up two levels to apply relative path
            relative_to_path = Path(__file__) / "../../"
        absolute_path = Path(relative_to_path) / relative_path

        ## convert to aboslute path:
        # don't use Pathlib resolve() because it follows symlinks (following symlinks can break summarynb),
        # # absolute_path = absolute_path.resolve()
        # Pathlib's undocumented absolute() method is also not the right choice, because it leaves "../" in path.
        # So use old-school os.path.abspath, then cast back to Path.
        # Also run os.path.expandvars to convert $USER to username or substitute other env vars.
        absolute_path = Path(os.path.expandvars(os.path.abspath(absolute_path)))

        ## store
        paths.__dict__[key] = absolute_path

    return paths


# Consider refactor: Make config a contextmanager, so we can do `with config.activate_embedder():` to temporarily change config.paths to a particular embedder.
paths = make_paths(
    embedder=embedder,
    cross_validation_split_strategy=cross_validation_split_strategy,
    dataset_version=dataset_version,
)


def make_dirs():
    """Create all necessary directories (except parquet directory), and necessary subdirectories in output folder, and all intermediate directories (like `mkdir -p`)"""
    dirs_to_make = [
        v
        for v in paths.__dict__.values()
        if v not in [paths.sequences, paths.sequences_sampled]
    ]
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Created directory: {d}")


# Dask config
dask_scheduler_port = 61094
# access dashbaord at f"http://127.0.0.1{dask_dashboard_address}"
dask_dashboard_address = ":61093"
# if Dask cluster already created, and want to connect from a new process, see https://stackoverflow.com/questions/60115736/dask-how-to-connect-to-running-cluster-scheduler-and-access-total-occupancy
dask_n_workers = 8

# Set joblib memory sharing to use /tmp instead of /dev/shm, which may be too small for model 3 training of large anndatas (see https://stackoverflow.com/a/43096735/130164)
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"


# Also configure huggingface/transformers cache
# https://huggingface.co/transformers/master/installation.html#caching-models
# For some reason, not all from_pretrained() calls pick this up, so we'll also pass in cache_dir kwarg.
os.environ["TRANSFORMERS_CACHE"] = str(
    paths.transformers_cache
)  # requires str, not Path

# Get API key for logging
sentry_api_key = os.getenv("SENTRY_API_KEY", default=None)


def get_fold_split_labels():
    # Default to: train on "train_smaller", evaluate on "validation" sets. (Don't make "test" set as easy to access.)
    return ["train_smaller", "validation"]


# Plotting preferences
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 12})
# ensure text is editable when we save figures in vector format
rc("pdf", fonttype=42)
rc("ps", fonttype=42)

if __name__ == "__main__":
    # print paths
    import pprint

    print(embedder.name)

    pprint.pprint(paths.__dict__)
