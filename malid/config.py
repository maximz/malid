#!/usr/bin/env python

import logging
import os
from pathlib import Path
import pandas as pd
from typing import Dict, Callable, Optional

from matplotlib import rc

logger = logging.getLogger(__name__)

# Struct-like: namedtuple (https://stackoverflow.com/a/45426493/130164) or simplenamespace (https://dbader.org/blog/records-structs-and-data-transfer-objects-in-python)
from types import SimpleNamespace

from malid import embedders
from malid.datamodels import GeneLocus, TargetObsColumnEnum
import choosegpu

## Feature flags for modeling

n_folds = 3  # number of cross validation folds
# list of all cross validation fold IDs
cross_validation_fold_ids = list(range(n_folds))
# also add the "global fold"
all_fold_ids = cross_validation_fold_ids + [-1]

# Which data to include?
# This is iterable even if we only specify a single flag.
gene_loci_used: GeneLocus = GeneLocus.BCR | GeneLocus.TCR

include_v_gene_as_dummy_variable = True

metamodel_base_model_names = SimpleNamespace()
# Base model names used in metamodels.
# This is set based on validation set performance of the base models.
metamodel_base_model_names.model_name_overall_repertoire_composition = (
    "lasso_multiclass"
)
metamodel_base_model_names.model_name_convergent_clustering = "lasso_multiclass"
metamodel_base_model_names.model_name_sequence_disease = "lasso_multiclass"

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

classification_targets = list(TargetObsColumnEnum)  # Enable all
# # Or, temporarily subset to certain classification targets only
# classification_targets = [
#     TargetObsColumnEnum.disease,
# ]

default_embedder = "unirep_fine_tuned"  # or "unirep"

choosegpu.preferred_gpu_ids = [3]

#######

DATASET_VERSION = "20221224"

# Configure paths here.
# Then create all necessary directories with: `python scripts/make_dirs.py`
def make_paths(
    embedder,
    base_data_dir="data",
    base_output_dir="out",
    base_scratch_dir="/srv/scratch/$USER/",
    relative_to_path=None,
    dataset_version: Optional[str] = None,
):
    """Configure file paths. Pass in embedder type to generate embedder type-specific paths"""
    paths = SimpleNamespace()
    paths.base_data_dir = Path(base_data_dir)
    paths.base_output_dir = Path(base_output_dir)
    paths.base_scratch_dir = Path(base_scratch_dir)

    if dataset_version is None:
        dataset_version = DATASET_VERSION

    # ********************************************************************************************************************

    ### General:
    paths.metadata_dir = "metadata"

    # Set logdir to something Github Actions CI can access,
    # and adjust ci.yaml accordingly to ensure this directory exists
    paths.log_dir = paths.base_data_dir / "logs"

    paths.tests_snapshot_dir = "tests/snapshot"
    paths.transformers_cache = paths.base_data_dir / "transformers_cache/"

    # Local cache of network drive files
    paths.local_machine_cache_dir = paths.base_scratch_dir / "cache"

    # ********************************************************************************************************************

    ### Outputs:

    # Not versioned to the data, and not associated with a particular embedder:
    #

    # Not versioned to the data, but associated with a particular embedder:
    paths.output_dir = paths.base_output_dir / f"{embedder.name}"
    # Model 1
    paths.repertoire_stats_classifier_output_dir = "out/repertoire_stats"
    # Model 2
    paths.convergent_clusters_output_dir = "out/convergent_clusters"
    paths.exact_matches_output_dir = "out/exact_matches"
    # Model 3
    paths.sequence_models_output_dir = paths.output_dir / "sequence_models"
    # Metamodel
    paths.second_stage_blending_metamodel_output_dir = (
        paths.output_dir / "blending_metamodel"
    )
    # interpretations
    paths.model_interpretations_output_dir = paths.output_dir / "interpretations"
    # visualizations - main outputs:
    paths.supervised_embedding_output_dir = paths.output_dir / "supervised_embedding"
    # external cohort evaluations
    paths.external_cohort_evaluation_output_dir = (
        paths.output_dir / "external_cohort_evaluation"
    )

    # ********************************************************************************************************************

    ### Raw data and non-embedder models:
    ## Versioned with the data, but not associated with a particular embedder:
    paths.data_versioned = paths.base_data_dir / f"data_v_{dataset_version}"
    paths.scratch_versioned = paths.base_scratch_dir / f"data_v_{dataset_version}"

    # all sequences
    paths.sequences = paths.data_versioned / f"sequences.parquet"
    # only selected clones
    paths.sequences_sampled = paths.data_versioned / f"sequences.sampled.parquet"

    paths.dataset_specific_metadata = paths.data_versioned / "metadata"

    paths.external_raw_data = paths.base_data_dir / "external_cohorts" / "raw_data"
    paths.external_processed_data = (
        paths.data_versioned / "external_cohorts_part_tables"
    )

    # Model 1
    paths.repertoire_stats_classifier_models_dir = (
        paths.data_versioned / "repertoire_stats"
    )
    # Model 2
    paths.convergent_clusters_models_dir = paths.data_versioned / "convergent_clusters"
    paths.exact_matches_models_dir = paths.data_versioned / "exact_matches"

    # dataset EDA
    paths.dotplots_input = paths.data_versioned / "dotplots" / "input"
    paths.dotplots_output = paths.data_versioned / "dotplots" / "output"

    # ********************************************************************************************************************

    ### Embedder-specific data and models:
    ## Versioned with the data and specific to an embeddder:
    paths.anndatas_scratch_dir = paths.scratch_versioned / f"{embedder.name}/anndatas"
    paths.base_embedder_data_dir = paths.data_versioned / f"embedded/{embedder.name}"

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

    # Metamodel
    paths.external_data_embeddings = (
        paths.base_embedder_data_dir / "external_cohort_embeddings"
    )

    # Store oversized outputs here
    paths.high_res_outputs_dir = paths.base_embedder_data_dir / "high_res_outputs"
    # visualizations - foreground overlays only:
    paths.supervised_embedding_foreground_output_dir = (
        paths.base_embedder_data_dir / "supervised_embedding"
    )

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


def choose_embedder():
    return embedders.get_embedder_by_name(os.getenv("EMBEDDER_TYPE", default_embedder))


embedder = choose_embedder()
paths = make_paths(embedder=embedder)


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


dask_dashboard_port = 51126

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


# Choose some "pure" specimens for the peak-timepoints-only training set
# Used by helpers.get_all_specimen_info()
subtypes_keep = [
    # Covid19-buffycoat (Cell Host + Microbe):
    "Covid19 - Sero-positive (ICU)",
    "Covid19 - Sero-positive (Admit)",
    # Covid19-Seattle:
    "Covid19 - Acute 2",
    # Covid19-Stanford:
    "Covid19 - Admit",
    "Covid19 - ICU",
]
# Don't do any disease-subtype filtering for these diseases
# Keep all of their subtypes in "pure" set:
diseases_to_keep_all_subtypes = [
    "Healthy/Background",
    "HIV",
    "Lupus",
]


def acute_disease_choose_most_peak_timepoint(df: pd.DataFrame) -> pd.Index:
    """
    For a single study, choose among the specimens that have passed quality filters:
    Choose the single most peak timepoint specimen for each participant. Ideally closest to day 15.
    But ok with anything from day 10 to day 40 after onset of symptoms.

    Ensure that index is unique so that groupby->idxmin works.
    """
    subset = df[
        (df["specimen_time_point_days"] >= 7) & (df["specimen_time_point_days"] <= 40)
    ].copy()

    # Compute difference from day 15
    subset["diff_from_15"] = (subset["specimen_time_point_days"] - 15).abs()

    # Minimize difference from day 15 for each individual
    selected_index = subset.groupby("participant_label")["diff_from_15"].idxmin()

    return selected_index


# Map study name to filtering function
study_names_with_special_peak_timepoint_filtering: Dict[
    str, Callable[[pd.DataFrame], pd.Index]
] = {
    "Covid19-buffycoat": acute_disease_choose_most_peak_timepoint,
    "Covid19-Stanford": acute_disease_choose_most_peak_timepoint,
    # Covid19-Seattle does not have granular timepoint data
}

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
