"""
Supervised embeddings, per patient

Patients are separated by fold, because:

- Patients in different folds are processed by separate models and thus are in separately created latent spaces that can't be stitched together in plots.
- If using a fine-tuned language model like here, that is also dependent on the fold, since it was trained on one fold's train-smaller set.


"""


import gc
import click
from typing import List

from malid import config, helpers, supervised_embedding, cli_utils, io
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
    UmapSupervisionStrategy,
)

import logging

logger = logging.getLogger(__name__)

###

output_dir = config.paths.supervised_embedding_output_dir
foreground_output_dir = config.paths.supervised_embedding_foreground_output_dir

# Don't use a decision-threshold-tuned lasso sequence model for calling labels of each sequence and doing supervised embedding,
# because tuning against noisy sequence labels is dangerous
individual_sequence_model_name = (
    "lasso_multiclass"  # instead of "lasso_multiclass.decision_thresholds_tuned"
)

metamodel_name = "lasso_cv"

target_obs_column: TargetObsColumnEnum = TargetObsColumnEnum.disease
# the sample_weight_strategy will not be used to reweight sequences for the sequence-level UMAP,
# but will be used to load the sequence-level model that was trained with knoweldge of sequence weights,
# and to reweight sequences for the specimen-level processing in the metamodel and in the rollup model
sample_weight_strategy: SampleWeightStrategy = SampleWeightStrategy.ISOTYPE_USAGE


@click.command()
@cli_utils.accepts_gene_loci
@cli_utils.accepts_fold_ids
def run(
    gene_locus: List[GeneLocus],
    fold_ids: List[int],
):
    # input arguments are lists.

    gene_locus: GeneLocus = GeneLocus.combine_flags_list_into_single_multiflag_value(
        gene_locus
    )
    click.echo(f"Selected gene_locus: {gene_locus}")
    click.echo(f"Selected fold_ids: {fold_ids}")
    click.echo(f"Output dirs: {output_dir}, {foreground_output_dir}")

    ###

    ## Get test specimens list
    specimen_metadata = helpers.get_all_specimen_info()
    specimen_metadata = specimen_metadata[
        specimen_metadata["cohort"] == "Boydlab"
    ].copy()

    # %%
    # extract and reformat day timepoint (unless "peak" or "3mth" or similar)
    day_timepoints = specimen_metadata["specimen_time_point"].fillna("")

    # check if "N days" format (as opposed to "Nmth" or "peak"): remove "days", strip whitespace, and confirm all remaining characters are digits
    is_numeric_days = day_timepoints.str.replace("days", "").str.strip().str.isnumeric()

    # extract day number from "N days" format
    specimen_metadata["timepoint_formatted"] = "day " + (
        day_timepoints[is_numeric_days].str.extract("(\d+)")
    )

    # fill in remaining other-format timepoints
    specimen_metadata["timepoint_formatted"].fillna(
        specimen_metadata["specimen_time_point"], inplace=True
    )

    # there still may be NaN timepoints, e.g. for a "Healthy/Background - CMV Unknown" specimen
    # fill those in
    specimen_metadata["timepoint_formatted"].fillna("Unknown", inplace=True)

    # replace specimen_time_point for the "N days" samples with N, or original timepoint for other formats
    specimen_metadata["specimen_time_point"] = (
        day_timepoints[is_numeric_days].str.extract("(\d+)").astype(int)
    )
    specimen_metadata["specimen_time_point"].fillna(
        specimen_metadata["timepoint_formatted"], inplace=True
    )

    assert not specimen_metadata["is_peak"].isna().any()

    specimen_metadata = specimen_metadata[
        [
            "participant_label",
            "specimen_label",
            "specimen_time_point",
            "timepoint_formatted",
            "is_peak",
            "disease.separate_past_exposures",
        ]
    ].sort_values(["participant_label", "specimen_time_point"])
    click.echo(f"Loaded metadata shape: {specimen_metadata.shape}")

    ###

    # Run

    for fold_id in fold_ids:
        for single_gene_locus in gene_locus:
            try:
                logger.info(
                    f"Processing fold {fold_id}, {single_gene_locus} with embedder {config.embedder.name}"
                )
                supervised_embedding.run_for_fold(
                    fold_id=fold_id,
                    gene_locus=single_gene_locus,
                    specimen_metadata=specimen_metadata,
                    metamodel_name=metamodel_name,
                    individual_sequence_model_name=individual_sequence_model_name,
                    output_dir=output_dir,
                    foreground_output_dir=foreground_output_dir,
                    # which fold label to plot as background reference map
                    background_fold_label="validation",  # train_smaller
                    # which fold label to use as foreground overlay for peak timepoints
                    foreground_peakset_fold_label="test",
                    # whether to plot people that are ONLY in peak timepoint set
                    foreground_include_peak=True,
                    # whether to plot people that are ONLY in offpeak timepoint set
                    foreground_include_offpeak=True,
                    # whether to plot offpeak patients whose diseases are never in training set
                    offpeak_include_diseases_outside_training_set=False,
                    supervision_type=UmapSupervisionStrategy.FULLY_SUPERVISED,
                    # one of these two must be provided: take top fraction or top N from each class
                    # (the higher the number, the weaker the filter)
                    high_confidence_top_percent_per_class=0.3,
                    high_confidence_top_n_per_class=None,
                    #
                    filter_naive_isotypes=False,
                    # subset each class's high-confidence set further: min diff between top two predicted probabilities
                    # (the higher the number, the stronger the filter)
                    difference_between_top_two_predicted_probas_cutoff=0.05,
                    # then this can optionally be provided to subsample a certain N at random from the set chosen above
                    subsample_n_from_chosen=10000,
                    ##
                    target_obs_column=target_obs_column,
                    sample_weight_strategy=sample_weight_strategy,
                )
            except Exception as err:
                logger.exception(
                    f"Fold {fold_id}, gene_locus={single_gene_locus} failed with error: {err}"
                )

            # Clear cache and garbage collect again, just in case
            io.clear_cached_fold_embeddings()
            gc.collect()


if __name__ == "__main__":
    run()
