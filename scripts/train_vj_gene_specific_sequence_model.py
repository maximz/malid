"""
Train sequence model.
Defaults to all folds and all TargetObsColumns, one at a time.

Usage examples:
    > python scripts/train_vj_gene_specific_sequence_model.py;
    > python scripts/train_vj_gene_specific_sequence_model.py --target_obs_column disease --target_obs_column disease_all_demographics_present;
    > python scripts/train_vj_gene_specific_sequence_model.py --target_obs_column disease --fold_id 0 --fold_id 1;
    > python scripts/train_vj_gene_specific_sequence_model.py --target_obs_column disease --fold_id 0;
"""

from typing import List
import gc

from malid import io, config, cli_utils
from malid.train import train_vj_gene_specific_sequence_model
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
import click
import time

from enumchoice import EnumChoice
from malid.trained_model_wrappers.vj_gene_specific_sequence_classifier import (
    SequenceSubsetStrategy,
)

import logging

logger = logging.getLogger(__name__)

# Base sequence model varieties to train
model_names_to_train = [
    # 1) Subset of config.model_names_to_train:
    # (Training is fast because the sequence category subsets are relatively small, so we can afford to throw in some more complex models)
    "dummy_most_frequent",
    "dummy_stratified",
    "rf_multiclass",
    #
    "lasso_cv",
    # "elasticnet_cv0.75",
    "elasticnet_cv",
    # "elasticnet_cv0.25",
    "ridge_cv",
    #
    "lasso_cv_patient_level_optimization",
    # "elasticnet_cv0.75_patient_level_optimization",
    "elasticnet_cv_patient_level_optimization",
    # "elasticnet_cv0.25_patient_level_optimization",
    "ridge_cv_patient_level_optimization",
    #
    #
    "linearsvm_ovr",
    ###
    # 2) Specialized extras
    "elasticnet_cv_ovr",
    # "elasticnet_cv0.75_ovr",
    # "elasticnet_cv0.25_ovr",
    "lasso_cv_ovr",
    "ridge_cv_ovr",
]

# Extend the list of model names with variahts will have been generated during the training process, e.g. add the sklearn derivatives of the glmnet models
# This is intended for downstream users to be aware of which model names exist.
resulting_trained_models = train_vj_gene_specific_sequence_model._extend_model_list_with_sklearn_versions_of_glmnet_models(
    model_names_to_train
)


def filter_sequence_level_model_names(
    sequence_model_names: List[str],
    gene_locus: GeneLocus,
    preferred_only: bool,
) -> List[str]:
    """
    Optionally train or use only the sequence model name that is used downstream in the metamodel
    (By default, many models will be trained for performance comparison)
    """
    if not preferred_only:
        # Default: train/use all base sequence-level model names
        return sequence_model_names

    # Filter down to preferred model name only. It is specific to the gene locus.
    preferred_model = config.metamodel_base_model_names.base_sequence_model_name[
        gene_locus
    ]
    if preferred_model not in sequence_model_names:
        raise ValueError(f"Preferred model {preferred_model} unavailable to train")
    return [preferred_model]


def run_standard_train(
    gene_loci_used: GeneLocus,
    target_obs_columns: List[TargetObsColumnEnum],
    fold_ids: List[int],
    composite_sample_weight_strategy: SampleWeightStrategy,
    sequence_subset_strategy: SequenceSubsetStrategy,
    n_jobs=1,
    exclude_rare_v_genes: bool = True,
    resume: bool = False,
    train_preferred_model_only: bool = False,
):
    """
    For each fold ID, for each targetobscolumn, for each sample weight strategy, run one train cycle.
    """

    # Model 3, separated by V and J genes:
    training_fold_name = "train_smaller1"  # Train on portion of train_smaller used to train base sequence model
    testing_fold_name = "train_smaller2"  # Evaluate on portion of train_smaller used to train rollup model on top of sequence model

    # map target_obs_column to model_output_prefix
    # Predict on "disease" (standard target) or some alternative target. This means we load an extra obs column, subset to specimens with that obs column defined, and restart from raw and rescale X.

    GeneLocus.validate(gene_loci_used)  # packed (by OR) into single item here
    SampleWeightStrategy.validate(
        composite_sample_weight_strategy
    )  # packed (by OR) into single item here

    for gene_locus in gene_loci_used:
        chosen_models = filter_sequence_level_model_names(
            sequence_model_names=model_names_to_train,
            gene_locus=gene_locus,
            preferred_only=train_preferred_model_only,
        )

        # Control fold_id and cache manually so that we limit repetitive I/O
        for fold_id in fold_ids:
            # Predict on "disease" (standard target) or some alternative target. This means we load an extra obs column, subset to specimens with that obs column defined, and restart from raw and rescale X.
            for target_obs_col in target_obs_columns:
                GeneLocus.validate_single_value(gene_locus)
                TargetObsColumnEnum.validate(target_obs_col)

                try:
                    logger.info(
                        f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={composite_sample_weight_strategy}"
                    )
                    train_itime = time.time()
                    # note that sequence data is already scaled automatically
                    train_vj_gene_specific_sequence_model.run_classify_with_all_models(
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_col,
                        sample_weight_strategy=composite_sample_weight_strategy,
                        fold_label_train=training_fold_name,
                        fold_label_test=testing_fold_name,
                        chosen_models=chosen_models,
                        n_jobs=n_jobs,
                        sequence_subset_strategy=sequence_subset_strategy,
                        exclude_rare_v_genes=exclude_rare_v_genes,
                        resume=resume,
                        # control fold_id and cache manually so that we limit repetitive I/O
                        fold_ids=[fold_id],
                        clear_cache=False,
                    )
                    train_etime = time.time()
                    logger.info(
                        f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={composite_sample_weight_strategy} completed in {train_etime - train_itime}"
                    )

                except Exception as err:
                    logger.exception(
                        f"fold {fold_id} - target={target_obs_col}, gene_locus={gene_locus}, sample_weight_strategy={composite_sample_weight_strategy} failed with error: {err}"
                    )

            # manage fold ID and cache manually: now that we are done with this fold, clear cache
            io.clear_cached_fold_embeddings()
            gc.collect()


@click.command()
@cli_utils.accepts_gene_loci
@cli_utils.accepts_target_obs_columns
@cli_utils.accepts_fold_ids
@cli_utils.accepts_sample_weight_strategies
@cli_utils.accepts_n_jobs
@click.option(
    "--sequence-subset-strategy",
    type=EnumChoice(SequenceSubsetStrategy, case_sensitive=False),
    default=config.metamodel_base_model_names.base_sequence_model_subset_strategy,
    show_default=True,
    help="Sequence splitting strategy used when training sequence models.",
)
@click.option(
    "--exclude-rare-v-genes/--include-rare-v-genes",
    is_flag=True,
    show_default=True,
    default=True,
    help="Exclude rare V genes. This parameter is ignored and this functionality is disabled (i.e. no extra filtering applied) if the sequence subset strategy does not include splitting by V gene.",
)
@click.option(
    "--resume/--overwrite",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optionally resume training an incomplete run by passing --resume. By default, all incomplete model training outputs are overwritten immediately and we retrain everything from scratch.",
)
@click.option(
    "--train-preferred-model-only",
    is_flag=True,
    show_default=True,
    default=False,
    help="Optionally train only the model name that is used downstream in the metamodel. By default, many models will be trained for performance comparison.",
)
def run(
    gene_locus: List[GeneLocus],
    target_obs_column: List[TargetObsColumnEnum],
    fold_ids: List[int],
    sample_weight_strategy: List[SampleWeightStrategy],
    n_jobs: int,
    sequence_subset_strategy: SequenceSubsetStrategy,
    exclude_rare_v_genes: bool = True,
    resume: bool = False,
    train_preferred_model_only: bool = False,
):
    # Input arguments are lists. Combine them into multipacked flag values.
    gene_locus: GeneLocus = GeneLocus.combine_flags_list_into_single_multiflag_value(
        gene_locus
    )
    sample_weight_strategy: SampleWeightStrategy = (
        SampleWeightStrategy.combine_flags_list_into_single_multiflag_value(
            sample_weight_strategy
        )
    )

    click.echo(f"Selected gene_locus: {gene_locus}")
    click.echo(f"Selected target_obs_columns: {target_obs_column}")
    click.echo(f"Selected fold_ids: {fold_ids}")
    click.echo(f"Selected sample_weight_strategies: {sample_weight_strategy}")
    click.echo(f"Selected n_jobs: {n_jobs}")
    click.echo(f"Selected sequence_subset_strategy: {sequence_subset_strategy}")
    click.echo(f"Selected exclude_rare_v_genes: {exclude_rare_v_genes}")
    click.echo(f"resume setting: {resume}")
    click.echo(f"train_preferred_model_only setting: {train_preferred_model_only}")

    run_standard_train(
        gene_loci_used=gene_locus,
        target_obs_columns=target_obs_column,
        fold_ids=fold_ids,
        composite_sample_weight_strategy=sample_weight_strategy,
        sequence_subset_strategy=sequence_subset_strategy,
        n_jobs=n_jobs,  # Inner V-J gene specific fits to run in parallel across V-J gene subsets.
        exclude_rare_v_genes=exclude_rare_v_genes,
        resume=resume,
        train_preferred_model_only=train_preferred_model_only,
    )


if __name__ == "__main__":
    run()
