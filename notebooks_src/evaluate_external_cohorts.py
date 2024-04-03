# %%
from collections import defaultdict
import numpy as np
import pandas as pd
import genetools
import scanpy as sc
from IPython.display import display, Markdown
from typing import List, Tuple, Dict
from typing import Optional
import os
import gc
from pathlib import Path

from malid import config, io, logger
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    SampleWeightStrategy,
    CrossValidationSplitStrategy,
    healthy_label,
)
from malid.trained_model_wrappers import BlendingMetamodel
import crosseval
from crosseval import FeaturizedData
from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)
from kdict import kdict
from slugify import slugify

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# We only support one split strategy here.
assert (
    config.cross_validation_split_strategy
    == CrossValidationSplitStrategy.in_house_peak_disease_timepoints
)

# %%
gene_loci = config.gene_loci_used
print(gene_loci)

# %%
evaluation_target = TargetObsColumnEnum.disease
# Also use an alternate evaluation_target that support demographics_only metamodel
evaluation_target_with_demographics = (
    TargetObsColumnEnum.disease_all_demographics_present
)

# %%
metamodel_names = ["ridge_cv", "elasticnet_cv", "lasso_cv", "rf_multiclass"]
base_model_train_fold_name = "train_smaller"
metamodel_fold_label_train = "validation"
metamodel_fold_label_test = "external"  # what we evaluate on

# %%

# %%
import dataclasses


def convert_abstentions_to_healthy(
    model_single_fold_performance: crosseval.ModelSingleFoldPerformance,
    fill_in_class: str,
) -> crosseval.ModelSingleFoldPerformance:
    """Convert abstentions to healthy with Pr(healthy) = 1, all other class probabilities zero. Healthy label indicated by fill_in_class parameter."""
    # Based on how we do the inverse in crosseval.ModelSingleFoldPerformance.apply_abstention_mask()
    return dataclasses.replace(
        model_single_fold_performance,
        # Pass in null InitVars to make a copy
        clf=None,
        X_test=None,
        # Make changes
        y_true=np.hstack(
            [
                model_single_fold_performance.y_true,
                model_single_fold_performance.test_abstentions,
            ]
        ),
        y_pred=np.hstack(
            [
                model_single_fold_performance.y_pred,
                [fill_in_class] * model_single_fold_performance.n_abstentions,
            ]
        ),
        X_test_shape=(
            model_single_fold_performance.X_test_shape[0]
            + model_single_fold_performance.n_abstentions,
            model_single_fold_performance.X_test_shape[1],
        )
        if model_single_fold_performance.X_test_shape is not None
        else None,
        # Remove logits, not sure how to update them:
        y_decision_function=None,
        # Update y_preds_proba:
        # TODO: In binary setting, are y_preds_proba a single column rather than 2 columns?
        y_preds_proba=pd.concat(
            [
                pd.DataFrame(
                    model_single_fold_performance.y_preds_proba,
                    columns=model_single_fold_performance.class_names,
                ),
                pd.DataFrame(
                    {fill_in_class: [1.0] * model_single_fold_performance.n_abstentions}
                ).reindex(
                    columns=model_single_fold_performance.class_names, fill_value=0.0
                ),
            ],
            axis=0,
        )
        if model_single_fold_performance.y_preds_proba is not None
        else None,
        test_metadata=pd.concat(
            [
                model_single_fold_performance.test_metadata,
                model_single_fold_performance.test_abstention_metadata,
            ],
            axis=0,
        ),
        test_sample_weights=np.hstack(
            [
                model_single_fold_performance.test_sample_weights,
                model_single_fold_performance.test_abstention_sample_weights,
            ]
        )
        if model_single_fold_performance.test_sample_weights is not None
        and model_single_fold_performance.test_abstention_sample_weights is not None
        else None,
        test_abstentions=None,  # This triggers default factory
        test_abstention_metadata=None,  # This triggers default factory
        test_abstention_sample_weights=None,
        # All other fields stay the same
    )


# %%

# %%
def run(
    gene_locus: GeneLocus,
    classification_target: TargetObsColumnEnum,
    output_dir: Optional[Path] = None,
    metamodel_flavor: str = "default",
) -> Tuple[crosseval.ExperimentSetGlobalPerformance, FeaturizedData]:
    """Load and run metamodel for a single locus or multiple loci"""
    display(
        Markdown(
            f"## {gene_locus}, {classification_target}, metamodel flavor {metamodel_flavor}"
        )
    )

    # Load external cohort data, which at the moment has completely different specimens in the BCR vs TCR sets, so we will not use BCR+TCR metamodels.
    GeneLocus.validate_single_value(gene_locus)
    adata = io.load_fold_embeddings(
        fold_id=-1,
        fold_label=metamodel_fold_label_test,
        gene_locus=gene_locus,
        target_obs_column=classification_target,
    )

    # Load the metamodels
    clfs = {}
    for metamodel_name in metamodel_names:
        clfs[metamodel_name] = BlendingMetamodel.from_disk(
            fold_id=-1,
            metamodel_name=metamodel_name,
            base_model_train_fold_name=base_model_train_fold_name,
            metamodel_fold_label_train=metamodel_fold_label_train,
            gene_locus=gene_locus,
            target_obs_column=classification_target,
            metamodel_flavor=metamodel_flavor,
        )

    # Load data and featurize.
    # Assumption: all metamodel names for the same metamodel flavor will produce the same featurization, because they all use the same base models and feature transforms.
    # So we only need to featurize once per metamodel flavor.
    # Just choose the first loaded metamodel and featurize with it. It will be the same as featurizing with a different loaded metamodel.
    featurized = next(iter(clfs.values())).featurize({gene_locus: adata})
    # Handle abstention
    if featurized.X.shape[0] == 0:
        raise ValueError("All abstained")

    # garbage collect
    del adata
    gc.collect()

    ## Decision threshold tuning:
    # Split parts of featurized data for optional decision threshold tuning.
    # (There's no splitting or tuning in the "untuned.all_data" results, if you want to ignore this.)

    # Set a random seed for reproducibility
    np.random.seed(42)

    # 'featurized' is a single FeaturizedData object for all participants
    metadata = featurized.metadata
    X = featurized.X
    y = featurized.y
    sample_names = featurized.sample_names

    # Extract unique participant labels
    participant_labels = metadata["participant_label"].unique()

    # Map each participant to their specimen indices (ilocs)
    participant_to_specimen_indices = {
        participant: np.where(metadata["participant_label"] == participant)[0]
        for participant in participant_labels
    }

    # Map each disease to participant labels
    disease_to_participants = (
        metadata.groupby(classification_target.value.obs_column_name)[
            "participant_label"
        ]
        .unique()
        .to_dict()
    )

    # Calculate the number of participants to take for validation per disease
    # Choose how many we will split off for validation. Take 30% of smallest class (measured by # unique patients) - constrained to be between 2 to 10 samples per class.
    amount_to_take = {
        disease: int(min(max(0.3 * len(participants), 2), 10))
        for disease, participants in disease_to_participants.items()
    }
    logger.info(
        f"{gene_locus}, {classification_target}, metamodel flavor {metamodel_flavor}: splitting off {amount_to_take} participants per class for decision threshold tuning."
    )

    # Split participants into validation and test sets for each disease
    validation_indices, test_indices = [], []

    for disease, participants in disease_to_participants.items():
        np.random.shuffle(participants)
        validation_participants = participants[: amount_to_take[disease]]
        test_participants = participants[amount_to_take[disease] :]

        for participant in validation_participants:
            validation_indices.append(participant_to_specimen_indices[participant])

        for participant in test_participants:
            test_indices.append(participant_to_specimen_indices[participant])

    validation_indices = np.hstack(validation_indices)
    test_indices = np.hstack(test_indices)

    # Sanity checks
    assert len(set(validation_indices).intersection(test_indices)) == 0
    assert len(validation_indices) + len(test_indices) == len(metadata)

    # Convert indices to arrays
    validation_indices = np.array(validation_indices)
    test_indices = np.array(test_indices)

    # Extract validation and test sets
    X_validation = (
        X.iloc[validation_indices]
        if isinstance(X, pd.DataFrame)
        else X[validation_indices]
    )
    y_validation = (
        y.iloc[validation_indices]
        if isinstance(y, pd.Series)
        else y[validation_indices]
    )
    metadata_validation = metadata.iloc[validation_indices]
    sample_names_validation = None
    if sample_names is not None:
        sample_names_validation = (
            sample_names.iloc[validation_indices]
            if isinstance(sample_names, pd.Series)
            else sample_names[validation_indices]
        )

    X_test = X.iloc[test_indices] if isinstance(X, pd.DataFrame) else X[test_indices]
    y_test = y.iloc[test_indices] if isinstance(y, pd.Series) else y[test_indices]
    metadata_test = metadata.iloc[test_indices]
    sample_names_test = None
    if sample_names is not None:
        sample_names_test = (
            sample_names.iloc[test_indices]
            if isinstance(sample_names, pd.Series)
            else sample_names[test_indices]
        )

    # Create FeaturizedData objects for validation and test. Add all abstentions to test set.
    featurized_validation = FeaturizedData(
        X=X_validation,
        y=y_validation,
        sample_names=sample_names_validation,
        metadata=metadata_validation,
    )
    featurized_test = FeaturizedData(
        X=X_test,
        y=y_test,
        sample_names=sample_names_test,
        metadata=metadata_test,
        abstained_sample_names=featurized.abstained_sample_names,
        abstained_sample_metadata=featurized.abstained_sample_metadata,
        abstained_sample_y=featurized.abstained_sample_y,
    )

    ## Run each metamodel

    # Create final result containers, for all metamodels
    results = crosseval.ExperimentSet()

    for metamodel_name, clf in clfs.items():
        # Tune model
        clf_tuned = AdjustedProbabilitiesDerivedModel.adjust_model_decision_thresholds(
            model=clf,
            X_validation=featurized_validation.X,
            y_validation_true=featurized_validation.y,
        )

        # Do evaluation
        for transformed_clf, transformed_model_name, fd in zip(
            [clf, clf, clf_tuned],
            [
                metamodel_name + ".untuned.all_data",
                metamodel_name + ".untuned.test_subset",
                metamodel_name + ".tuned.test_subset",
            ],
            [featurized, featurized_test, featurized_test],
        ):
            results.add(
                crosseval.ModelSingleFoldPerformance(
                    model_name=transformed_model_name,
                    fold_id=-1,
                    y_true=fd.y,
                    clf=transformed_clf,
                    X_test=fd.X,
                    fold_label_train=metamodel_fold_label_train,
                    fold_label_test=metamodel_fold_label_test,
                    test_metadata=fd.metadata,
                    test_abstentions=fd.abstained_sample_y,
                    test_abstention_metadata=fd.abstained_sample_metadata,
                )
            )

    # Convert abstentions to healthy with Pr(healthy) = 1, all other class probabilities zero. Add these as extra model names.
    original_keys = list(results.model_outputs.keys())
    for key in original_keys:  # Don't iterate over dict that's actively changing
        original_model_single_fold_performance = results.model_outputs[key]
        if original_model_single_fold_performance.n_abstentions == 0:
            # No abstentions. Skip.
            continue
        modified_model_single_fold_performance = convert_abstentions_to_healthy(
            model_single_fold_performance=original_model_single_fold_performance,
            fill_in_class=healthy_label,
        )
        # Also adjust model name
        modified_model_single_fold_performance = dataclasses.replace(
            modified_model_single_fold_performance,
            model_name=modified_model_single_fold_performance.model_name
            + ".abstentions_converted_to_healthy",
        )
        # Store the modified version under its new model name
        results.add(modified_model_single_fold_performance)

    results = results.summarize()
    combined_stats = results.get_model_comparison_stats(sort=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        results.export_all_models(
            func_generate_classification_report_fname=lambda model_name: output_dir
            / f"classification_report.{model_name}.txt",
            func_generate_confusion_matrix_fname=lambda model_name: output_dir
            / f"confusion_matrix.{model_name}.png",
            dpi=300,
        )
        combined_stats.to_csv(
            output_dir / "compare_model_scores.tsv",
            sep="\t",
        )

    display(combined_stats)
    for model_name, model_perf in results.model_global_performances.items():
        print(model_name)
        print(model_perf.full_report())
        display(model_perf.confusion_matrix_fig())
        print()
        print("*" * 60)
        print()

    return results, featurized


# %%

# %% [markdown]
# # External cohorts similar to our sequencing process

# %%
results_by_gene_locus = {}
featurized_by_gene_locus = {}

for gene_locus in gene_loci:
    # run on single locus

    # default flavor:
    results, featurized = run(
        gene_locus=gene_locus,
        classification_target=evaluation_target,
        output_dir=config.paths.external_cohort_evaluation_output_dir
        / "default"
        / gene_locus.name,
    )
    if results is not None:
        results_by_gene_locus[gene_locus] = results
        featurized_by_gene_locus[gene_locus] = featurized

    ## models 1+3 only:
    display(Markdown("## Metamodel with Models 1 + 3 only for comparison"))
    run(
        gene_locus=gene_locus,
        classification_target=evaluation_target,
        output_dir=config.paths.external_cohort_evaluation_output_dir
        / "subset_of_submodels_repertoire_stats_sequence_model"
        / gene_locus.name,
        metamodel_flavor="subset_of_submodels_repertoire_stats_sequence_model",
    )

    ## models 1+2 only:
    display(Markdown("## Metamodel with Models 1 + 2 only for comparison"))
    run(
        gene_locus=gene_locus,
        classification_target=evaluation_target,
        output_dir=config.paths.external_cohort_evaluation_output_dir
        / "subset_of_submodels_repertoire_stats_convergent_cluster_model"
        / gene_locus.name,
        metamodel_flavor="subset_of_submodels_repertoire_stats_convergent_cluster_model",
    )

    ## models 2+3 only:
    display(Markdown("## Metamodel with Models 2 + 3 only for comparison"))
    run(
        gene_locus=gene_locus,
        classification_target=evaluation_target,
        output_dir=config.paths.external_cohort_evaluation_output_dir
        / "subset_of_submodels_convergent_cluster_model_sequence_model"
        / gene_locus.name,
        metamodel_flavor="subset_of_submodels_convergent_cluster_model_sequence_model",
    )

    ## demographics only:
    display(Markdown("## Demographics-only metamodel for comparison"))
    run(
        gene_locus=gene_locus,
        classification_target=evaluation_target_with_demographics,
        output_dir=config.paths.external_cohort_evaluation_output_dir
        / "demographics_only"
        / gene_locus.name,
        metamodel_flavor="demographics_only",
    )

    io.clear_cached_fold_embeddings()

# %%

# %%

# %%
# Visualize subcomponent predictions
# (Have to use this awkward kdict notation because GeneLocus flag-enum key is not well handled)
for gene_locus, featurized in featurized_by_gene_locus.items():
    output_dir = (
        config.paths.high_res_outputs_dir
        / "external_cohort_evaluation"
        / "default"
        / gene_locus.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    display(Markdown(f"### {gene_locus}"))
    display(featurized.X)

    for col in featurized.X.columns:
        fig = plt.figure()
        sns.boxplot(
            data=pd.concat([featurized.X[col], featurized.metadata], axis=1),
            x=col,
            y="disease",
        )
        genetools.plots.savefig(
            fig, output_dir / f"subcomponent.{slugify(col)}.png", dpi=300
        )
        display(fig)
        plt.close(fig)
    display(Markdown("---"))

# %%

# %%
