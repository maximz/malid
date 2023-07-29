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
from malid.datamodels import GeneLocus, TargetObsColumnEnum, SampleWeightStrategy
from malid.trained_model_wrappers import BlendingMetamodel
from malid.external import model_evaluation
from malid.external.model_evaluation import FeaturizedData
from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)
from kdict import kdict
from slugify import slugify

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# %%

# %%
gene_loci = config.gene_loci_used
gene_loci

# %%
metamodel_names = [
    "lasso_cv",
    "ridge_cv",
    "elasticnet_cv",
    "lasso_multiclass",
    "xgboost",
    "linearsvm_ovr",
    "rf_multiclass",
]

# %%
evaluation_target = TargetObsColumnEnum.disease
# Also use an alternate evaluation_target that support demographics_only metamodel
evaluation_target_with_demographics = (
    TargetObsColumnEnum.disease_all_demographics_present
)


# %%

# %%
def list_external_cohort_samples(
    requested_gene_loci: GeneLocus,
    classification_target: TargetObsColumnEnum,
    different_platform: bool,
) -> Optional[pd.DataFrame]:
    """
    Get external cohort entries for a particular gene locus or set of gene loci.
    different_platform: if true, filter ONLY to different sequencing platforms; otherwise filter out different sequencing platforms.
    """
    external_cohort_specimens = pd.read_csv(
        config.paths.metadata_dir / "generated.external_cohorts.all_specimens.tsv",
        sep="\t",
    )
    # The gene_locus field is currently a string name corresponding to a GeneLocus enum
    # Change gene_locus field to be a true GeneLocus object
    external_cohort_specimens["gene_locus"] = external_cohort_specimens[
        "gene_locus"
    ].apply(lambda gene_locus_name: GeneLocus[gene_locus_name])

    ## Find specimens that only have exactly the locus/loci requested, and don't also have data from other loci.
    # The requested_gene_loci flag object may represent a single locus or multiple loci: BCR-only, TCR-only, or BCR+TCR.
    # - If requested gene_locus is BCR, return BCR-only specimens. Don't return specimens that are BCR+TCR.
    # - If requested gene_locus is TCR, return TCR-only specimens. Don't return specimens that are BCR+TCR.
    # - If requested gene_locus is BCR | TCR , return specimens that have BCR and TCR data. Don't return specimens that are only BCR or only TCR.

    # Get gene loci available for each specimen
    loci_for_each_specimen = external_cohort_specimens.groupby("specimen_label")[
        "gene_locus"
    ].unique()

    # keep these: available loci are the same as requested; specimens don't have data from other loci
    specimens_with_data_from_selected_loci = loci_for_each_specimen[
        loci_for_each_specimen.apply(lambda arr: set(arr) == set(requested_gene_loci))
    ].index.tolist()

    # subselect
    external_cohort_specimens = external_cohort_specimens[
        external_cohort_specimens["specimen_label"].isin(
            specimens_with_data_from_selected_loci
        )
    ]

    if external_cohort_specimens.shape[0] == 0:
        # No specimens for desired gene loci
        return None

    ## Other important filter: must exist
    # Get file name
    external_cohort_specimens["fname"] = external_cohort_specimens.apply(
        lambda row: config.paths.external_data_embeddings
        / f"{row['gene_locus'].name}"
        / f"{row['participant_label']}.h5",
        axis=1,
    )
    # Mask
    mask_file_exists = external_cohort_specimens["fname"].apply(os.path.exists)
    if (~mask_file_exists).sum() > 0:
        logger.warning(
            f"Skipping these missing specimens for {requested_gene_loci}: {external_cohort_specimens[~mask_file_exists]['specimen_label_by_locus'].values}"
        )
    # filter down
    external_cohort_specimens = external_cohort_specimens[mask_file_exists]

    ## Also filter out different sequencing platforms
    external_cohort_specimens = external_cohort_specimens[
        external_cohort_specimens["different_platform"] == different_platform
    ]

    # Filter out specimens that don't have demographic data, if requested by this classification_target
    if (
        classification_target.value.require_metadata_columns_present is not None
        and len(classification_target.value.require_metadata_columns_present) > 0
    ):
        # Filter out rows with obs[required_metadata_cols] has any NaNs
        external_cohort_specimens = external_cohort_specimens.loc[
            ~external_cohort_specimens[
                classification_target.value.require_metadata_columns_present
            ]
            .isna()
            .any(axis=1)
        ]

    if external_cohort_specimens.shape[0] == 0:
        # No specimens for desired gene loci
        return None

    return external_cohort_specimens


# %%
def load_external_cohort_anndatas(
    external_cohort_specimens: pd.DataFrame,
    sample_weight_strategy: SampleWeightStrategy = SampleWeightStrategy.ISOTYPE_USAGE,
):
    """Generator yielding external cohort participant anndatas"""
    # the data is stored as one embedded anndata per locus per participant (i.e. can have multiple specimens)
    for participant_label, grp in external_cohort_specimens.groupby(
        "participant_label", observed=True
    ):
        adatas_by_locus: Dict[GeneLocus, anndata.AnnData] = {}
        # subselect to one file name per locus per participant (that's all that is available)
        filename_by_locus: Dict[GeneLocus, List[str]] = (
            grp.groupby("gene_locus", sort=False)["fname"].unique().to_dict()
        )
        for gene_locus, fname_lst in filename_by_locus.items():
            assert (
                len(fname_lst) == 1
            ), "should only have one unique filename per participant per locus"
            fname = fname_lst[0]

            adata = sc.read(fname)
            adata.obs = adata.obs.assign(fold_id=-1, fold_label="external")

            # produces 'disease.separate_past_exposures' obs column.
            adata = io.label_past_exposures(adata)
            adata = io.fix_gene_names(adata)

            # load sequence weights for off-peak too
            if sample_weight_strategy == SampleWeightStrategy.ISOTYPE_USAGE:
                # calculate sample weights to balance out isotype proportions for each specimen
                adata.obs = genetools.helpers.merge_into_left(
                    adata.obs,
                    io.compute_isotype_sample_weights_column(adata).rename(
                        "sample_weight_isotype_rebalance"
                    ),
                )

            # Add demographics
            for colname in ["age", "sex", "ethnicity_condensed"]:
                # should be same for all specimens for this participant
                adata.obs[colname] = grp[colname].iloc[0]

            adatas_by_locus[gene_locus] = adata

        # For each participant, yield Tuple[participant_label string, Dict[GeneLocus, anndata.AnnData]]
        yield participant_label, adatas_by_locus


# %%

# %%
def run(
    gene_locus: GeneLocus,
    different_platform: bool,
    classification_target: TargetObsColumnEnum,
    output_dir: Optional[Path] = None,
    metamodel_flavor: str = "default",
) -> Tuple[model_evaluation.ExperimentSetGlobalPerformance, Dict[str, FeaturizedData]]:
    """Load and run metamodel for a single locus or multiple loci"""
    display(Markdown(f"## {gene_locus}"))

    external_cohort_specimens = list_external_cohort_samples(
        requested_gene_loci=gene_locus,
        classification_target=classification_target,
        different_platform=different_platform,
    )
    if external_cohort_specimens is None or external_cohort_specimens.shape[0] == 0:
        logger.warning(f"No external cohort specimens found for {gene_locus}")
        return None, None

    # Final result containers, for all metamodels
    results = model_evaluation.ExperimentSet()
    featurized_by_metamodel_name = {}

    # Load the metamodels
    clfs = {}
    for metamodel_name in metamodel_names:
        clfs[metamodel_name] = BlendingMetamodel.from_disk(
            fold_id=-1,
            metamodel_name=metamodel_name,
            base_model_train_fold_name="train_smaller",
            metamodel_fold_label_train="validation",
            gene_locus=gene_locus,
            target_obs_column=classification_target,
            metamodel_flavor=metamodel_flavor,
        )

    # Load data and featurize.
    # Store a List[FeaturizedData] per metamodel. (TODO: is this necessary? aren't the featurizations all going to be the same?)
    featurized_lists_per_metamodel: Dict[str, List[FeaturizedData]] = defaultdict(list)
    for participant_label, adata_by_locus in load_external_cohort_anndatas(
        external_cohort_specimens=external_cohort_specimens
    ):
        for metamodel_name, clf in clfs.items():
            # Featurize one participant.
            # (external cohort is too big to create an anndata with all participants)

            # adata_by_locus has input anndatas wrapped as Dict[GeneLocus, anndata.AnnData],
            # allowing use of single-locus or multi-locus metamodel.
            featurized_lists_per_metamodel[metamodel_name].append(
                clf.featurize(adata_by_locus)
            )

        # garbage collect
        del adata_by_locus
        gc.collect()

    ## Split out some featurized data for optional decision threshold tuning (old, ignore this, just use the "untuned.all_data" results.)

    # All FeaturizedData lists in featurized_lists_per_metamodel are ordered consistently between metamodels,
    # so we can use any to determine the split indices:
    one_featurized_list: List[FeaturizedData] = next(
        iter(featurized_lists_per_metamodel.values())
    )

    # Each participant has one FeaturizedData object,
    # though some participants may have multiple specimens (i.e. the single merged FeaturizedData object has shape[0] > 1),
    # and some participants may have no specimens due to abstention (i.e. the object has shape[0] == 0).

    # Sanity checks:
    # Each FeaturizedData has only a single unique participant label (or is empty due to abstentions)
    assert all(
        featurized.metadata["participant_label"].nunique() == 1
        or featurized.metadata.shape[0] == 0
        for featurized in one_featurized_list
    )
    # In the case of participants with multiple specimens: confirm they have been merged at this point
    participant_label_of_each = [
        featurized.metadata.iloc[0]["participant_label"]
        for featurized in one_featurized_list
        if featurized.y.shape[0] > 0
    ]
    assert len(participant_label_of_each) == len(
        set(participant_label_of_each)
    ), "Same participant label shared by multiple entries in featurized_list"

    # Get y label of each featurized object
    disease_label_of_each = [
        featurized.y[0] if featurized.y.shape[0] > 0 else None
        for featurized in one_featurized_list
    ]

    # Choose how many we will split off for validation. Take 30% of smallest class (measured by # unique patients) - constrained to be between 2 to 10 samples per class.
    amount_to_take = int(
        min(
            max(
                0.3 * pd.Series(disease_label_of_each).dropna().value_counts().min(), 2
            ),
            10,
        )
    )
    logger.info(
        f"{gene_locus}, {classification_target}, metamodel flavor {metamodel_flavor}: splitting off {amount_to_take} participants per class for decision threshold tuning."
    )

    # Get indices for each disease. We want to choose the same number per disease.
    indices_per_disease = {
        disease: [
            ix for ix, label in enumerate(disease_label_of_each) if label == disease
        ]
        for disease in pd.Series(disease_label_of_each).dropna().unique()
    }

    # Split off some validation from remaining test
    indices_per_disease = {
        disease: (indices[:amount_to_take], indices[amount_to_take:])
        for disease, indices in indices_per_disease.items()
    }
    indices_validation_all = np.hstack(
        [
            validation_indices
            for disease, (
                validation_indices,
                test_indices,
            ) in indices_per_disease.items()
        ]
    )
    indices_test_all = np.hstack(
        [
            test_indices
            for disease, (
                validation_indices,
                test_indices,
            ) in indices_per_disease.items()
        ]
    )
    # Add abstention indices to indices_test_all
    indices_test_all = np.hstack(
        [indices_test_all, np.where(pd.Series(disease_label_of_each).isna())[0]]
    )

    # sanity check
    assert len(set(indices_validation_all).intersection(set(indices_test_all))) == 0
    assert (
        len(indices_test_all) + len(indices_validation_all)
        == len(disease_label_of_each)
        == len(one_featurized_list)
    )

    ## Run each metamodel
    for metamodel_name in metamodel_names:
        featurized_list = featurized_lists_per_metamodel[metamodel_name]
        clf = clfs[metamodel_name]

        # Combine featurized_list in a few different ways:
        # - combine all
        featurized_all: FeaturizedData = FeaturizedData.concat(featurized_list)
        # - combine validation subset
        featurized_validation: FeaturizedData = FeaturizedData.concat(
            [featurized_list[ix] for ix in indices_validation_all]
        )
        # - combine test subset
        featurized_test: FeaturizedData = FeaturizedData.concat(
            [featurized_list[ix] for ix in indices_test_all]
        )

        # Handle abstention
        if (
            featurized_all.X.shape[0] == 0
            or featurized_validation.X.shape[0] == 0
            or featurized_test.X.shape[0] == 0
        ):
            raise ValueError("All abstained")

        # Tune model
        clf_tuned = AdjustedProbabilitiesDerivedModel.adjust_model_decision_thresholds(
            model=clf,
            X_validation=featurized_validation.X,
            y_validation_true=featurized_validation.y,
        )

        # Do evaluation
        for transformed_clf, transformed_model_name, featurized in zip(
            [clf, clf, clf_tuned],
            [
                metamodel_name + ".untuned.all_data",
                metamodel_name + ".untuned.test_subset",
                metamodel_name + ".tuned.test_subset",
            ],
            [featurized_all, featurized_test, featurized_test],
        ):
            results.add(
                model_evaluation.ModelSingleFoldPerformance(
                    model_name=transformed_model_name,
                    fold_id=-1,
                    y_true=featurized.y,
                    clf=transformed_clf,
                    X_test=featurized.X,
                    fold_label_train="train_smaller",
                    fold_label_test="external",
                    test_metadata=featurized.metadata,
                    test_abstentions=featurized.abstained_sample_y,
                    test_abstention_metadata=featurized.abstained_sample_metadata,
                )
            )
            featurized_by_metamodel_name[transformed_model_name] = featurized

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

    return results, featurized_by_metamodel_name


# %%

# %% [markdown]
# # External cohorts similar to our sequencing process

# %%
featurized_by_gene_locus_and_model_name = kdict()
results_by_gene_locus = {}

for gene_locus in gene_loci:
    # run on single locus
    results, featurized_by_metamodel_name = run(
        gene_locus=gene_locus,
        different_platform=False,
        classification_target=evaluation_target,
        output_dir=config.paths.external_cohort_evaluation_output_dir
        / "default"
        / gene_locus.name,
    )
    if results is not None:
        results_by_gene_locus[gene_locus] = results
        for metamodel_name, featurized in featurized_by_metamodel_name.items():
            featurized_by_gene_locus_and_model_name[
                gene_locus, metamodel_name
            ] = featurized

if len(gene_loci) > 1:
    # run on multiple loci
    results, featurized_by_metamodel_name = run(
        gene_locus=gene_loci,
        different_platform=False,
        classification_target=evaluation_target,
        output_dir=config.paths.external_cohort_evaluation_output_dir
        / "default"
        / gene_loci.name,
    )
    if results is not None:
        results_by_gene_locus[gene_loci] = results
        for metamodel_name, featurized in featurized_by_metamodel_name.items():
            featurized_by_gene_locus_and_model_name[
                gene_loci, metamodel_name
            ] = featurized


# Visualize subcomponent predictions
# (Have to use this awkward kdict notation because GeneLocus flag-enum key is not well handled)
any_metamodel_name = featurized_by_gene_locus_and_model_name.keys(dimensions=1)[0]
for key, featurized in featurized_by_gene_locus_and_model_name[
    :, any_metamodel_name
].items():
    gene_locus, _ = key
    output_dir = (
        config.paths.high_res_outputs_dir
        / "external_cohort_evaluation"
        / "default"
        / gene_locus.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    display(Markdown(f"### {key}"))
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

# %% [markdown]
# ## External cohorts similar to our sequencing process: Demographics-only metamodel for comparison

# %%
for gene_locus in gene_loci:
    # run on single locus
    run(
        gene_locus=gene_locus,
        different_platform=False,
        classification_target=evaluation_target_with_demographics,
        output_dir=config.paths.external_cohort_evaluation_output_dir
        / "demographics_only"
        / gene_locus.name,
        metamodel_flavor="demographics_only",
    )

if len(gene_loci) > 1:
    # run on multiple loci
    run(
        gene_locus=gene_loci,
        different_platform=False,
        classification_target=evaluation_target_with_demographics,
        output_dir=config.paths.external_cohort_evaluation_output_dir
        / "demographics_only"
        / gene_loci.name,
        metamodel_flavor="demographics_only",
    )

# %%

# %% [markdown]
# # External cohorts different from our sequencing process

# %%
featurized_by_gene_locus_and_model_name = kdict()
results_by_gene_locus = {}

for gene_locus in gene_loci:
    # run on single locus
    results, featurized_by_metamodel_name = run(
        gene_locus=gene_locus,
        different_platform=True,
        classification_target=evaluation_target,
        output_dir=config.paths.high_res_outputs_dir
        / "external_cohort_evaluation"
        / "different_platform"
        / gene_locus.name,
    )
    if results is not None:
        results_by_gene_locus[gene_locus] = results
        for metamodel_name, featurized in featurized_by_metamodel_name.items():
            featurized_by_gene_locus_and_model_name[
                gene_locus, metamodel_name
            ] = featurized

if len(gene_loci) > 1:
    # run on multiple loci
    results, featurized_by_metamodel_name = run(
        gene_locus=gene_loci,
        different_platform=True,
        classification_target=evaluation_target,
        output_dir=config.paths.high_res_outputs_dir
        / "external_cohort_evaluation"
        / "different_platform"
        / gene_loci.name,
    )
    if results is not None:
        results_by_gene_locus[gene_loci] = results
        for metamodel_name, featurized in featurized_by_metamodel_name.items():
            featurized_by_gene_locus_and_model_name[
                gene_loci, metamodel_name
            ] = featurized


# Visualize subcomponent predictions
# (Have to use this awkward kdict notation because GeneLocus flag-enum key is not well handled)
any_metamodel_name = featurized_by_gene_locus_and_model_name.keys(dimensions=1)[0]
for key, featurized in featurized_by_gene_locus_and_model_name[
    :, any_metamodel_name
].items():
    gene_locus, _ = key
    output_dir = (
        config.paths.high_res_outputs_dir
        / "external_cohort_evaluation"
        / "different_platform"
        / gene_locus.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    display(Markdown(f"### {key}"))
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

# %%

# %%

# %%

# %%
