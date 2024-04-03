# %%

# %%
from pathlib import Path
from malid import config, logger
from malid.train import train_metamodel
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    map_cross_validation_split_strategy_to_default_target_obs_column,
)
import pandas as pd
from IPython.display import display, Markdown
from typing import List

# %%

# %%
base_model_train_fold_name = "train_smaller"
metamodel_fold_label_train = "validation"

# %%
models_of_interest = config.model_names_to_train
models_of_interest

# %%
# We only support split strategies with default target obs column == TargetObsColumnEnum.disease
assert (
    map_cross_validation_split_strategy_to_default_target_obs_column[
        config.cross_validation_split_strategy
    ]
    == TargetObsColumnEnum.disease
)

# %%
def choose(gene_locus: GeneLocus, classification_targets: List[TargetObsColumnEnum]):
    for target_obs_column in classification_targets:
        try:
            flavors = train_metamodel.get_metamodel_flavors(
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                fold_id=config.all_fold_ids[0],
                base_model_train_fold_name=base_model_train_fold_name,
                use_stubs_instead_of_submodels=True,
            )
        except Exception as err:
            logger.warning(
                f"Failed to generate metamodel flavors for {gene_locus}, {target_obs_column}: {err}"
            )
            continue

        for metamodel_flavor, metamodel_config in flavors.items():
            _output_suffix = (
                Path(gene_locus.name)
                / target_obs_column.name
                / metamodel_flavor
                / f"{base_model_train_fold_name}_applied_to_{metamodel_fold_label_train}_model"
            )
            results_output_prefix = (
                config.paths.second_stage_blending_metamodel_output_dir / _output_suffix
            )

            display(
                Markdown(
                    f"# {gene_locus}, {target_obs_column}, metamodel flavor {metamodel_flavor}"
                )
            )

            try:
                ## All results in a table
                df = pd.read_csv(
                    f"{results_output_prefix}.compare_model_scores.test_set_performance.tsv",
                    sep="\t",
                    index_col=0,
                )
                df["missing_classes"] = df["missing_classes"].map(
                    {False: "✅ no", True: "❗ Missing classes!"}
                )
                df = df.loc[df.index.intersection(models_of_interest)].sort_values(
                    "ROC-AUC (weighted OvO) per fold", ascending=False
                )
                if "Accuracy global with abstention" in df.columns:
                    df = df[
                        [
                            "ROC-AUC (weighted OvO) per fold",
                            "au-PRC (weighted OvO) per fold",
                            "Accuracy global with abstention",
                            "MCC global with abstention",
                            "abstention_rate",
                            "sample_size including abstentions",
                            "n_abstentions",
                            "missing_classes",
                        ]
                    ]
                else:
                    df = df[
                        [
                            "ROC-AUC (weighted OvO) per fold",
                            "au-PRC (weighted OvO) per fold",
                            "Accuracy global",
                            "MCC global",
                            "abstention_rate",
                            "sample_size including abstentions",
                            "n_abstentions",
                            "missing_classes",
                        ]
                    ]
                display(df)
            except Exception as err:
                logger.warning(
                    f"{gene_locus}, {target_obs_column} flavor '{metamodel_flavor}': not yet run: {err}"
                )
                continue


# %%
for single_gene_locus in config.gene_loci_used:
    choose(single_gene_locus, config.classification_targets)

# %%

# %%
if len(config.gene_loci_used) > 1:
    choose(config.gene_loci_used, config.classification_targets)

# %%

# %%

# %%
# Default
for target in [
    TargetObsColumnEnum.disease,
]:
    for gene_locus in config.gene_loci_used:
        choose(gene_locus, [target])
    choose(config.gene_loci_used, [target])

# %%
# Demographic controlled
for target in [
    TargetObsColumnEnum.disease_all_demographics_present,
]:
    for gene_locus in config.gene_loci_used:
        choose(gene_locus, [target])
    choose(config.gene_loci_used, [target])

# %%
# Demographics from healthy
for target in [
    TargetObsColumnEnum.ethnicity_condensed_healthy_only,
    TargetObsColumnEnum.age_group_healthy_only,
    TargetObsColumnEnum.age_group_binary_healthy_only,
    TargetObsColumnEnum.age_group_pediatric_healthy_only,
    TargetObsColumnEnum.sex_healthy_only,
]:
    for gene_locus in config.gene_loci_used:
        choose(gene_locus, [target])
    choose(config.gene_loci_used, [target])

# %%
