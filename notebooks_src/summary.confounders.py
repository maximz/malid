# %%
from summarynb import show, indexed_csv, plaintext, chunks, table

from malid import config, logger, helpers
from malid.datamodels import healthy_label
from slugify import slugify
import pandas as pd

# %%
from IPython.display import display, Markdown

# %%

# %%

# %% [markdown]
# # Isotype counts plotted
#
# Average isotype proportions in specimens of each disease type.
#
# Also plot 95% confidence intervals:
# - Standard error of the mean is basically the standard deviation of many sample means drawn by bootstrap
# - Create a sampling distribution of the mean by bootstrap repeated sampling and recording the mean each time
# - Plot mean +/- 1.96 * standard error. Gives you average value +/- X at the 95% confidence level.

# %%
show(
    config.paths.base_output_dir_for_selected_cross_validation_strategy
    / "isotype_counts_by_class.png"
)
show(
    config.paths.base_output_dir_for_selected_cross_validation_strategy
    / "isotype_counts_by_class.inverted.png"
)

# %%

# %% [markdown]
# # V gene usage counts in specimens of each disease type, or specimens of different ancestries or batches
#
# Plot average V gene use proportions in specimens of each disease type.
#
# Also plot 95% confidence intervals:
# - Standard error of the mean is basically the standard deviation of many sample means drawn by bootstrap
# - Create a sampling distribution of the mean by bootstrap repeated sampling and recording the mean each time
# - Plot mean +/- 1.96 * standard error. Gives you average value +/- X at the 95% confidence level.

# %%
for gene_locus in config.gene_loci_used:
    main_output_dir = (
        config.paths.model_interpretations_for_selected_cross_validation_strategy_output_dir
        / gene_locus.name
    )
    high_res_output_dir = (
        config.paths.high_res_outputs_dir_for_cross_validation_strategy
        / "model_interpretations"
        / gene_locus.name
    )
    show(
        high_res_output_dir
        / "v_gene_proportions_by_specimen.filtered_v_genes.by_disease.png",
        headers=[f"Overall V gene usage proportions by specimen - {gene_locus}"],
        max_width=1200,
    )

    for disease in [healthy_label, "Covid19"]:
        show(
            main_output_dir
            / f"v_gene_proportions_by_specimen.filtered_v_genes.disease.{slugify(disease)}.by_ethnicity.png",
            headers=[
                f"{disease} V gene usage proportions by specimen by ethnicity - {gene_locus}"
            ],
            max_width=1200,
        )

        show(
            main_output_dir
            / f"v_gene_proportions_by_specimen.filtered_v_genes.disease.{slugify(disease)}.by_study_name.png",
            headers=[
                f"{disease} V gene usage proportions by specimen by study name - {gene_locus}"
            ],
            max_width=1200,
        )

    show(
        [
            high_res_output_dir
            / "v_gene_proportions_by_specimen.pca.color_by_disease_batch.png",
            high_res_output_dir
            / "v_gene_proportions_by_specimen.umap.color_by_disease_batch.png",
        ],
        headers=[
            f"V gene proportions PCA by disease+batch - {gene_locus}",
            f"V gene proportions UMAP by disease+batch - {gene_locus}",
        ],
    )

    show(
        [
            high_res_output_dir
            / "v_gene_proportions_by_specimen.means_by_disease_batch.distance_heatmap.png",
            high_res_output_dir
            / "v_gene_proportions_by_specimen.medians_by_disease_batch.distance_heatmap.png",
        ],
        headers=[
            f"Distance between disease+batch mean V gene usage proportion vectors - {gene_locus}",
            f"Distance between disease+batch median V gene usage proportion vectors - {gene_locus}",
        ],
    )
    show(
        high_res_output_dir
        / "v_gene_proportions_by_specimen.pairwise_distances.boxplot_by_disease_batch.png",
        headers=[
            f"Pairwise distances between specimen V gene use proportions - {gene_locus}"
        ],
    )

# %%

# %%
