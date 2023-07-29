# %%
from summarynb import show
from malid import config, helpers
from IPython.display import display, Markdown
from slugify import slugify

# %%
representations = ["pca", "umap"]

# %%

# %% [markdown]
# # Plot PCA and UMAP of language model embedding of test sets ("unsupervised embedding")
#
# - Data was scaled using training set's scaler
# - PCA was computed already, using training set's PCA
# - UMAP computed from scratch (first on training set, then applied to test set)
#
# This is different from supervised embedding which comes out of the trained classifiers.
#
# The relative density plots are adjusted for class size differences.

# %%
for fold_id in config.cross_validation_fold_ids:
    for gene_locus in config.gene_loci_used:
        display(Markdown(f"## Fold {fold_id}-test, {gene_locus}"))
        show(
            [
                config.paths.output_dir
                / f"language_model_embedding.all_diseases.{representation}.fold_{fold_id}_test.{gene_locus.name}.png"
                for representation in representations
            ],
            headers=representations,
        )
        for disease in helpers.diseases_in_peak_timepoint_dataset():
            display(Markdown(f"### {disease} batches"))
            show(
                [
                    config.paths.high_res_outputs_dir
                    / f"language_model_embedding.by_batch.{slugify(disease)}.{representation}.fold_{fold_id}_test.{gene_locus.name}.png"
                    for representation in representations
                ],
                headers=[
                    f"{representation} - scatterplot"
                    for representation in representations
                ],
            )

            show(
                [
                    config.paths.high_res_outputs_dir
                    / f"language_model_embedding.by_batch.{slugify(disease)}.{representation}.overall_density.fold_{fold_id}_test.{gene_locus.name}.png"
                    for representation in representations
                ],
                headers=[
                    f"{representation} - overall density"
                    for representation in representations
                ],
            )

            show(
                [
                    config.paths.high_res_outputs_dir
                    / f"language_model_embedding.by_batch.{slugify(disease)}.{representation}.relative_density_with_min_density_requirement.fold_{fold_id}_test.{gene_locus.name}.png"
                    for representation in representations
                ],
                headers=[
                    f"{representation} - relative density (filtered out low overall density bins)"
                    for representation in representations
                ],
            )

            show(
                [
                    config.paths.high_res_outputs_dir
                    / f"language_model_embedding.by_batch.{slugify(disease)}.{representation}.relative_density.fold_{fold_id}_test.{gene_locus.name}.png"
                    for representation in representations
                ],
                headers=[
                    f"{representation} - relative density (all bins, for comparison)"
                    for representation in representations
                ],
            )

        display(Markdown(f"---"))

# %%

# %%
