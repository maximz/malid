# %%
from summarynb import show
from IPython.display import display, Markdown
from slugify import slugify
from malid import config, helpers
from malid.datamodels import GeneLocus

# %%

# %% [markdown]
# # Get disease prediction rankings/scores for all sequences (uniques combined from all our test folds)

# %% [markdown]
# ### We filtered out rare V genes
#
# To be kept, a V gene must exceed the purple dashed line in at least one disease (see plot below). This threshold was chosen because it's the median of the max proportion a V gene takes up of any disease, for all V genes. Therefore half of the V genes are discarded at this step.

# %%
for gene_locus in config.gene_loci_used:
    main_output_base_dir = (
        config.paths.model_interpretations_output_dir / gene_locus.name
    )
    highres_output_base_dir = (
        config.paths.high_res_outputs_dir / "model_interpretations" / gene_locus.name
    )

    display(Markdown(f"## {gene_locus}"))
    show(
        [
            highres_output_base_dir / f"v_gene_disease_proportions.png",
            # highres_output_base_dir / f"v_gene_disease_proportions.filtered.png",
        ],
        headers=[
            "original V gene proportions in each disease group",
            # "remaining V genes after filter: how prevalent they are by disease",
        ],
    )

    for target_obs_column in config.classification_targets:
        display(Markdown(f"### {gene_locus}, {target_obs_column}"))
        # Subdirectories for each classification target
        main_output_dir = main_output_base_dir / target_obs_column.name
        highres_output_dir = highres_output_base_dir / target_obs_column.name

        if gene_locus == GeneLocus.BCR:
            show(
                [
                    main_output_dir / "all.without_healthy.png",
                    highres_output_dir / "isotype_usage.png",
                ],
                headers=["All (without healthy)", "Isotype usage"],
                max_width=1000,
                max_height=600,
            )
        else:
            show(
                main_output_dir / "all.without_healthy.png",
                max_width=1000,
                max_height=600,
                headers=["All (without healthy)"],
            )

        #         show(
        #             highres_output_dir / "all.png",
        #             max_width=1500,
        #             headers=["All"],
        #         )

        #         # plot V gene enrichment
        #         show(
        #             highres_output_dir / "v_gene_rankings.filtered.png",
        #             max_width=1500,
        #             headers=["V gene rankings"],
        #         )

        show(
            highres_output_dir / "v_gene_rankings.png",
            max_width=1500,
            headers=["V gene rankings (unfiltered)"],
        )

#         show(
#             highres_output_dir / "cdr3_lengths.png",
#             headers=["CDR3 length"],
#             max_width=2000,
#         )


# %%

# %%
