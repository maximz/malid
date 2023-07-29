# %%

# %%

# %%
from malid import config, interpretation

# %%

# %%

# %% [markdown]
# # Get disease prediction scores for all sequences from those-disease patients (uniques combined from all our test folds)

# %%
for gene_locus in config.gene_loci_used:
    print(gene_locus)
    main_output_base_dir = (
        config.paths.model_interpretations_output_dir / gene_locus.name
    )
    main_output_base_dir.mkdir(parents=True, exist_ok=True)

    highres_output_base_dir = (
        config.paths.high_res_outputs_dir / "model_interpretations" / gene_locus.name
    )
    highres_output_base_dir.mkdir(parents=True, exist_ok=True)

    ranked_sequence_dfs = interpretation.rank_entire_locus_sequences(
        gene_locus=gene_locus,
        target_obs_columns=config.classification_targets,
        main_output_base_dir=main_output_base_dir,
        highres_output_base_dir=highres_output_base_dir,
    )

# %%

# %%

# %%
