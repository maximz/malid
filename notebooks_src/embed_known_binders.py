# %% [markdown]
# # Embed known binder sequences with each fold's fine-tuned language model, and apply existing scaling and PCA transformations
#
# See `scripts/off_peak.run_embedding_fine_tuned.and_scale.py`.
#
# Recall that we have a separate fine-tuned language model for each train-smaller set. So treat this as an extension of the test set. For each test fold ID, and apply the language model, scaling, and PCA transformations trained on that fold's train-smaller set.

# %%
import numpy as np
import pandas as pd
import joblib
import choosegpu
from malid import config, apply_embedding, interpretation
from malid.datamodels import GeneLocus

# %%
# Embed with GPU
choosegpu.configure_gpu(enable=True)

# %%
config.embedder.name


# %%

# %%
def process(gene_locus):
    print(gene_locus)
    GeneLocus.validate_single_value(gene_locus)
    df, cluster_centroids_by_supergroup = interpretation.load_reference_dataset(
        gene_locus
    )
    print(df.shape)

    # total number of clusters across all data
    df["global_resulting_cluster_ID"].nunique()
    # a number of sequences were joined into a single cluster
    df["global_resulting_cluster_ID"].value_counts()
    # how many sequences were merged
    (df["global_resulting_cluster_ID"].value_counts() > 1).value_counts()

    # choose one entry per cluster
    df = df.groupby("global_resulting_cluster_ID").head(n=1).copy()
    print(df.shape)

    # Note: we don't have v_mut or isotype for CoV-AbDab
    if "isotype_supergroup" not in df.columns:
        df["isotype_supergroup"] = "IGHG"
    if "v_mut" not in df.columns:
        df["v_mut"] = 0.0

    df["participant_label"] = interpretation.reference_dataset_name[gene_locus]
    df["specimen_label"] = interpretation.reference_dataset_name[gene_locus]
    df["disease"] = "Covid19"
    df["disease_subtype"] = "Covid19 - known binder"

    embedded = {}
    for fold_id in config.all_fold_ids:
        fold_df = df.copy()
        fold_df["participant_label"] += f"_{fold_id}"
        fold_df["specimen_label"] += f"_{fold_id}"

        # Make adata
        adata = apply_embedding.run_embedding_model(
            embedder=apply_embedding.load_embedding_model(
                gene_locus=gene_locus, fold_id=fold_id
            ),
            df=fold_df,
            gene_locus=gene_locus,
            fold_id=fold_id,
        )
        adata = apply_embedding.transform_embedded_anndata(
            transformations_to_apply=apply_embedding.load_transformations(
                gene_locus=gene_locus, fold_id=fold_id
            ),
            adata=adata,
        )

        embedded[fold_id] = adata
        print(fold_id, adata)

    joblib.dump(
        embedded,
        config.paths.scaled_anndatas_dir
        / gene_locus.name
        / "known_binders.embedded.in.all.folds.joblib",
    )


# %%

# %%
for gene_locus in config.gene_loci_used:
    process(gene_locus)

# %%

# %%

# %%

# %%

# %%
