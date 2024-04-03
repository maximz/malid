# %% [markdown]
# # Embed known binder sequences with each fold's fine-tuned language model, and apply existing scaling and PCA transformations
#
# See `scripts/off_peak.run_embedding_fine_tuned.and_scale.py`.
#
# Recall that we have a separate fine-tuned language model for each train-smaller set. So treat this as an extension of the test set. For each test fold ID, and apply the language model, scaling, and PCA transformations trained on that fold's train-smaller set.

# %%
from slugify import slugify
import joblib
import choosegpu
from malid import config, apply_embedding, interpretation
from malid.datamodels import GeneLocus, healthy_label

# %%
# Embed with GPU
choosegpu.configure_gpu(enable=True)

# %%
config.embedder.name


# %%

# %%
def process(gene_locus: GeneLocus, disease: str, known_binder: bool):
    status = "binders" if known_binder else "nonbinders"
    print(gene_locus, disease, status)

    GeneLocus.validate_single_value(gene_locus)
    (
        df,
        cluster_centroids_by_supergroup,
        reference_dataset_name,
    ) = interpretation.load_reference_dataset(
        gene_locus=gene_locus, disease=disease, known_binder=known_binder
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

    # We don't have v_mut or isotype for some known binder datasets
    # Set defaults
    if "isotype_supergroup" not in df.columns:
        df["isotype_supergroup"] = "IGHG"
    if "v_mut" not in df.columns:
        df["v_mut"] = 0.0

    df["participant_label"] = reference_dataset_name
    df["specimen_label"] = reference_dataset_name
    if known_binder:
        df["disease"] = disease
        df["disease_subtype"] = f"{disease} - known {status}"
    else:
        df["disease"] = healthy_label
        df["disease_subtype"] = f"{healthy_label} - known {status} to {disease}"

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
        / f"known_{status}.{slugify(disease)}.embedded.in.all.folds.joblib",
    )


# %%

# %%
if GeneLocus.BCR in config.gene_loci_used:
    process(gene_locus=GeneLocus.BCR, disease="Covid19", known_binder=True)
    process(gene_locus=GeneLocus.BCR, disease="Covid19", known_binder=False)

    process(gene_locus=GeneLocus.BCR, disease="Influenza", known_binder=True)

if GeneLocus.TCR in config.gene_loci_used:
    process(gene_locus=GeneLocus.TCR, disease="Covid19", known_binder=True)

# %%

# %%

# %%

# %%

# %%
