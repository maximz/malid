from typing import Dict, Tuple
import logging
import pandas as pd
from malid import config
from malid.trained_model_wrappers import ConvergentClusterClassifier
from malid.datamodels import GeneLocus

logger = logging.getLogger(__name__)


###
### Load known binder databases


def _load_covabdab(
    clustering_train_threshold: float,
    known_binder: bool,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, int], pd.DataFrame], str]:
    """Cluster CoV-AbDab CDR3s"""
    train_sequences_df = pd.read_csv(
        config.paths.base_data_dir / "CoV-AbDab_130623.filtered.annotated.tsv", sep="\t"
    )
    train_sequences_df = train_sequences_df[
        train_sequences_df["Status"] == ("Positive" if known_binder else "Negative")
    ].reset_index(drop=True)

    old_shape = train_sequences_df.shape[0]
    train_sequences_df = ConvergentClusterClassifier._cluster_training_set(
        df=train_sequences_df,
        sequence_identity_threshold=clustering_train_threshold,
        # skip fold_id and fold_label check
        validate_same_fold=False,
    )
    assert train_sequences_df.shape[0] == old_shape

    # # total number of clusters across all Cov-abdab data
    # train_sequences_df["global_resulting_cluster_ID"].nunique()

    # # a number of cov-abdab sequences were joined into a single cluster
    # train_sequences_df["global_resulting_cluster_ID"].value_counts()

    # # how many cov-abdab sequences were merged
    # (train_sequences_df["global_resulting_cluster_ID"].value_counts() > 1).value_counts()

    # train_sequences_df["global_resulting_cluster_ID"].value_counts()
    # train_sequences_df["global_resulting_cluster_ID"].value_counts().hist(bins=20)

    ## Consider all of these to be "predictive clusters", since they are from Covabdab. I.e. no further filtering.
    ## Make cluster centroids for clusters
    # Since we don't have `num_clone_members`, it was set to 1, so these will not be weighed by number of clone members (number of unique VDJ sequences)
    # train_sequences_df["num_clone_members"].value_counts()

    # Make cluster centroids for predictive clusters, weighed by number of clone members (number of unique VDJ sequences)
    # Except here we have weights=1
    # And all clusters are predictive
    cluster_centroids_df = ConvergentClusterClassifier._get_cluster_centroids(
        clustered_df=train_sequences_df
    )
    # Reshape as dict
    cluster_centroids_by_supergroup = (
        ConvergentClusterClassifier._wrap_cluster_centroids_as_dict_by_supergroup(
            cluster_centroids=cluster_centroids_df
        )
    )
    return train_sequences_df, cluster_centroids_by_supergroup, "CoV-AbDab"


def _load_mira(
    clustering_train_threshold: float,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, int], pd.DataFrame], str]:
    """Cluster MIRA CDR3s"""
    train_sequences_df = pd.read_csv(
        config.paths.external_raw_data
        / "immunecode_all/mira/ImmuneCODE-MIRA-Release002.1"
        / "mira_combined.filtered.tsv",
        sep="\t",
    )

    old_shape = train_sequences_df.shape[0]
    train_sequences_df = ConvergentClusterClassifier._cluster_training_set(
        df=train_sequences_df,
        sequence_identity_threshold=clustering_train_threshold,
        # skip fold_id and fold_label check
        validate_same_fold=False,
    )
    assert train_sequences_df.shape[0] == old_shape

    # # total number of clusters across all reference data
    # train_sequences_df["global_resulting_cluster_ID"].nunique()

    # # a number of cov-abdab sequences were joined into a single cluster
    # train_sequences_df["global_resulting_cluster_ID"].value_counts()

    # # how many cov-abdab sequences were merged
    # (train_sequences_df["global_resulting_cluster_ID"].value_counts() > 1).value_counts()

    # train_sequences_df["global_resulting_cluster_ID"].value_counts()
    # train_sequences_df["global_resulting_cluster_ID"].value_counts().hist(bins=20)

    ## Consider all of these to be "predictive clusters". No further filtering.

    ## Make cluster centroids for clusters
    # Since we don't have `num_clone_members`, it was set to 1, so this will not be weighed by number of clone members (number of unique VDJ sequences)
    # train_sequences_df["num_clone_members"].value_counts()

    # Make cluster centroids for predictive clusters, weighed by number of clone members (number of unique VDJ sequences)
    # Except here we have weights=1
    # And all clusters are predictive
    cluster_centroids_df = ConvergentClusterClassifier._get_cluster_centroids(
        clustered_df=train_sequences_df
    )
    # Reshape as dict
    cluster_centroids_by_supergroup = (
        ConvergentClusterClassifier._wrap_cluster_centroids_as_dict_by_supergroup(
            cluster_centroids=cluster_centroids_df
        )
    )
    return train_sequences_df, cluster_centroids_by_supergroup, "MIRA"


def _load_flu_known_binders(
    clustering_train_threshold: float,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, int], pd.DataFrame], str]:
    """Cluster flu known binders"""
    train_sequences_df = pd.read_csv(
        config.paths.base_data_dir / "flu_known_binders.filtered.annotated.tsv",
        sep="\t",
    )

    # Merge nearly identical sequences
    old_shape = train_sequences_df.shape[0]
    train_sequences_df = ConvergentClusterClassifier._cluster_training_set(
        df=train_sequences_df,
        sequence_identity_threshold=clustering_train_threshold,
        # skip fold_id and fold_label check
        validate_same_fold=False,
    )
    assert train_sequences_df.shape[0] == old_shape

    # Consider all of these to be "predictive clusters", i.e. no further filtering.
    # Make cluster centroids for predictive clusters, weighed by number of clone members (number of unique VDJ sequences):
    # Except here we have weights=1 (since we don't have `num_clone_members`, it was set to 1, so these will not be weighed by number of clone members (number of unique VDJ sequences))
    # And all clusters are predictive
    cluster_centroids_df = ConvergentClusterClassifier._get_cluster_centroids(
        clustered_df=train_sequences_df
    )
    # Reshape as dict
    cluster_centroids_by_supergroup = (
        ConvergentClusterClassifier._wrap_cluster_centroids_as_dict_by_supergroup(
            cluster_centroids=cluster_centroids_df
        )
    )
    return train_sequences_df, cluster_centroids_by_supergroup, "Flu known binders"


def load_reference_dataset(
    gene_locus: GeneLocus,
    disease: str,
    known_binder: bool,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, int], pd.DataFrame], str]:
    """
    Load dataset of known binders, clustered by CDR3
    If known_binder is True, load known binders. If False, load known non-binders.
    Returns: dataframe, cluster centroids by supergroup, reference database name
    """
    GeneLocus.validate(gene_locus)
    GeneLocus.validate_single_value(gene_locus)

    # Sequence identity thresholds for clustering amino acid sequences (across patients):
    # threshold for combining reference-dataset source seqs into clusters
    sequence_identity_thresholds = {
        # for BCR this is very high because we just want to merge near-exact dupes
        GeneLocus.BCR: 0.95,
        # for TCR this is 1.0 because we want exact matches
        GeneLocus.TCR: 1.0,
    }

    if disease == "Covid19":
        if gene_locus == GeneLocus.BCR:
            # sequence identity thresholds for clustering amino acid sequences (across patients):
            # threshold for combining cov-abdab source seqs into clusters
            return _load_covabdab(
                clustering_train_threshold=sequence_identity_thresholds[gene_locus],
                known_binder=known_binder,
            )
        elif gene_locus == GeneLocus.TCR and known_binder:
            return _load_mira(
                clustering_train_threshold=sequence_identity_thresholds[gene_locus]
            )

    elif disease == "Influenza" and gene_locus == GeneLocus.BCR and known_binder:
        return _load_flu_known_binders(
            clustering_train_threshold=sequence_identity_thresholds[gene_locus]
        )

    raise ValueError(
        f"No known binder data for disease={disease}, {gene_locus}, binding status={known_binder}"
    )
