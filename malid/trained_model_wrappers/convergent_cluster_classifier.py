import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from malid import config, helpers
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    combine_classification_option_names,
)
from extendanything import ExtendAnything
from malid.external.genetools_arrays import (
    strings_to_numeric_vectors,
    make_consensus_sequence,
    masked_argmin,
)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from malid.external.model_evaluation import FeaturizedData
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)
from fisher import pvalue_npy

logger = logging.getLogger(__name__)


class ConvergentClusterClassifier(ImmuneClassifierMixin, ExtendAnything):
    """Wrapper around pre-trained convergent cluster classification models.

    Load different model versions by configuring fold_label_train, e.g. "train" vs "train_smaller".

    This wrapper leaves decision_function, predict_proba, and predict of the underlying classifier untouched,
    but provides helper functions to featurize an anndata into the right format for the underlying classifier.

    So run featurize() on your anndata first, then pass the resulting feature matrix to predict(), predict_proba(), or decision_function().

    Example usage:

    ```python
        from malid.trained_model_wrappers import ConvergentClusterClassifier

        clf = ConvergentClusterClassifier(
            fold_id=0,
            model_name="lasso_multiclass",
            fold_label_train="train_smaller",
        )
        X_test, y_test, test_metadata = clf.featurize(adata_test.obs)
        predicted_probas = pd.DataFrame(
            clf.predict_proba(X_test),
            index=test_metadata.index,
            columns=clf.classes_
        )
    ```

    """

    # ExtendAnything means will pass through to base instance's attributes, exposing the base model's classes_, predict(), predict_proba(), etc.

    @classmethod
    def _cluster_training_set(
        cls,
        df: pd.DataFrame,
        sequence_identity_threshold: float,
        validate_same_fold=True,
        inplace=False,
        higher_order_group_columns=None,
        sequence_column="cdr3_seq_aa_q_trim",
    ) -> pd.DataFrame:
        if not inplace:
            df = df.copy()  # don't modify original df

        if validate_same_fold:
            cls._validate_same_fold_id_and_label(df)

        if higher_order_group_columns is None:
            higher_order_group_columns = [
                "v_gene",
                "j_gene",
                "cdr3_aa_sequence_trim_len",
            ]

        # Cluster

        ## group by v, j, len
        groups = df.groupby(higher_order_group_columns, observed=True)

        # assign group IDs, i.e. putative cluster IDs across patients, that will later be subset into finer cluster IDs
        # (we will run single linkage clustering on the sequences within each putative_cross_patient_group_ID)
        # df["putative_cross_patient_group_ID"] = groups.ngroup()

        # actually instead we will just use ("v_gene", "j_gene", "cdr3_aa_sequence_trim_len") tuple

        ## cluster each group

        # Single linkage clustering docs:
        #
        # ```
        # A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster n+i. A cluster with an index less than n corresponds to one of the n original observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.
        # ```

        def cluster_sequences(series):
            # input: pandas.Series of all the sequences within a group
            # process each group:

            if series.nunique() == 1:
                # Only one unique sequence

                # This means all distances will be 0
                # Meaning single linkage clustering will error out:
                # ValueError: The number of observations cannot be determined on an empty distance matrix.

                # In this case, return the same cluster ID for every input sequence, since they are identical
                return np.zeros(series.shape[0])

            # convert array of strings into array of int-arrays
            sequences_as_vectors = strings_to_numeric_vectors(series.values)

            # distance matrix: normalized hamming distance
            dist_mat = pdist(sequences_as_vectors, metric="hamming")

            # single linkage clustering, returns (n-1)x4 matrix
            clustering = linkage(dist_mat, method="single")

            # enforce sequence identity threshold for clusters
            flat_clusters = fcluster(
                clustering,
                1 - sequence_identity_threshold,
                criterion="distance",
            )

            # cluster ID per observation
            # these IDs are only unique within the group (the putative_cross_patient_group_ID)
            return flat_clusters

        df["cluster_id_within_clustering_group"] = groups[sequence_column].transform(
            cluster_sequences
        )
        assert not df["cluster_id_within_clustering_group"].isna().any()

        # Recall:
        #
        # * for every sample (participant+run ID), we had their filtered sequences --> that's each row of df
        # * across samples, we grouped sequences together into "putative clusters" if they have same V gene, J gene, and CDR3 length. Those are defined by ~`putative_cross_patient_group_ID`~ `(v_gene, j_gene, cdr3_aa_sequence_trim_len)` tuple
        # * then we split each putative cluster into finer clusters based on sequence identity, requiring that the sequences within each fine cluster have >= 90% sequence identity. these finer cluster IDs are `cluster_id_within_clustering_group`
        #
        # Therefore, we must remember in this analysis that the `cluster_id_within_clustering_group` cluster IDs are only meaningful within their group (defined by `(v_gene, j_gene, cdr3_aa_sequence_trim_len)`)

        # Let's create a globally meaningful "resulting cluster ID" for each row of df (each input sequence from each participant):

        df["global_resulting_cluster_ID"] = df[
            higher_order_group_columns
            + [
                "cluster_id_within_clustering_group",
            ]
        ].apply(tuple, axis=1)

        # If any specimens did not come with num_clone_members annotations for their sequences, set to 1
        if "num_clone_members" not in df.columns:
            df["num_clone_members"] = 1
        df["num_clone_members"].fillna(1, inplace=True)

        return df

    @staticmethod
    def _get_cluster_centroids(clustered_df: pd.DataFrame) -> pd.Series:
        """
        Make cluster centroids for clusters, weighed by number of clone members (number of unique VDJ sequences).
        Returns Series of cluster centroid consensus CDR3 sequences, indexed by v_gene, j_gene, cdr3_aa_sequence_trim_len, and cluster_id_within_clustering_group.
        """
        # Consider groupby "global_resulting_cluster_ID" only and then unpacking that with MultiIndex.from_tuples -- doesn't seem much faster but maybe if doing at scale

        # See slower but clearer implementation in tests.

        # Deduplicating sequences first makes this a lot faster.
        # But we have to preserve the total count per unique sequence for correct frequencies.
        # So we can't just drop duplicates.
        sizes = (
            clustered_df.groupby(
                [
                    "v_gene",
                    "j_gene",
                    "cdr3_aa_sequence_trim_len",
                    "cluster_id_within_clustering_group",
                    "cdr3_seq_aa_q_trim",
                    "num_clone_members",
                ],
                observed=True,
                sort=False,
            )
            .size()
            .rename("size")
        )
        sizes *= sizes.index.get_level_values("num_clone_members")
        cluster_centroids = (
            sizes.reset_index()
            .groupby(
                [
                    "v_gene",
                    "j_gene",
                    "cdr3_aa_sequence_trim_len",
                    "cluster_id_within_clustering_group",
                ],
                observed=True,
                sort=False,
            )
            .apply(
                lambda grp: make_consensus_sequence(
                    grp["cdr3_seq_aa_q_trim"], grp["size"]
                )
            )
            .rename("consensus_sequence")
        )
        # TODO: might be faster if we compute cluster sizes first, then only run this for the clusters that have size > 1, and fillna on the rest
        return cluster_centroids

    @staticmethod
    def _wrap_cluster_centroids_as_dict_by_supergroup(
        cluster_centroids: Union[pd.Series, pd.DataFrame]
    ) -> Dict[Tuple[str, str, int], pd.DataFrame]:
        """
        Get cluster centroids indexed by V-J-CDR3length "supergroup".
        Input is a cluster_centroids series as produced by _get_cluster_centroids, or equivalent dataframe whose columns include _get_cluster_centroids's index and value ("consensus_sequence"), possibly filtered down to a predictive subset of clusters first.
        Returns dict mapping (v_gene, j_gene, cdr3_aa_sequence_trim_len) to DataFrame of ("cluster_id_within_clustering_group", "consensus_sequence").
        """
        return {
            key: grp[["cluster_id_within_clustering_group", "consensus_sequence"]]
            for key, grp in cluster_centroids.reset_index().groupby(
                [
                    "v_gene",
                    "j_gene",
                    "cdr3_aa_sequence_trim_len",
                ],
                observed=True,
                sort=False,
            )
        }

    @staticmethod
    def _compute_fisher_scores_for_clusters(
        train_df: pd.DataFrame, target_obs_column: TargetObsColumnEnum
    ) -> pd.DataFrame:
        """
        Compute Fisher scores for clusters in train_df. Returns DataFrame of fisher scores for clusters in train_df.
        """
        TargetObsColumnEnum.validate(target_obs_column)

        # First create 2x2 contingency table, counting the number of unique participants that are/aren't in a disease class and do/don't fall into a particular cluster
        # Clusters are defined by V-J-CDR3length-CDR3centroid

        # df of clusters x disease classes. cells are unique participants in that cluster and that disease class.
        number_of_unique_participants_by_disease_type_in_each_cluster = (
            train_df.groupby(
                [
                    "v_gene",
                    "j_gene",
                    "cdr3_aa_sequence_trim_len",
                    "cluster_id_within_clustering_group",
                    # Consider groupby "global_resulting_cluster_ID" instead, and then unpacking that with MultiIndex.from_tuples
                    target_obs_column.value.obs_column_name,
                ],
                observed=True,
            )["participant_label"]
            .nunique()
            .unstack(fill_value=0)
        )

        # row sums
        count_sum = number_of_unique_participants_by_disease_type_in_each_cluster.sum(
            axis=1
        )

        # total number of unique participants by class
        nunique_participants_by_class = train_df.groupby(
            [target_obs_column.value.obs_column_name]
        )["participant_label"].nunique()
        nunique_participants_total = nunique_participants_by_class.sum()

        # subtract
        num_unique_participants_in_each_class_that_dont_fall_into_this_cluster = (
            nunique_participants_by_class
            - number_of_unique_participants_by_disease_type_in_each_cluster
        )

        results = {}
        for (
            col
        ) in number_of_unique_participants_by_disease_type_in_each_cluster.columns:

            # number of unique participants that are in this class and are in this cluster
            right_class_and_in_this_cluster = (
                number_of_unique_participants_by_disease_type_in_each_cluster[col]
            )

            # number of unique participants that are in another class but are in this cluster
            wrong_class_but_are_in_this_cluster = (
                count_sum
                - number_of_unique_participants_by_disease_type_in_each_cluster[col]
            )

            # number of unique participants that are in this class but do not fall into this cluster
            # is the same as [total number of unique participants in this class] - [number of unique participants that are in this class and are in this cluster]
            # right_class_but_not_in_this_cluster = nunique_participants_by_class[col] - number_of_unique_participants_by_disease_type_in_each_cluster[col]
            # same as:
            # right_class_but_not_in_this_cluster = nunique_participants_by_class[col] - right_class_and_in_this_cluster
            # same as vectorized subtraction for all columns:
            right_class_but_not_in_this_cluster = (
                num_unique_participants_in_each_class_that_dont_fall_into_this_cluster[
                    col
                ]
            )

            # number of unique participants that are in another class and do not fall into this cluster
            # is the same as [total number of unique participants in another class] - [number of unique participants that are in another class but fall into this cluster]
            number_unique_participants_in_another_class = (
                nunique_participants_total - nunique_participants_by_class[col]
            )
            wrong_class_and_dont_fall_into_this_cluster = (
                number_unique_participants_in_another_class
                - wrong_class_but_are_in_this_cluster
            )

            # Contigency table is:
            # ---------------------------------------------|---------------------- |
            # is other class |                                                     |
            # ----------------      # unique participants with this property       -
            # is this class  |                                                     |
            # ---------------|-----------------------------|---------------------- |
            #                | does not fall into cluster  | falls into cluster -- |

            # Run fisher test - vectorized version
            # returns: "lefts, rights, twos"
            # Sticking to default (int64) gives error: Buffer dtype mismatch, expected 'uint_t' but got 'long'
            fisher_dtype = np.uint
            _, one_sided_p_value_right_tail, _ = pvalue_npy(
                wrong_class_and_dont_fall_into_this_cluster.values.astype(fisher_dtype),
                wrong_class_but_are_in_this_cluster.values.astype(fisher_dtype),
                right_class_but_not_in_this_cluster.values.astype(fisher_dtype),
                right_class_and_in_this_cluster.values.astype(fisher_dtype),
            )

            # sanity check shape
            assert (
                one_sided_p_value_right_tail.shape[0]
                == number_of_unique_participants_by_disease_type_in_each_cluster.shape[
                    0
                ]
            )

            # save result
            results[col] = one_sided_p_value_right_tail

        # DataFrame of p-value for each disease (columns) for each unique cluster in train_df
        results_df = pd.DataFrame(
            results,
            index=number_of_unique_participants_by_disease_type_in_each_cluster.index,
        )
        # return in sorted column order
        return results_df[sorted(results_df.columns)]

    @staticmethod
    def _merge_cluster_centroids_with_cluster_class_association_scores(
        cluster_centroids, cluster_pvalues_per_label
    ):
        # Merge score columns in. Resulting columns are "consensus_sequence" and class-specific score columns (whose names are in feature_names_order)
        # Neither dataframe has "global_resulting_cluster_ID"; they use a multiindex
        return pd.merge(
            cluster_centroids,
            cluster_pvalues_per_label,
            left_index=True,
            right_on=cluster_centroids.index.names,
            how="inner",
            validate="1:1",
        )

    @staticmethod
    def _get_model_base_dir(
        gene_locus: GeneLocus, target_obs_column: TargetObsColumnEnum
    ):
        return (
            config.paths.convergent_clusters_models_dir
            / gene_locus.name
            / combine_classification_option_names(target_obs_column)
        )

    def __init__(
        self,
        fold_id: int,
        model_name: str,
        fold_label_train: str,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        models_base_dir: Optional[Path] = None,
        sequence_identity_threshold: Optional[float] = None,
    ):
        # Load model properties from disk
        if models_base_dir is None:
            models_base_dir = self._get_model_base_dir(
                gene_locus=gene_locus, target_obs_column=target_obs_column
            )
        models_base_dir = Path(models_base_dir)

        # Load and wrap classifier
        # sets self._inner to loaded model, to expose its attributes
        super().__init__(
            inner=joblib.load(
                models_base_dir
                / f"{fold_label_train}_model.{model_name}.{fold_id}.joblib"
            ),
            fold_id=fold_id,
            fold_label_train=fold_label_train,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            models_base_dir=models_base_dir,
        )

        self.model_name = model_name
        self.sequence_identity_threshold = (
            sequence_identity_threshold
            if sequence_identity_threshold is not None
            else config.sequence_identity_thresholds.assign_test_sequences_to_clusters[
                gene_locus
            ]
        )

        # Load cross-validated p-value threshold
        self.p_value_threshold = joblib.load(
            self.models_base_dir
            / f"{self.fold_label_train}_model.{self.model_name}.{self.fold_id}.p_value.joblib"
        )
        # Load training clusters along with their class-association p-values
        self.cluster_centroids_with_class_specific_p_values = joblib.load(
            self.models_base_dir
            / f"{self.fold_label_train}_model.{self.fold_id}.{self.fold_label_train}.clusters_joblib"
        )

    @staticmethod
    def assign_sequences_to_cluster_centroids(
        test_sequences: pd.Series,
        cluster_centroids_by_supergroup: Dict[Tuple[str, str, int], pd.DataFrame],
        sequence_identity_threshold: float,
    ):
        """
        For each sequence: get nearest cluster centroid with same V-gene, J-gene, CDR3-length, subject to distance criterion.
        Also returns the distance to the nearest cluster centroid without thresholding by the distance criterion.

        Returns multiple columns packaged as single series of tuples.
        """
        # input: pandas.Series of all the sequences within a group ("group" means a V-J-CDR3-length supergroup)
        # process each group:

        # Each group is endowed the attribute ‘name’ in case you need to know which group you are working on.
        group_key = test_sequences.name

        if group_key not in cluster_centroids_by_supergroup:
            # Edge case: there are no cluster centroids with this V gene, J gene, and CDR3 length. That means we can't assign to any cluster centroid.
            # Return nan cluster ID and nan distance to nearest centroid. As below, combine the two series into one.
            two_columns_of_nans = np.full((test_sequences.shape[0], 2), np.nan)
            return pd.Series(two_columns_of_nans.tolist(), test_sequences.index)

        # cluster_centroids_by_supergroup are already narrowed down to kept_clusters only (we won't assign to a cluster that isn't predictive and will be disregarded, even if its centroid are closer than any predictive cluster)
        centroids_to_consider = cluster_centroids_by_supergroup[group_key]

        # convert array of strings into array of int-arrays
        test_sequences_as_vectors = strings_to_numeric_vectors(test_sequences.values)
        reference_sequences_as_vectors = strings_to_numeric_vectors(
            centroids_to_consider["consensus_sequence"].values
        )

        # distance matrix: normalized hamming distance
        dist_mat = cdist(
            test_sequences_as_vectors, reference_sequences_as_vectors, metric="hamming"
        )

        # mask any distances that exceed proximity threshold
        dist_mat_masked = np.ma.MaskedArray(
            dist_mat,
            dist_mat > (1 - sequence_identity_threshold),
        )

        # choose a cluster ID or NaN per test sequence (per row)
        cluster_column_index = masked_argmin(dist_mat_masked, axis=1)

        # convert from column index to actual cluster name
        mapping = {
            ix: val
            for ix, val in enumerate(
                centroids_to_consider["cluster_id_within_clustering_group"].values
            )
        }
        mapping.update({np.nan: np.nan})
        cluster_names = pd.Series(
            cluster_column_index, index=test_sequences.index
        ).replace(mapping)
        # these IDs are only unique within the overall group (the V gene, J gene, CDR3 length tuple)
        # return cluster_names

        # return both the chosen cluster IDs and also the distance to the nearest centroid (regardless of whether a cluster was assigned per the distance thresholds)
        # combine the two series as tuples
        return pd.Series(
            zip(cluster_names, np.min(dist_mat, axis=1)), index=test_sequences.index
        )

    @staticmethod
    def _validate_same_fold_id_and_label(df: pd.DataFrame):
        df["fold_id"] = pd.to_numeric(
            df["fold_id"], errors="raise"
        )  # make sure fold_id has a numeric dtype, for downstream merging

        if df["fold_id"].nunique() != 1 or df["fold_label"].nunique() != 1:
            raise ValueError("All data must be from the same fold ID and fold label.")

    @classmethod
    def _assign_sequences_to_known_clusters(
        cls,
        df: pd.DataFrame,
        cluster_centroids_by_supergroup: Dict[Tuple[str, str, int], pd.DataFrame],
        sequence_identity_threshold: float,
        validate_same_fold_id_and_label: bool = True,
    ):
        # Performs clustering. Returns new copy of df with any earlier cluster assignment columns overwritten.
        if validate_same_fold_id_and_label:
            cls._validate_same_fold_id_and_label(df)

        # Dedupe sequences to reduce redundant assign_sequences_to_cluster_centroids searches
        deduped_df = df[
            ["v_gene", "j_gene", "cdr3_aa_sequence_trim_len", "cdr3_seq_aa_q_trim"]
        ].drop_duplicates()

        # Create higher-order groups: group by v, j, len
        test_groups = deduped_df.groupby(
            ["v_gene", "j_gene", "cdr3_aa_sequence_trim_len"], observed=True
        )

        # Assign each test sequence to a cluster with nearest centroid, using higher-order groups as a starting point
        deduped_df[
            ["cluster_id_within_clustering_group", "distance_to_nearest_centroid"]
        ] = (
            test_groups["cdr3_seq_aa_q_trim"]
            .apply(
                lambda test_sequences: cls.assign_sequences_to_cluster_centroids(
                    test_sequences,
                    cluster_centroids_by_supergroup,
                    sequence_identity_threshold,
                )
            )
            .to_list()  # faster than .apply(pd.Series)
        )

        # Merge back to the full df with redundant sequences
        # If df["cluster_id_within_clustering_group"].isna(), then this test sequence is not assigned to any predictive cluster
        df = pd.merge(
            # Drop any existing cluster assignment columns
            df.drop(
                columns=[
                    "cluster_id_within_clustering_group",
                    "distance_to_nearest_centroid",
                ],
                errors="ignore",
            ),
            deduped_df,
            how="left",
            validate="m:1",
            on=["v_gene", "j_gene", "cdr3_aa_sequence_trim_len", "cdr3_seq_aa_q_trim"],
        )

        # Create a globally meaningful "resulting cluster ID" for each row of df (each input sequence from each participant):
        df["global_resulting_cluster_ID"] = df[
            [
                "v_gene",
                "j_gene",
                "cdr3_aa_sequence_trim_len",
                "cluster_id_within_clustering_group",
            ]
        ].apply(tuple, axis=1)

        # If any specimens did not come with num_clone_members annotations for their sequences, set to 1
        if "num_clone_members" not in df.columns:
            df["num_clone_members"] = 1
        df["num_clone_members"].fillna(1, inplace=True)
        return df

    def featurize(self, dataset: anndata.AnnData) -> FeaturizedData:
        """
        Pass adata.
        Make sure all data is from the same fold ID and fold label, and match the classifier's fold settings.
        """
        return self._featurize(
            df=dataset.obs,
            p_value_threshold=self.p_value_threshold,
            cluster_centroids_with_class_specific_p_values=self.cluster_centroids_with_class_specific_p_values,
            sequence_identity_threshold=self.sequence_identity_threshold,
            feature_order=self.feature_names_in_,
            gene_locus=self.gene_locus,
            target_obs_column=self.target_obs_column,
        )

    @classmethod
    def _featurize(
        cls,
        df: pd.DataFrame,
        p_value_threshold: float,
        cluster_centroids_with_class_specific_p_values: pd.DataFrame,
        sequence_identity_threshold: float,
        feature_order: List[str],
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
    ) -> FeaturizedData:
        """
        Given a p-value threshold, featurize to specimen-level feature vectors.
        Pass adata.obs.
        Make sure all data is from the same fold ID and fold label, and match the classifier's fold settings.
        """
        if (
            "cluster_id_within_clustering_group" in df.columns
            or "global_resulting_cluster_ID" in df.columns
        ):
            raise ValueError("df passed to featurize() already has cluster assignments")

        # Filter cluster_pvalues_per_disease by p_value_threshold, to a list of clusters predictive for any label(s).
        # This keeps clusters (rows) that are associated with >= 1 class.
        cluster_centroids_filtered = cluster_centroids_with_class_specific_p_values[
            cluster_centroids_with_class_specific_p_values[feature_order].min(axis=1)
            <= p_value_threshold
        ]

        # Assign each test sequence to known cluster with nearest centroid
        df = cls._assign_sequences_to_known_clusters(
            df=df,
            # Get a cluster_centroids_by_supergroup dict that is filtered down to those kept clusters only.
            cluster_centroids_by_supergroup=cls._wrap_cluster_centroids_as_dict_by_supergroup(
                cluster_centroids_filtered
            ),
            sequence_identity_threshold=sequence_identity_threshold,
        )

        # Also get class association of each cluster (some may be associated with multiple classes)
        # DataFrame indexed by (v_gene, j_gene, cdr3_aa_sequence_trim_len, cluster_id_within_clustering_group), with columns "cluster_dominant_label" and associated "p_value"
        cluster_class_associations = cluster_centroids_filtered[feature_order].melt(
            # preserve index
            ignore_index=False,
            var_name="cluster_dominant_label",
            value_name="p_value",
        )
        # Refilter. Remember we kept all clusters that were associated with >= 1 class, but after the melt we may have multiple rows with high p-values for the other classes for each cluster.
        cluster_class_associations = cluster_class_associations[
            cluster_class_associations["p_value"] <= p_value_threshold
        ]

        ## Make table of predictive-cluster memberships for each specimen.
        # Entries of df have only been matched to predictive clusters only, because centroids list was constructed from kept clusters only.
        # For a particular cluster: for a particular specimen:
        # * `total_num_clone_members` (sum): total number of unique sequences per specimen that fall into the cluster
        # * `total_num_clones` (size): number of clones per specimen that fall into the cluster
        #    - this size should be 1 since choosing only 1 sequence per clone per specimen
        predictive_cluster_membership_and_sizes = (
            df[~df["cluster_id_within_clustering_group"].isna()]
            .groupby(
                helpers.get_obs_column_list_with_target_obs_column_included(
                    [
                        # "global_resulting_cluster_ID",
                        "v_gene",
                        "j_gene",
                        "cdr3_aa_sequence_trim_len",
                        "cluster_id_within_clustering_group",
                        "participant_label",
                        "specimen_label",
                        "disease",
                        "disease_subtype",
                        "fold_id",
                        "fold_label",
                        "past_exposure",
                        "disease.separate_past_exposures",
                    ],
                    target_obs_column,
                ),
                observed=True,
            )["num_clone_members"]
            .agg(["sum", "size"])
            .rename(
                columns={"sum": "total_num_clone_members", "size": "total_num_clones"}
            )
            .reset_index()
        )

        # Annotate predictive cluster memberships with information about the predictive cluster
        predictive_cluster_membership_and_sizes_annot = pd.merge(
            # Specimen memberships in clusters.
            predictive_cluster_membership_and_sizes,
            # Associations of clusters with classes
            # Note that a cluster may be associated with multiple classes.
            # (For example, at p_value_threshold=1.0, all clusters are associated with all classes.)
            # (Therefore this can be a many-to-many merge.)
            cluster_class_associations,
            how="inner",
            on=[
                "v_gene",
                "j_gene",
                "cdr3_aa_sequence_trim_len",
                "cluster_id_within_clustering_group",
            ],
        )
        if (
            predictive_cluster_membership_and_sizes_annot.shape[0]
            < predictive_cluster_membership_and_sizes.shape[0]
        ):
            # The merge above should not have lost any rows.
            # The number of rows should be equal if there are no clusters associated with multiple classes.
            # The number of rows should increase if any clusters are associated with multiple classes (e.g. with a very permissive p-value threshold)
            raise ValueError(
                f"n_rows decreased unexpectedly during merge. Expected at least {predictive_cluster_membership_and_sizes.shape[0]} rows, but only got {predictive_cluster_membership_and_sizes_annot.shape[0]} rows."
            )

        # Get specimen scores.
        # The feature matrix is in .values of this dataframe
        # Specimen label, fold ID, and fold label are in .index (a multi-index)
        specimen_scores = cls._get_specimen_score(
            predictive_cluster_membership_and_sizes_annot
        )

        # TODO: Sanity check that all data is from same fold ID and fold label?

        # We are assuming that all data is from same fold ID and fold label,
        # so discard those multi-index levels.
        # Condense multi-index to specimen label level only.
        specimen_scores.index = specimen_scores.index.get_level_values("specimen_label")

        # Extract x, obs, and y
        X = specimen_scores

        # Reorder feature matrix according to feature_names_in_ order, inserting 0s for any missing features
        X = X.reindex(columns=feature_order).fillna(0)

        # Extract specimen metadata
        obs = helpers.extract_specimen_metadata_from_obs_df(
            df=df, gene_locus=gene_locus, target_obs_column=target_obs_column
        )

        # Find abstentions: any missing specimens that did not have a single sequence fall into a predictive cluster
        # These specimens will be in obs but will not have a row in X.
        abstained_specimens = obs.loc[obs.index.difference(X.index)].copy()

        # Make order match
        scored_specimens = obs.index.intersection(X.index)
        X = X.loc[scored_specimens]
        obs = obs.loc[scored_specimens]

        # extract target metadata column
        target_obs_col_name = target_obs_column.value.obs_column_name
        y = obs[target_obs_col_name]
        abstained_specimens_ground_truth_labels = abstained_specimens[
            target_obs_col_name
        ]

        # Confirm no rows of all 0s -- these should be abstained specimens
        if (X.values == 0).all(axis=1).any():
            raise ValueError(
                "Some specimens (feature matrix rows) have all 0s. These should be abstentions."
            )

        return FeaturizedData(
            X=X,
            y=y,
            metadata=obs,
            sample_names=obs.index,
            abstained_sample_y=abstained_specimens_ground_truth_labels,
            abstained_sample_names=abstained_specimens.index,
            abstained_sample_metadata=abstained_specimens,
            extras={
                "p_value_threshold": p_value_threshold,
            },
        )

    @staticmethod
    def _get_specimen_score(predictive_cluster_membership_and_sizes: pd.DataFrame):
        """Individual specimen's score (for a particular disease) = number of predictive (i.e. high-purity, many-patients) clusters into which some sequences from the specimen have clustered. I.e. presence/absence of convergent IGH sequences"""
        # Input doesn't have "global_resulting_cluster_ID" anymore. TODO: pass this in so we don't have to recreate

        # Create a globally meaningful "resulting cluster ID" for each row of df (each input sequence from each participant):
        predictive_cluster_membership_and_sizes[
            "global_resulting_cluster_ID"
        ] = predictive_cluster_membership_and_sizes[
            [
                "v_gene",
                "j_gene",
                "cdr3_aa_sequence_trim_len",
                "cluster_id_within_clustering_group",
            ]
        ].apply(
            tuple, axis=1
        )

        # For each specimen: for each cluster_dominant_label: number of unique global_resulting_cluster_IDs
        specimen_scores = (
            predictive_cluster_membership_and_sizes.groupby(
                [
                    "specimen_label",
                    "fold_id",
                    "fold_label",
                    "cluster_dominant_label",
                ],
                observed=True,
            )["global_resulting_cluster_ID"]
            .nunique()
            .rename("score")
        )

        # Pivot to a feature matrix per specimen. And fill NaNs with 0 (i.e. no clusters of that type)
        specimen_scores_pivot = (
            specimen_scores.reset_index()
            .pivot(
                index=[
                    "specimen_label",
                    "fold_id",
                    "fold_label",
                ],
                columns="cluster_dominant_label",
                values="score",
            )
            .fillna(0)
        )
        # Change columns to not be a CategoricalIndex, so any downstream reset_index() will not fail with TypeError: cannot insert an item into a CategoricalIndex that is not already an existing category
        specimen_scores_pivot.columns = specimen_scores_pivot.columns.astype(str)
        return specimen_scores_pivot
