import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import anndata
import genetools
import joblib
import numpy as np
import pandas as pd
from sklearn.utils._testing import ignore_warnings

from malid import config, helpers
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    combine_classification_option_names,
    healthy_label,
)
from extendanything import ExtendAnything
from static_class_property import classproperty
from malid.external.model_evaluation import FeaturizedData
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)

logger = logging.getLogger(__name__)


class RepertoireClassifier(ImmuneClassifierMixin, ExtendAnything):
    """Wrapper around repertoire stats classification models. predict(), predict_proba(), and decision_function() all accept anndata arguments representing repertoire(s) from same fold; featurization happens automatically."""

    # ExtendAnything means will pass through to base instance's attributes, exposing the base model's classes_, predict(), predict_proba(), etc.

    sick_binary_label = "Active Infection"
    n_pcs = 15

    def __init__(
        self,
        fold_id: int,
        model_name: str,
        fold_label_train: str,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        models_base_dir: Optional[Path] = None,
    ):
        if models_base_dir is None:
            models_base_dir = self._get_model_base_dir(
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
            )
        models_base_dir = Path(models_base_dir)

        fname = (
            models_base_dir / f"{fold_label_train}_model.{model_name}.{fold_id}.joblib"
        )

        # sets self._inner to loaded model, to expose its attributes
        # Load and wrap classifier
        super().__init__(
            inner=joblib.load(fname),
            fold_id=fold_id,
            fold_label_train=fold_label_train,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            models_base_dir=models_base_dir,
        )

        self.model_name = model_name

    @staticmethod
    def _get_model_base_dir(
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
    ) -> Path:
        return (
            config.paths.repertoire_stats_classifier_models_dir
            / gene_locus.name
            / combine_classification_option_names(target_obs_column)
        )

    @classproperty
    def _features_from_obs(cls) -> Dict[GeneLocus, List[str]]:
        # Static read-only property:
        # What features do we want to extract from adata.obs?
        features_from_obs = {}
        for gene_locus in GeneLocus:
            features_from_obs[gene_locus] = []
            for isotype_group in helpers.isotype_groups_kept[gene_locus]:
                if gene_locus == GeneLocus.BCR:
                    features_from_obs[gene_locus].extend(
                        [
                            f"v_mut_median_per_specimen:{isotype_group}",
                            f"v_sequence_is_mutated:{isotype_group}",
                        ]
                    )
        return features_from_obs

    def featurize(self, dataset: anndata.AnnData) -> FeaturizedData:
        """
        Featurize repertoire composition.
        compute repertoire stats features for specimen(s) from a single fold.
        you must ensure all specimens do come from the same fold!
        returns one row (feature vector) per specimen.

        don't use this for initial training. use this for running a trained model on new data.
        (for training, instead use _featurize directly with vj_count_matrix_column_order=None,
        which generates the V-J count matrix structure rather than loading it)

        Allows missing isotypes, because we don't know whether test specimens will have all desired isotypes necessarily.
        But beware that the models may behave weirdly if an important isotype is missing.
        (For training, call _featurize with allow_missing_isotypes=False)

        fold_id and fold_label_train come from class properties
        """

        # get repertoire stats model features for full test set (not high-confidence subset)
        # dataset includes many specimens

        # Load columns list from the training fold used for this model version
        train_count_matrix_columns = joblib.load(
            self.models_base_dir
            / f"{self.fold_label_train}_model.{self.fold_id}.{self.fold_label_train}.specimen_vj_gene_counts_columns_joblib"
        )

        featurized = self._featurize(
            dataset=dataset,
            gene_locus=self.gene_locus,
            target_obs_column=self.target_obs_column,
            allow_missing_isotypes=True,
            vj_count_matrix_column_order=train_count_matrix_columns,
        )

        # confirm that feature order matches what model expects
        if not np.array_equal(
            featurized.X.columns, self._inner["columntransformer"].feature_names_in_
        ):
            raise ValueError(
                "Featurization returned incorrect columns or incorrect column order."
            )

        return featurized

    @classmethod
    def _featurize(
        cls,
        dataset: anndata.AnnData,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        allow_missing_isotypes: bool,
        vj_count_matrix_column_order: Optional[Dict[str, pd.Index]],
    ):
        """Featurize."""
        (
            data_X,
            adata_featurized,
            vj_count_matrix_columns_by_isotype,
        ) = cls._create_repertoire_stats(
            df=dataset.obs,
            gene_locus=gene_locus,
            allow_missing_isotypes=allow_missing_isotypes,
            vj_count_matrix_column_order=vj_count_matrix_column_order,
        )

        specimen_order = data_X.index
        data_y, metadata = cls._get_ground_truth_repertoire_labels(
            dataset=dataset,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
        )

        # Reorder and return
        return FeaturizedData(
            X=data_X,
            y=data_y.loc[specimen_order],
            sample_names=specimen_order,
            metadata=metadata.loc[specimen_order],
            extras={
                "vj_count_matrix_columns_by_isotype": vj_count_matrix_columns_by_isotype
            },
        )

    @classmethod
    def _get_ground_truth_repertoire_labels(
        cls,
        dataset: anndata.AnnData,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
    ):
        """get ground truth labels and metadata for repertoire classification from a dataset containing many specimens from single foldID+label"""
        TargetObsColumnEnum.validate(target_obs_column)
        GeneLocus.validate(gene_locus)

        # dataset is not a single repertoire; it includes many specimens
        specimen_metadata = helpers.extract_specimen_metadata_from_anndata(
            adata=dataset, gene_locus=gene_locus, target_obs_column=target_obs_column
        )

        y_values = specimen_metadata[target_obs_column.value.obs_column_name]

        if (
            target_obs_column.value.is_target_binary_for_repertoire_composition_classifier
        ):
            # make binary Healthy or Sick labels
            y_values = y_values.copy()

            # probably Categorical, so cast defensively to categorical and register new categories first
            y_values = y_values.astype("category").cat.add_categories(
                cls.sick_binary_label
            )
            if healthy_label not in y_values.cat.categories:
                y_values = y_values.cat.add_categories(healthy_label)

            # mark all who aren't Healthy as sick
            y_values[y_values != healthy_label] = cls.sick_binary_label

            # mark survivors (past exposures) as now Healthy
            y_values.loc[specimen_metadata["past_exposure"]] = healthy_label

            # remove unused categories
            y_values = y_values.cat.remove_unused_categories()

            specimen_metadata["healthy_or_sick_binary"] = y_values

        return y_values, specimen_metadata

    @classmethod
    # Ignore "ImplicitModificationWarning: Transforming to str index."
    @ignore_warnings(category=anndata.ImplicitModificationWarning)
    def _create_repertoire_stats(
        cls,
        df,
        gene_locus: GeneLocus,
        allow_missing_isotypes: bool,
        vj_count_matrix_column_order: Optional[Dict[str, pd.Index]],
    ) -> Tuple[pd.DataFrame, anndata.AnnData, Dict[str, pd.Index]]:
        """Pass vj_count_matrix_column_order=None to generate V-J count matrix structure"""
        # copy before making changes
        df = df.copy()

        if not all(
            df["isotype_supergroup"].isin(helpers.isotype_groups_kept[gene_locus])
        ):
            # had extra isotypes beyond what we expected. we don't support these additional isotypes.
            raise ValueError("All isotype_supergroups must be in whitelisted set")

        if gene_locus == GeneLocus.BCR:
            # Make features for each isotype-supergroup (amplified separately, so don't compare across them) -- aggregate to specimen level (within each fold)

            v_mut_median_by_specimen = (
                df.groupby(["specimen_label", "isotype_supergroup"], observed=True)[
                    "v_mut"
                ]
                .median()
                .rename("v_mut_median_per_specimen")
            )

            # proportion of nonmutated vs mutated IgG (of any subisotypes)
            # (allows analysis of datasets that don't distinguish subisotypes)
            # Looking for low-mutation-level (recently class switched) B cells
            df["v_sequence_is_mutated"] = df["v_mut"] >= 0.01
            mutated_proportion_of_sequences_by_specimen = (
                df.groupby(["specimen_label", "isotype_supergroup"], observed=True)[
                    "v_sequence_is_mutated"
                ]
                .value_counts(normalize=True)
                .to_frame(name="proportion_of_sequences_mutated")
                .reset_index()
            )
            # Choose "v_sequence_is_mutated=True" rows
            mutated_proportion_of_sequences_by_specimen = (
                mutated_proportion_of_sequences_by_specimen[
                    mutated_proportion_of_sequences_by_specimen["v_sequence_is_mutated"]
                    == True
                ].drop("v_sequence_is_mutated", axis=1)
            )
            # However it's possible that some specimens had no mutated V sequences at all, and thus aren't in this table
            # Let's readd them, with 0% mutated
            mutated_proportion_of_sequences_by_specimen = (
                mutated_proportion_of_sequences_by_specimen.set_index(
                    ["specimen_label", "isotype_supergroup"]
                ).reindex(
                    pd.MultiIndex.from_product(
                        [
                            df["specimen_label"].unique(),
                            df["isotype_supergroup"].unique(),
                        ],
                        names=["specimen_label", "isotype_supergroup"],
                    ),
                    fill_value=0.0,
                )
            )

            # sanity check that we are left with one entry per specimen (even specimens that had no mutated sequences and were filled to 0%)
            for (
                isotype_supergroup,
                grp,
            ) in mutated_proportion_of_sequences_by_specimen.groupby(
                "isotype_supergroup", observed=True
            ):
                if grp.shape[0] != df["specimen_label"].nunique():
                    raise ValueError(
                        "Some specimen had no mutated V sequences at all and was dropped from the table"
                    )

        # V and J gene pair count matrix
        df["vgene_jgene"] = df["v_gene"].astype(str) + "|" + df["j_gene"].astype(str)

        specimen_vj_gene_counts = (
            df.groupby(["specimen_label", "isotype_supergroup"], observed=True)[
                "vgene_jgene"
            ]
            .value_counts(normalize=True)
            .rename("frequency")
            .reset_index()
        )

        vj_gene_use_count_matrices_by_isotype = {}
        for isotype_supergroup in helpers.isotype_groups_kept[gene_locus]:
            # Process all isotype_supergroups, even the ones not present in the repertoire
            grp = specimen_vj_gene_counts[
                specimen_vj_gene_counts["isotype_supergroup"] == isotype_supergroup
            ]

            # pivot to create a counts matrix
            specimen_vj_gene_counts_matrix = pd.pivot_table(
                grp,
                index=["specimen_label"],
                columns="vgene_jgene",
                values="frequency",
            ).fillna(0)

            # within each fold: make columns consistent between train and test
            # - columns (vgene-jgene pairs) that exist in train but don't exist in test: add column to test and fillna with 0
            # - columns that exist in test but don't exist in train: drop from test
            # - reorder test to match train's list of columns

            # Rename columns to include isotype-supergroup name, so that combining the count matrices later will result in unique column names
            specimen_vj_gene_counts_matrix = specimen_vj_gene_counts_matrix.rename(
                columns=lambda col: f"{col}:{isotype_supergroup}"
            )

            if vj_count_matrix_column_order is not None:
                # Suppose we are processing the test matrix, and we just loaded the column name order of the train matrix
                # first downselect this test count matrix to the intersection of columns
                train_count_matrix_columns_for_this_isotype = (
                    vj_count_matrix_column_order[isotype_supergroup]
                )
                specimen_vj_gene_counts_matrix = specimen_vj_gene_counts_matrix[
                    train_count_matrix_columns_for_this_isotype.intersection(
                        specimen_vj_gene_counts_matrix.columns
                    )
                ]

                # then reindex to train's full list of columns, and fill NA columns with 0.
                # first: change multiindex columns to non-hierarchical standard index. otherwise, if this isotype is missing and there are no columns to start, reindex will fail
                specimen_vj_gene_counts_matrix.columns = (
                    specimen_vj_gene_counts_matrix.columns.to_flat_index()
                )
                specimen_vj_gene_counts_matrix = specimen_vj_gene_counts_matrix.reindex(
                    columns=train_count_matrix_columns_for_this_isotype,
                ).fillna(0)

                # reorder
                specimen_vj_gene_counts_matrix = specimen_vj_gene_counts_matrix[
                    train_count_matrix_columns_for_this_isotype
                ]
            else:
                logger.info(
                    "Generating V-J gene count matrix structure from scratch, i.e. not filtering to a specific column list + order"
                )

            # also reindex to include all specimens, even those with no data for this isotype, and fill N/As with 0
            specimen_vj_gene_counts_matrix = specimen_vj_gene_counts_matrix.reindex(
                index=df["specimen_label"].unique(),
            ).fillna(0)

            # adjust for sampling depth (normalize each row to sum to 1)
            # this renormalizing is important because we may have dropped columns if vj_count_matrix_column_order was provided
            specimen_vj_gene_counts_matrix = genetools.stats.normalize_rows(
                specimen_vj_gene_counts_matrix
            )

            # it's possible for a row to have been all 0s originally, specifically if a specimen has no data for this isotype (row of 0s added via reindex above)
            # in this case, normalize_rows() will set the row to all NaNs, because there's no way to make it sum to 1
            # so follow this up by changing NaNs to 0s
            specimen_vj_gene_counts_matrix.fillna(0, inplace=True)

            vj_gene_use_count_matrices_by_isotype[
                isotype_supergroup
            ] = specimen_vj_gene_counts_matrix

        ## combine into joint anndata (i.e. match by index)

        # horizontally combine the count matrices from different isotype supergroups
        # variable/column names will still be unique because we added isotype-supergroup names to the column names previously above
        count_matrix_combined = pd.concat(
            vj_gene_use_count_matrices_by_isotype.values(), axis=1
        )

        # mark which columns belong to which original isotype-supergroup count matrix
        column_provenance = np.hstack(
            [
                np.tile(isotype_supergroup, isotype_supergroup_count_matrix.shape[1])
                for (
                    isotype_supergroup,
                    isotype_supergroup_count_matrix,
                ) in vj_gene_use_count_matrices_by_isotype.items()
            ]
        )

        assert count_matrix_combined.shape[1] == column_provenance.shape[0]

        adata = anndata.AnnData(
            X=count_matrix_combined,
            # Set dtype explicitly to solve this warning:
            # "FutureWarning: X.dtype being converted to np.float32 from float64. In the next version of anndata (0.9) conversion will not be automatic. Pass dtype explicitly to avoid this warning. Pass `AnnData(X, dtype=X.dtype, ...)` to get the future behavour."
            dtype=count_matrix_combined.dtypes.iloc[0],
        )
        adata.var["isotype_supergroup"] = column_provenance

        if gene_locus == GeneLocus.BCR:
            ## Add other features for each isotype supergroup

            # Get v_mut_median_by_specimen
            mutation_rates = v_mut_median_by_specimen.reset_index().pivot(
                index="specimen_label",
                columns="isotype_supergroup",
                values="v_mut_median_per_specimen",
            )
            # Check each specimen's isotypes against desired list
            if not allow_missing_isotypes and (
                mutation_rates.isna().any().any()
                or set(mutation_rates.columns)
                != set(helpers.isotype_groups_kept[gene_locus])
            ):
                raise ValueError(
                    "Some specimens are missing some isotype_supergroups from whitelisted set. We require all isotypes to be present for repertoire stats featurization."
                )
            # Merge in
            adata.obs = genetools.helpers.merge_into_left(
                adata.obs,
                mutation_rates.rename(
                    columns=lambda isotype_supergroup: f"v_mut_median_per_specimen:{isotype_supergroup}"
                ),
            )
            adata.obs = genetools.helpers.merge_into_left(
                adata.obs,
                # This should be covered by the above N/A check
                mutated_proportion_of_sequences_by_specimen.reset_index()
                .pivot(
                    index="specimen_label",
                    columns="isotype_supergroup",
                    values="proportion_of_sequences_mutated",
                )
                .rename(
                    columns=lambda isotype_group: f"v_sequence_is_mutated:{isotype_group}"
                ),
            )

        ## concatenate into one dataframe, not yet transformed

        # What features do we want to extract from adata.obs?
        if allow_missing_isotypes:
            for feature in cls._features_from_obs[gene_locus]:
                if feature not in adata.obs.columns:
                    # This isotype-supergroup was missing in all specimens.
                    # Add this feature with 0s
                    adata.obs[feature] = 0.0
                    logger.warning(
                        f"Feature {feature} was missing — the isotype group is not in the data. Added this feature (and V-J gene counts for this isotype) as all 0s."
                    )
                elif adata.obs[feature].isna().any():
                    # Also fill N/A values with 0 in case some specimens were missing some isotypes
                    adata.obs[feature].fillna(0.0, inplace=True)
                    logger.warning(
                        f"Filled missing entries in feature {feature} with 0s (this isotype group was not in all specimens)."
                    )

        X = adata.obs[cls._features_from_obs[gene_locus]].copy()
        # concat and rename Vgene-Jgene count matrix columns to have "pca" prefix (log1p, scale, and PCA will be applied in the sklearn pipeline)
        for isotype_group in helpers.isotype_groups_kept[gene_locus]:
            X = pd.concat(
                [
                    X,
                    adata[:, adata.var["isotype_supergroup"] == isotype_group]
                    .to_df()
                    .rename(columns=lambda col: f"{isotype_group}:pca_{col}"),
                ],
                axis=1,
            )

        ## export V-J count matrix column order so we can create features for new test sets later
        vj_count_matrix_columns_by_isotype = {
            isotype_supergroup: isotype_supergroup_specimen_vj_gene_counts_matrix.columns
            for isotype_supergroup, isotype_supergroup_specimen_vj_gene_counts_matrix in vj_gene_use_count_matrices_by_isotype.items()
        }

        return X, adata, vj_count_matrix_columns_by_isotype
