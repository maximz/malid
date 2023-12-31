from dataclasses import dataclass
from collections import defaultdict
import functools
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re

import anndata
import joblib
import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.preprocessing import MatchVariables
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from regressout import RegressOutCovariates

from malid import config, io, helpers
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
from extendanything import ExtendAnything
from malid.external.model_evaluation import FeaturizedData
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
    MetadataFeaturizerMixin,
)

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class MetamodelConfig:
    # For each gene locus: provide submodels along with their names (used as feature prefixes)
    # Submodels implement abstract ImmuneClassifierMixin interface, i.e. they have a featurize(adata)->df method.
    # The submodels are gene-locus specific.
    submodels: Optional[Dict[GeneLocus, Dict[str, ImmuneClassifierMixin]]]

    # Also support featurizers like DemographicsFeaturizer that are non-gene-locus specific and operate on a dataframe, not an anndata.
    # Provide those additional non-gene-locus specific metadata featurizers here, along with their names (used as feature prefixes)
    extra_metadata_featurizers: Optional[Dict[str, MetadataFeaturizerMixin]] = None

    # Support interaction terms
    interaction_terms: Optional[Tuple[List[str], List[str]]] = None

    # Support regressing out covariates from the computed features.
    regress_out_featurizers: Optional[Dict[str, MetadataFeaturizerMixin]] = None
    regress_out_pipeline: Optional[Pipeline] = None  # Set after first train featurize

    # Other metamodel settings
    sample_weight_strategy: SampleWeightStrategy = SampleWeightStrategy.ISOTYPE_USAGE


class DemographicsFeaturizer(MetadataFeaturizerMixin):
    """
    DemographicsFeaturizer extracts demographics features from an anndata.
    It implements the abstract MetadataFeaturizerMixin interface and has a featurize(DataFrame)->DataFrame function.
    It can be included in a BlendingMetamodel, which will add demographics columns alongside any other features.
    """

    def __init__(self, covariate_columns: List[str]):
        # Make sure that covariate_columns are also listed in helpers.extract_specimen_metadata_from_obs_df() so that this metadata is available when training metamodel.
        self.covariate_columns = covariate_columns
        self.one_hot_encoder_ = None

    def _get_one_hot_encodings(self, df: pd.DataFrame):
        """Fit and save the one-hot encoder, then return the one-hot encoded dataframe."""
        # Find categorical columns that need to become dummy variables.
        # Pass continuous variables like age through without modification.
        categorical_variables: List[Union[str, int]] = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if len(categorical_variables) == 0:
            # Special case: no categorical variables. Pass through.
            return df

        if self.one_hot_encoder_ is None:
            self.one_hot_encoder_ = OneHotEncoder(
                variables=categorical_variables,
                # For variables with only two categories (like sex), create only one dummy variable.
                drop_last_binary=True,
                # For variables with >2 categories, create dummy variables for all categories; don't drop the final category.
                drop_last=False,
            )
            self.one_hot_encoder_.fit(df)

            # For binary variables that are being reduced to a single dummy variable, standardize which of the two categories is chosen.
            # This is not necessary to achieve the same feature names between train and test time; that's already built-in because the one_hot_encoder_ is fit on the train data and saved.
            # But if we want to compare or aggregate feature importances across models trained on different folds, we need to standardize the feature names.
            # For example, if "sex" is a binary categorical variable with a defined category order ["M", "F"], ensure that "sex=F" is always the dummy variable produced, and never "sex=M".
            for binary_var_name in self.one_hot_encoder_.variables_binary_:
                # check if this is an ordered categorical column
                if (
                    pd.api.types.is_categorical_dtype(df[binary_var_name])
                    and df[binary_var_name].cat.ordered
                ):
                    categories = df[binary_var_name].cat.categories
                    if len(categories) != 2:
                        # Maybe not all categories were present; i.e. this might be more than a binary variable but only appeared binary in this dataset.
                        # Skip it
                        logger.warning(
                            f"Attempted to standardize binary variable {binary_var_name}'s one hot encoding but it actually has {len(categories)} categories, not 2."
                        )
                        continue
                    else:
                        # This is a binary variable with a defined category order.
                        # Ensure that the last category is the one that gets kept, and the first category is the one that is dropped.
                        self.one_hot_encoder_.encoder_dict_[binary_var_name] = [
                            categories[-1]
                        ]

        return self.one_hot_encoder_.transform(df)

    def featurize(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # TODO: should we check for NaNs and throw error?
        demographics_df = self._get_one_hot_encodings(dataset[self.covariate_columns])

        # TODO: unit test that this returns pandas dataframe with correct columns and index matches dataset sample_names, and shape matches dataset.X
        # TODO: unit test that confirms failure if dataset does not include covariate_columns
        return demographics_df


def cartesian_product(
    df: pd.DataFrame,
    features_left: List[str],
    features_right: List[str],
    separator="|",
    prefix="interaction",
) -> pd.DataFrame:
    """Add interaction terms: Cartesian product of df[features_left] and df[features_right]"""
    # TODO: Unit test: shape[0] stays the same, shape[1] expands
    def _dedupe_preserving_order(features: List[str]) -> pd.Series:
        # dedupe without changing order
        return pd.Series(features).drop_duplicates()

    # TODO: speed this up with numpy elementwise matrix multiplcation - something like:
    # pd.DataFrame(
    #   np.multiply(df[_dedupe_preserving_order(features_left)].values, df[_dedupe_preserving_order(features_right)].values),
    #   columns=[separator.join([prefix, feature1, feature2]) for feature1 in _dedupe_preserving_order(features_left) for feature2 in _dedupe_preserving_order(features_right)]
    # )
    return pd.concat(
        [df]
        + [
            (df[feature1] * df[feature2]).rename(
                separator.join([prefix, feature1, feature2])
            )
            for feature1 in _dedupe_preserving_order(features_left)
            for feature2 in _dedupe_preserving_order(features_right)
        ],
        axis=1,
    )


class BlendingMetamodel(ExtendAnything):
    """Blending meta-model that combines repertoire stats model, sequence disease classifier, and optionally convergent cluster classifier.

    Featurize manually using `featurize()` before calling `predict()`, `predict_proba()`, or `decision_function()`.

    Usage for an anndata with multiple specimens:
    ```
        # featurize
        featurized = clf.featurize(subset_adata)

        # Handle abstention
        if featurized.X.shape[0] == 0:
            return np.array(["Unknown"]), np.array([np.nan])

        blended_model_predicted_label = clf.predict(featurized.X)
        blended_results_proba = clf.predict_proba(featurized.X)
    ```

    Note that this classifier handles multiple specimens at once.
    If you pass an anndata with a single specimen, this will still return array shapes as though there are multiple specimens, rather than flattened.
    """

    # ExtendAnything means will pass through to base instance's attributes, exposing the base model's classes_, predict(), predict_proba(), etc.

    @classmethod
    def _get_metamodel_base_dir(
        cls,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        metamodel_flavor: str,
    ) -> Path:
        """
        The metamodel base dir captures the gene locus (or loci), the classification target, and the name of the metamodel flavor (corresponds to all the metamodel settings)

        Within this base dir we will store different metamodel models (e.g. random forest vs lasso at this 2nd stage metamodel step) for different fold IDs.
        """
        return (
            config.paths.second_stage_blending_metamodel_models_dir
            / gene_locus.name
            / target_obs_column.name
            / metamodel_flavor
        )

    @classmethod
    def from_disk(
        cls,
        fold_id: int,
        metamodel_name: str,
        base_model_train_fold_name: str,
        metamodel_fold_label_train: str,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        metamodel_flavor: str,
    ):
        metamodel_base_dir = cls._get_metamodel_base_dir(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            metamodel_flavor=metamodel_flavor,
        )
        metamodel_config = joblib.load(
            metamodel_base_dir
            / f"{base_model_train_fold_name}_applied_to_{metamodel_fold_label_train}_model.{fold_id}.metamodel_components.joblib"
        )["metamodel_config"]
        return cls(
            fold_id=fold_id,
            metamodel_name=metamodel_name,
            base_model_train_fold_name=base_model_train_fold_name,
            metamodel_fold_label_train=metamodel_fold_label_train,
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            metamodel_config=metamodel_config,
            metamodel_base_dir=metamodel_base_dir,
        )

    def __init__(
        self,
        fold_id: int,
        metamodel_name: str,
        base_model_train_fold_name: str,
        metamodel_fold_label_train: str,
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        metamodel_config: MetamodelConfig,
        metamodel_base_dir: Path,
    ):
        """Load trained metamodel from disk.
        metamodel_config contains submodels, other featurizers, and other settings. ALl of these are stored on disk. Use BlendingMetamodel.from_disk to load and initialize.

        Rationale:
        Storing the submodels inside the BlendingMetamodel, including when it is serialized to disk, guarantees that the exact same submodels are run whenever the BlendingMetamodel is used later, so the features are consistent between train and test time.
        Importing the BlendingMetamodel later imports it along with the submodels literally bundled in!
        """
        # Validation
        GeneLocus.validate(gene_locus)
        TargetObsColumnEnum.validate(target_obs_column)
        SampleWeightStrategy.validate(metamodel_config.sample_weight_strategy)

        if (
            metamodel_config.regress_out_featurizers is not None
            and len(metamodel_config.regress_out_featurizers) > 0
        ):
            if metamodel_config.regress_out_pipeline is None:
                raise ValueError(
                    "If regress_out_featurizers is not None, regress_out_pipeline must be specified"
                )

        self.fold_id = fold_id
        self.metamodel_name = metamodel_name
        self.base_model_train_fold_name = base_model_train_fold_name
        self.metamodel_fold_label_train = metamodel_fold_label_train
        self.gene_locus = gene_locus
        self.target_obs_column = target_obs_column
        self.metamodel_config = metamodel_config
        self.metamodel_base_dir = Path(metamodel_base_dir)  # defensive cast

        # Load and wrap classifier
        fname = (
            self.metamodel_base_dir
            / f"{self.base_model_train_fold_name}_applied_to_{self.metamodel_fold_label_train}_model.{self.metamodel_name}.{self.fold_id}.joblib"
        )
        clf = joblib.load(fname)
        if not hasattr(clf, "feature_names_in_"):
            raise ValueError(
                "Loaded classifier must have a feature_names_in_ attribute."
            )
        # sets self._inner to loaded model, to expose its attributes
        super().__init__(clf)

    def featurize(
        self,
        adatas: Dict[GeneLocus, anndata.AnnData],
    ) -> FeaturizedData:
        """
        Featurizes one adata (set of many specimen repertoires) per gene locus. Uses many input featurizers and horizontally stacks their features.
        Imposes a consistent feature order.
        (Make sure to supply all sequences, not just high-confidence subset, because we will run repertoire classifier.)
        """
        featurized = self._featurize(
            data=adatas,
            gene_locus=self.gene_locus,
            target_obs_column=self.target_obs_column,
            metamodel_config=self.metamodel_config,
        )
        # Apply feature order
        feature_order = self._inner.feature_names_in_
        featurized.X = featurized.X[feature_order]
        return featurized

    @classmethod
    def _featurize(
        cls,
        data: Dict[GeneLocus, anndata.AnnData],
        gene_locus: GeneLocus,
        target_obs_column: TargetObsColumnEnum,
        metamodel_config: MetamodelConfig,
    ) -> FeaturizedData:
        """
        Featurizes one adata (set of many specimen repertoires) per gene locus. Uses many input featurizers and horizontally stacks their features.
        (Make sure to supply all sequences, not just high-confidence subset, because we will run repertoire classifier.)

        Does NOT impose a particular feature order for the returned X.

        Featurizes with models (loads models 1+2+3 from disk, but generates model3-rollup from scratch)
        Optionally regresses out covariates
        Optionally concatenates coviarate features, possibly with interaction terms
        Returns this in a way that we can apply to new data.
        """

        GeneLocus.validate(gene_locus)
        TargetObsColumnEnum.validate(target_obs_column)
        SampleWeightStrategy.validate(metamodel_config.sample_weight_strategy)

        model_outputs = []
        metadata_objects_by_single_locus = defaultdict(list)

        def _subselect_columns_if_binary(y_preds_proba: pd.DataFrame) -> pd.DataFrame:
            """
            If binary model, switch to single column, rather than including p and 1-p as features in the metamodel.
            But keep it a DataFrame rather than a Series.
            """
            if y_preds_proba.shape[1] == 2:
                y_preds_proba = y_preds_proba[[y_preds_proba.columns[1]]]
            return y_preds_proba

        # Filter to matching specimen subset
        # We might have more BCR specimens than TCR specimens,
        # but for BCR+TCR metamodel, we are provided both anndatas, and should filter them to the intersection of specimens.
        # Do this here rather than after the featurizations, because having no data should not count as a model abstention.
        specimens_by_locus = {
            single_gene_locus: set(adata.obs["specimen_label"].unique())
            for single_gene_locus, adata in data.items()
        }
        specimen_intersection = set.intersection(*specimens_by_locus.values())
        for single_gene_locus in data.keys():
            original_specimen_list = specimens_by_locus[single_gene_locus]
            if original_specimen_list != specimen_intersection:
                logger.info(
                    f"Metamodel featurization with data keys {data.keys()} and gene_locus {gene_locus}: dropping specimens from {single_gene_locus} anndata: {original_specimen_list - specimen_intersection}"
                )
                data[single_gene_locus] = data[single_gene_locus][
                    data[single_gene_locus]
                    .obs["specimen_label"]
                    .isin(specimen_intersection)
                ]

        if metamodel_config.submodels is not None:
            # Call all submodel featurize()->df steps and horizontally concatenate.
            for single_gene_locus in gene_locus:
                GeneLocus.validate_single_value(single_gene_locus)
                data_for_locus = data[single_gene_locus]

                # Process all models identically since they all subclass ImmuneClassifierMixin.
                for submodel_name, clf in metamodel_config.submodels[
                    single_gene_locus
                ].items():
                    # get predictions for each specimen
                    featurized = clf.featurize(data_for_locus)
                    if featurized.X.shape[0] == 0:
                        # Edge case: If abstained on all specimens, can't make any predictions
                        specimen_probas = pd.DataFrame(columns=clf.classes_)
                    else:
                        # TODO: for RollupSequenceClassifier, write a test case that confirms that clf.predict_proba(featurized.X).index == featurized.sample_names
                        specimen_probas = pd.DataFrame(
                            clf.predict_proba(featurized.X),
                            index=featurized.sample_names,
                            columns=clf.classes_,
                        )

                    # If binary model, switch to single column, rather than including p and 1-p as features in the metamodel
                    specimen_probas = _subselect_columns_if_binary(specimen_probas)

                    # Store, renaming column names to indicate source
                    model_outputs.append(
                        specimen_probas.rename(
                            columns=lambda col: f"{single_gene_locus.name}:{submodel_name}:{col}"
                        )
                    )
                    # Store metadata object
                    metadata_objects_by_single_locus[single_gene_locus].append(
                        featurized.metadata
                    )

        ## Harmonize across models and gene loci:
        # Some models may have abstentions, and the abstentions may differ between gene loci.

        # Combine metadata objects from different models, some of which may have extra columns
        def combine_dfs(
            df1: pd.DataFrame, df2: pd.DataFrame, allow_nonequal_indexes=False
        ):
            """
            concatenate two dataframes horizontally, matching on their index, without making duplicate copies of identical columns.
            if allow_nonequal_indexes: allows non-equal indexes (i.e. some rows may only be in either df1 or df2)
            """
            index_name_left = df1.index.name
            index_name_right = df2.index.name
            if index_name_left != index_name_right:
                raise ValueError(
                    f"Indexes of left and right dataframe have different names: {index_name_left} != {index_name_right}"
                )

            temporary_index_column_name = "!INDEX!"
            if (
                temporary_index_column_name in df1.columns
                or temporary_index_column_name in df2.columns
            ):
                raise ValueError(
                    "Left or right dataframe have column name that interferes with index renaming"
                )

            merged = pd.merge(
                df1.rename_axis(
                    temporary_index_column_name, axis="index"
                ).reset_index(),
                df2.rename_axis(
                    temporary_index_column_name, axis="index"
                ).reset_index(),
                on=df1.columns.intersection(df2.columns)
                .union([temporary_index_column_name])
                .tolist(),
                how="outer" if allow_nonequal_indexes else "inner",
                validate="1:1",
            ).set_index(temporary_index_column_name)

            if not allow_nonequal_indexes and not (
                set(merged.index) == set(df1.index)
                and set(merged.index) == set(df2.index)
            ):
                raise ValueError(
                    "Index changed in merge but allow_nonequal_indexes was set to False"
                )

            merged.index.name = index_name_left  # undo rename
            return merged

        if metamodel_config.submodels is not None:
            # Combine all metadata objects into one dataframe
            # Set allow_nonequal_indexes=True because some models may abstain on some specimens
            metadata_df = functools.reduce(
                lambda accumulated, update: combine_dfs(
                    accumulated, update, allow_nonequal_indexes=True
                ),
                (
                    metadata_object
                    for metadata_objects in metadata_objects_by_single_locus.values()
                    for metadata_object in metadata_objects
                ),
            )

            # For each locus, get all specimens that were scored by any model:
            # First we will do this for each locus separately, and confirm that they are the same
            all_specimens = {
                single_gene_locus: set.union(
                    *(
                        set(metadata_object.index)
                        for metadata_object in metadata_objects
                    )
                )
                for single_gene_locus, metadata_objects in metadata_objects_by_single_locus.items()
            }
            first_value = next(iter(all_specimens.values()))
            if (
                any(specimens != first_value for specimens in all_specimens.values())
                or set(metadata_df.index) != first_value
            ):
                # Sanity check
                raise ValueError(
                    "We had different specimen lists in different gene loci"
                )
            # extract specimen list now that we confirmed they are the same
            all_specimens = first_value

            # Now get the specimens that were scored by all models in all loci:
            specimens_with_full_predictions = set.intersection(
                *(
                    set(metadata_object.index)
                    for metadata_objects in metadata_objects_by_single_locus.values()
                    for metadata_object in metadata_objects
                )
            )

            # Get abstained specimens: those scored by some but not all classifiers, in any gene locus.
            # This is the difference between specimens scored by any classifiers and specimens scored by all classifiers.
            abstained_specimen_names = list(
                all_specimens.difference(specimens_with_full_predictions)
            )
            if len(abstained_specimen_names) > 0:
                logger.info(f"Abstained specimens: {abstained_specimen_names}")
        else:
            # Handle edge case that submodels was None.
            # Extract specimen_metadata manually from each gene locus anndata and combine.
            # We expect matching specimen lists across gene loci, which is checked for us by combine_dfs(..., allow_nonequal_indexes=False)
            def _get_metadata(single_gene_locus):
                return helpers.extract_specimen_metadata_from_anndata(
                    adata=data[single_gene_locus],
                    gene_locus=single_gene_locus,
                    target_obs_column=target_obs_column,
                )

            metadata_df = functools.reduce(
                combine_dfs,
                (_get_metadata(single_gene_locus) for single_gene_locus in gene_locus),
            )
            all_specimens = set(metadata_df.index)
            specimens_with_full_predictions = all_specimens
            abstained_specimen_names = []

        # All models should have a subset of the full specimen list:
        if not all(
            set(model_predict_proba.index) <= all_specimens
            for model_predict_proba in model_outputs
        ):
            raise ValueError(
                "All model_outputs from all gene locuses should be indexed by a subset of (or full copy of) the list of specimens."
            )

        # Subset the metadata object to the subset used in all models
        abstained_specimen_metadata = metadata_df.loc[abstained_specimen_names].copy()
        specimen_metadata = metadata_df.loc[list(specimens_with_full_predictions)]

        # Combine the model predicted probabilities into a single dataframe
        # This will be the input data matrix to our blending meta-model.
        if metamodel_config.submodels is not None:
            X = pd.concat(model_outputs, axis=1)
            # Subset and reorder the predicted probabilities to match the specimen order of the reconciled metadata object
            X = X.loc[specimen_metadata.index]
        else:
            # Handle edge case that submodels was None.
            X = None

        y_col = target_obs_column.value.blended_evaluation_column_name
        if y_col is None:
            y_col = target_obs_column.value.obs_column_name

        featurized = FeaturizedData(
            X=X,
            y=specimen_metadata[y_col].values,
            sample_names=specimen_metadata.index,
            metadata=specimen_metadata,
            abstained_sample_names=abstained_specimen_names,
            abstained_sample_y=abstained_specimen_metadata[y_col].values,
            abstained_sample_metadata=abstained_specimen_metadata,
        )

        #### Post-processing after featurizing with all models for all gene loci

        if (
            metamodel_config.extra_metadata_featurizers is not None
            and len(metamodel_config.extra_metadata_featurizers) > 0
        ):
            for (
                featurizer_name,
                extra_metadata_featurizer,
            ) in metamodel_config.extra_metadata_featurizers.items():
                # Optionally add metadata features based on the compiled featurized.metadata object
                logger.info(
                    f"Adding metadata featurizer {featurizer_name} for {target_obs_column}"
                )

                # Extract variables, align by featurized_data.sample_order, verify identical shapes, and rename column names to indicate source
                # This may update the featurizer's internal state, e.g. a DemographicsFeaturizer has its one_hot_encoder_ fitted at train time
                demographics_df = (
                    extra_metadata_featurizer.featurize(featurized.metadata)
                    .loc[featurized.sample_names]
                    .rename(columns=lambda col: f"{featurizer_name}:{col}")
                )
                if featurized.X is not None:
                    assert featurized.X.shape[0] == demographics_df.shape[0]
                    featurized.X = pd.concat([featurized.X, demographics_df], axis=1)
                else:
                    # If submodels was None, meaning featurized.X starts as None.
                    featurized.X = demographics_df

                # Store updated featurizer (e.g. a DemographicsFeaturizer has its one_hot_encoder_ fitted at train time)
                metamodel_config.extra_metadata_featurizers[
                    featurizer_name
                ] = extra_metadata_featurizer

        if metamodel_config.interaction_terms is not None:
            # Optionally add interaction terms: Cartesian product of BlendingMetamodel features:
            # This should happen after the production of the features dataframe. It should happen before training the metamodel.
            # This transformation executes at train and test time.

            # We don’t want Cartesian product of all the features. We want Cartesian product between [model1,model2,model3] features and [demographics_featurizer] features.
            # We want model1_covid_pr x age, model2_covid_pr x age, and model1_covid_pr x sex, but not model1_covid_pr x model2_covid_pr or age x sex. (Though Those wouldn’t hurt?)

            # So we accept column names of what to include on either side of the outer join: a CartesianProduct(features_left, features_right) structure.
            logger.info(
                f"Adding interactions {metamodel_config.interaction_terms} for {target_obs_column}"
            )
            # Allow wildcards: Create feature list that matches interaction_terms[0] (left) and interaction_terms[1] (right)
            def expand_wildcards(one_set_of_interaction_terms: List[str]) -> List[str]:
                matching_features = []
                for i in one_set_of_interaction_terms:
                    matching_features.extend(
                        featurized.X.columns[
                            featurized.X.columns.str.contains(i)
                        ].tolist()
                    )
                # return - might have dupes
                return list(matching_features)

            left_features = expand_wildcards(metamodel_config.interaction_terms[0])
            right_features = expand_wildcards(metamodel_config.interaction_terms[1])
            featurized.X = cartesian_product(
                featurized.X, left_features, right_features
            )

        ## Optionally regress out certain metadata columns from the feature matrix.

        # Alternative considered:
        # The regress-out happens independently on each metamodel component, so we can run regress-out on the submodels instead.
        # We could wrap submodels in a RegressOutWrapper, meaning we'd initialize BlendingMetamodel with submodels=[RegressOutWrapper(RepertoireClassifier()), RegressOutWrapper(SequenceClassifier())].
        # This would ensure that regress-out is always enabled at both train and test time, or at neither time.
        # This would replace passing in regress_out_featurizers here, and will happen automatically in the component featurization step.
        # RegressOutWrapper would be initialized with two arguments: a classifier and a featurizer. RegressOutWrapper overloads the classifier’s featurize():
        # - Calls classifier’s featurize: what to regress out from. (For now we have already done this - stored at featurized.X)
        # - Then calls featurizer’s featurize: what to regress out with.
        # - Then calls the regress out. At train time, this fits the regress out. At test time, it uses the fitted regress out.
        # - Unit test that RegressOutWrapper's featurize() output shape matches inner classifier’s featurize() shape
        # Two downsides:
        # 1) Abstentions differ between models, meaning models may have inconsistent metadata objects, so the featurized covariates and resulting regress-out may operate differently on the different models.
        #   In other words, regress-out columns could have more entries in some model featurizations than in other models that abstained on some samples.
        # 2) Regressing out as a wrapper around the submodels means we'd never be able to regress out from any interaction terms.
        #   The order would be: featurize regressed-out-submodels -> concatenate horizontally -> expand with interaction terms.
        #   This would mean that Cartesian product expansion always happens after regressing out; there's no way to regress out from the new interaction term columns.
        if (
            metamodel_config.regress_out_featurizers is not None
            and len(metamodel_config.regress_out_featurizers) > 0
        ):
            regress_out_covariates_df = pd.DataFrame()
            for (
                featurizer_name,
                featurizer,
            ) in metamodel_config.regress_out_featurizers.items():
                # Collect features from the compiled featurized.metadata object
                # Extract variables, align by featurized_data.sample_order, verify identical shapes, and rename column names to indicate source
                # This may update the featurizer's internal state, e.g. a DemographicsFeaturizer has its one_hot_encoder_ fitted at train time
                regress_out_covariates_df_part = (
                    featurizer.featurize(featurized.metadata)
                    .loc[featurized.sample_names]
                    .rename(columns=lambda col: f"{featurizer_name}:{col}")
                )
                assert featurized.X.shape[0] == regress_out_covariates_df_part.shape[0]
                regress_out_covariates_df = pd.concat(
                    [regress_out_covariates_df, regress_out_covariates_df_part], axis=1
                )

                # Store updated featurizer (e.g. a DemographicsFeaturizer has its one_hot_encoder_ fitted at train time)
                metamodel_config.regress_out_featurizers[featurizer_name] = featurizer

            # Regress out demographics:
            # Preprocess the metamodel feature matrix to remove the effects of demographics variables.
            # Replace each metamodel feature marix column independently by residuals having regressed out any component correlated with age/sex/ethnicity
            # i.e. regress each column (y) on all confounders (X) together, then swap the column for y-yhat.

            logger.info(
                f"Regressing out covariates {regress_out_covariates_df.columns.tolist()} for {target_obs_column}"
            )

            # May already have passed in a regress_out_pipeline (if applying to a new dataset)
            if metamodel_config.regress_out_pipeline is None:
                # Make pipeline to regress out demographic variables (obs) from metamodel input feature matrix:
                # 1. Confirm obs variables are in same order (Puts in same order if they're not; throw error if any column missing; drop any test column not found in train)
                # 2. Scale obs (wrap in SklearnTransformerWrapper to keep pandas dataframe structure)
                # 3. Regress out this obs matrix from the feature matrix

                metamodel_config.regress_out_pipeline = make_pipeline(
                    MatchVariables(missing_values="raise"),
                    SklearnTransformerWrapper(StandardScaler()),
                    RegressOutCovariates(),
                )

                # Fit
                metamodel_config.regress_out_pipeline.fit(
                    regress_out_covariates_df, featurized.X
                )

            # Transform
            featurized.X = metamodel_config.regress_out_pipeline.predict(
                X=regress_out_covariates_df, y=featurized.X
            )

        # metamodel_config may have been modified, e.g. with a new regress_out_pipeline, or with trained featurizers
        featurized.extras["metamodel_config"] = metamodel_config
        return featurized

    @classmethod
    def convert_feature_name_to_friendly_name(cls, feature_name: str) -> str:
        featurization_map = {
            "repertoire_stats": "Repertoire composition",
            "convergent_cluster_model": "CDR3 clustering",
            "sequence_model": "Language embedding",
        }
        possible_loci = "|".join([g.name for g in GeneLocus])
        if re.match(rf"^({possible_loci}):[a-zA-Z_]*:[a-zA-Z0-9_\/]*$", feature_name):
            # e.g. BCR:repertoire_stats:Covid19
            locus, featurization, class_name = feature_name.split(":")
            return f"{featurization_map[featurization]} ({locus}): P({class_name})"

        if feature_name.startswith("demographics:"):
            # e.g. demographics:ethnicity_condensed_African
            return feature_name.replace("demographics:", "Demographics: ")

        if feature_name.startswith("interaction|"):
            # e.g. interaction|BCR:repertoire_stats:Covid19|demographics:age
            # call recursively for each part of the interaction term
            _, interaction1, interaction2 = feature_name.split("|")
            return f"Interaction: [{cls.convert_feature_name_to_friendly_name(interaction1)}] x [{cls.convert_feature_name_to_friendly_name(interaction2)}]"

        if feature_name.startswith("isotype_counts:isotype_proportion:"):
            # e.g. isotype_counts:isotype_proportion:IGHD-M
            return feature_name.replace(
                "isotype_counts:isotype_proportion:", "Isotype proportion of "
            )

        # pass through if we don't recognize
        return feature_name
