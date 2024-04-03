from pathlib import Path
import numpy as np
import anndata
import pandas as pd

from malid import config
from malid.datamodels import (
    GeneLocus,
    TargetObsColumnEnum,
    combine_classification_option_names,
)
from malid.trained_model_wrappers import (
    VJGeneSpecificSequenceClassifier,
    VGeneSpecificSequenceClassifier,
    RollupSequenceClassifier,
    AbstractSequenceClassifier,
    SequenceClassifier,
)
import pytest

from malid.trained_model_wrappers.vj_gene_specific_sequence_classifier import (
    SequenceSubsetStrategy,
)

classes = np.array(["disease1", "disease2"])


class MockSequenceModel(AbstractSequenceClassifier):
    def __init__(self, classes_, *args, **kwargs):
        # Don't call superclass
        self.classes_ = classes_

    def featurize(self, dataset: anndata.AnnData):
        raise NotImplementedError("Should not be called")

    def predict_proba(self, X: np.ndarray):
        return np.random.rand(X.shape[0], len(self.classes_))

    @staticmethod
    def _get_model_base_dir(*args, **kwargs):
        return SequenceClassifier._get_model_base_dir(*args, **kwargs)

    @staticmethod
    def _get_output_base_dir(*args, **kwargs):
        return SequenceClassifier._get_output_base_dir(*args, **kwargs)


@pytest.fixture
def adata():
    ad = anndata.AnnData(
        X=np.random.randn(4, 10),
        obs=pd.DataFrame(
            {
                "v_gene": ["IGHV1-24", "IGHV1-24", "IGHV1-18", "IGHV1-18"],
                "j_gene": ["IGHJ6", "IGHJ6", "IGHJ4", "IGHJ4"],
                "specimen_label": ["specimen1", "specimen1", "specimen1", "specimen2"],
                "disease": ["disease1", "disease1", "disease1", "disease2"],
                "isotype_supergroup": "IGHG",
                "sample_weight_isotype_rebalance": 0.8,
                # generate v_mut between 0 and 0.2 for BCR (if TCR, should always be 0)
                "v_mut": np.random.uniform(low=0, high=0.2, size=4),
            },
            index=["cell1", "cell2", "cell3", "cell4"],
        ),
    )
    # Set raw to be the same as X. To train the submodels, we return to the raw data, and then standardize each subset independently.
    ad.raw = ad
    return ad


def test_vj_gene_specific_sequence_classifier(adata):
    # Data construction:
    # - The first two sequences have a V/J gene combination for which a model is available, while the third and fourth sequences do not.
    # - The first specimen has sequences that will be scored, but the second specimen has no sequences that will be scored at all.
    models = {
        ("IGHV1-24", "IGHJ6"): MockSequenceModel(classes),
    }

    # Make sequence classifier from these submodels
    clf = VJGeneSpecificSequenceClassifier(
        fold_id=1,
        # In this test, we don't care about the model name,
        # but generally we will set this to the name of the model class that is used.
        model_name_sequence_disease="elasticnet_cv",
        fold_label_train="train_smaller1",
        gene_locus=GeneLocus.BCR,
        target_obs_column=TargetObsColumnEnum.disease,
        sample_weight_strategy=config.sample_weight_strategy,
        models=models,
        classes=classes,
    )
    # Make rollup (speicmen-level) classifier
    rollup_clf = RollupSequenceClassifier(
        fold_id=1,
        # In this test, we don't care about the model name,
        # but generally we will set this to the name of the model class that is used.
        model_name_sequence_disease="elasticnet_cv",
        fold_label_train="train_smaller1",
        gene_locus=GeneLocus.BCR,
        target_obs_column=TargetObsColumnEnum.disease,
        sample_weight_strategy=config.sample_weight_strategy,
        sequence_classifier=clf,
    )
    assert np.array_equal(clf.classes_, classes)
    assert np.array_equal(rollup_clf.classes_, classes)

    # Featurize with sequence classifier
    featurized = clf.featurize(adata)
    assert isinstance(featurized.X, anndata.AnnData)
    assert featurized.X.shape == (4, 10)
    assert (
        featurized.X.shape[0]
        == featurized.y.shape[0]
        == featurized.sample_names.shape[0]
        == featurized.sample_weights.shape[0]
        == 4
    )
    assert np.array_equal(featurized.X.obs_names, featurized.sample_names)
    assert np.array_equal(featurized.X.obs_names, ["cell1", "cell2", "cell3", "cell4"])
    assert featurized.metadata.equals(adata.obs)

    # Predict with sequence classifier - we expect two abstentions
    sequence_predictions = clf.predict_proba(featurized.X)
    sequence_predicted_labels = clf.predict(featurized.X)
    assert isinstance(sequence_predictions, pd.DataFrame)
    assert isinstance(sequence_predicted_labels, pd.Series)
    assert sequence_predictions.shape == (featurized.X.shape[0], len(classes))
    assert sequence_predicted_labels.shape == (featurized.X.shape[0],)
    assert np.array_equal(sequence_predictions.index, featurized.sample_names)
    assert np.array_equal(sequence_predicted_labels.index, featurized.sample_names)
    assert np.array_equal(sequence_predictions.columns, classes)
    assert (
        not sequence_predictions.iloc[:2].isna().any().any()
    ), "First two sequences should have predictions"
    assert (
        sequence_predictions.iloc[2:].isna().all().all()
    ), "Last two sequences should NOT have predictions"
    assert (
        not sequence_predicted_labels.iloc[:2].isna().any()
    ), "First two sequences should have predicted labels"
    assert (
        sequence_predicted_labels.iloc[2:].isna().all()
    ), "Last two sequences should NOT have predicted labels"

    # Run rollup classifier
    rollup_predictions = rollup_clf.predict_proba(
        adata, proportiontocut=0.0, strategy="trimmed_mean"
    )
    assert np.array_equal(rollup_predictions.columns, classes)
    assert np.array_equal(
        rollup_predictions.index, ["specimen1", "specimen2"]
    ), "Both specimens should have rows"
    assert not rollup_predictions.loc["specimen1"].isna().any()
    assert (
        rollup_predictions.isna()
        .all(axis=1)
        .equals(pd.Series({"specimen1": False, "specimen2": True}))
    ), "All rollup predictions should be present for specimen1 and NaN for specimen2"

    expected_path = (
        Path(GeneLocus.BCR.name)
        / combine_classification_option_names(
            target_obs_column=TargetObsColumnEnum.disease,
            sample_weight_strategy=config.sample_weight_strategy,
        )
        / "vj_gene_specific_models"
        / "rollup_models"
    )
    assert str(expected_path) in str(rollup_clf.output_base_dir)


def test_submodel_with_fewer_classes(adata):
    # Test what happens when some sub-models have fewer classes
    models = {
        ("IGHV1-24", "IGHJ6"): MockSequenceModel(classes),
        # This one has fewer classes
        ("IGHV1-18", "IGHJ4"): MockSequenceModel(classes[:-1]),
    }
    # Make sequence classifier from these submodels
    clf = VJGeneSpecificSequenceClassifier(
        fold_id=1,
        # In this test, we don't care about the model name,
        # but generally we will set this to the name of the model class that is used.
        model_name_sequence_disease="elasticnet_cv",
        fold_label_train="train_smaller1",
        gene_locus=GeneLocus.BCR,
        target_obs_column=TargetObsColumnEnum.disease,
        sample_weight_strategy=config.sample_weight_strategy,
        models=models,
        classes=classes,
    )
    assert np.array_equal(clf.classes_, classes)

    featurized = clf.featurize(adata)
    sequence_predictions = clf.predict_proba(featurized.X)
    # Missing classes should be filled with 0s
    assert sequence_predictions.shape == (featurized.X.shape[0], len(classes))
    assert np.array_equal(sequence_predictions.index, featurized.sample_names)
    assert np.array_equal(sequence_predictions.columns, classes)
    assert not sequence_predictions.isna().any().any()
    assert (
        sequence_predictions.loc[adata.obs["v_gene"] == "IGHV1-18"][classes[-1]] == 0
    ).all()


def test_split_on_vgene_only_has_appropriate_file_path():
    # Mimic test_vj_gene_specific_sequence_classifier():
    models = {
        "IGHV1-24": MockSequenceModel(classes),
    }

    # Make sequence classifier from these submodels
    clf = VGeneSpecificSequenceClassifier(
        fold_id=1,
        # In this test, we don't care about the model name,
        # but generally we will set this to the name of the model class that is used.
        model_name_sequence_disease="elasticnet_cv",
        fold_label_train="train_smaller1",
        gene_locus=GeneLocus.BCR,
        target_obs_column=TargetObsColumnEnum.disease,
        sample_weight_strategy=config.sample_weight_strategy,
        models=models,
        classes=classes,
    )
    # Make rollup (speicmen-level) classifier
    rollup_clf = RollupSequenceClassifier(
        fold_id=1,
        # In this test, we don't care about the model name,
        # but generally we will set this to the name of the model class that is used.
        model_name_sequence_disease="elasticnet_cv",
        fold_label_train="train_smaller1",
        gene_locus=GeneLocus.BCR,
        target_obs_column=TargetObsColumnEnum.disease,
        sample_weight_strategy=config.sample_weight_strategy,
        sequence_classifier=clf,
    )
    assert np.array_equal(clf.classes_, classes)
    assert np.array_equal(rollup_clf.classes_, classes)

    assert "vj_gene_specific_models" not in str(clf.models_base_dir)
    assert "v_gene_specific_models" in str(clf.models_base_dir)

    assert "vj_gene_specific_models" not in str(clf.output_base_dir)
    assert "v_gene_specific_models" in str(clf.output_base_dir)

    assert "vj_gene_specific_models" not in str(rollup_clf.models_base_dir)
    assert "v_gene_specific_models" in str(rollup_clf.models_base_dir)

    assert "vj_gene_specific_models" not in str(rollup_clf.output_base_dir)
    assert "v_gene_specific_models" in str(rollup_clf.output_base_dir)


@pytest.mark.parametrize("possible_value", list(SequenceSubsetStrategy))
def test_sequence_subset_strategy(possible_value: SequenceSubsetStrategy):
    SequenceSubsetStrategy.validate(possible_value)
    # Confirm that every legitimate enum value is supported by base_model:
    # you can access base_model for each enum value without hitting an error
    assert possible_value.base_model is not None
    assert issubclass(possible_value.base_model, VJGeneSpecificSequenceClassifier)
