import copy
import gc
import logging
from typing import List

import numpy as np
import pytest
import scanpy as sc

from malid import (
    config,
    helpers,
    interpretation,
    io,
)
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
)
from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)
from malid.train import (
    train_repertoire_stats_model,
    train_sequence_model,
    train_metamodel,
    train_convergent_cluster_classifier,
    train_exact_matches_classifier,
    model_definitions,
)
from malid.trained_model_wrappers import (
    RepertoireClassifier,
    ConvergentClusterClassifier,
    ExactMatchesClassifier,
    BlendingMetamodel,
)
from malid.train.train_rollup_sequence_classifier import (
    generate_rollups_on_all_classification_targets,
)
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)

logger = logging.getLogger(__name__)


@pytest.fixture()
def modified_config():
    """Modify the config for end to end tests."""
    ## Copy test anndatas to temporary config.paths directory, set up in conftest.py
    for gene_locus in GeneLocus:
        for fname in (
            config.paths.tests_snapshot_dir / "scaled_anndatas_dir" / gene_locus.name
        ).glob("fold.*.h5ad"):
            # shutil.copy2(fname, config.paths.scaled_anndatas_dir)

            # Finish scaling here, to save space on disk for the test snapshots\:
            # Previously we scaled the data before storing the snapshot, but this saves space by not storing raw.

            # Technically we should scale train_smaller and then apply those scale parameters to the other fold labels,
            # but for tests that doesn't matter.
            adata = sc.read(fname)
            adata.raw = adata
            sc.pp.scale(adata)

            # To save space, we had removed these columns, but let's regenerate
            adata.obs["disease_subtype"] = adata.obs["disease"]
            adata.obs["cdr3_aa_sequence_trim_len"] = adata.obs[
                "cdr3_seq_aa_q_trim"
            ].str.len()

            # Generating these too - fake:
            adata.obs["cdr1_seq_aa_q_trim"] = adata.obs["cdr3_seq_aa_q_trim"].copy()
            adata.obs["cdr2_seq_aa_q_trim"] = adata.obs["cdr3_seq_aa_q_trim"].copy()

            destination_dir = config.paths.scaled_anndatas_dir / gene_locus.name
            destination_dir.mkdir(parents=True, exist_ok=True)
            adata.write(destination_dir / fname.name, compression="gzip")

    ## Change sequence identity thresholds
    # save original values for later
    # TODO: do we still want this?
    old_sequence_identity_thresholds = copy.copy(config.sequence_identity_thresholds)
    # modify the thresholds to be easier
    config.sequence_identity_thresholds.cluster_amino_acids_across_patients = {
        locus: 0.5 for locus in GeneLocus
    }
    config.sequence_identity_thresholds.assign_test_sequences_to_clusters = {
        locus: 0.5 for locus in GeneLocus
    }

    ## Change n_lambda used in hyperparameter-tuned linear models
    # save original value for later
    old_n_lambda = model_definitions.DEFAULT_N_LAMBDAS_FOR_TUNING
    model_definitions.DEFAULT_N_LAMBDAS_FOR_TUNING = 5

    # run tests
    yield

    # reset to original values
    config.sequence_identity_thresholds = old_sequence_identity_thresholds
    model_definitions.DEFAULT_N_LAMBDAS_FOR_TUNING = old_n_lambda


def test_first_anndata_load_does_not_affect_second_anndata_load_from_cache(
    modified_config,
):
    def load():
        return io.load_fold_embeddings(
            fold_id=0,
            fold_label="train_smaller",
            gene_locus=GeneLocus.BCR,
            target_obs_column=TargetObsColumnEnum.disease,
        )

    # first load, cache miss
    adata1 = load()
    adata1.obs = adata1.obs.assign(new_col=1)
    df1 = adata1.obs

    # second load, cache hit
    adata2 = load()

    assert "new_col" in df1.columns
    assert "new_col" in adata1.obs.columns
    assert "new_col" not in adata2.obs.columns


def test_modfiying_second_anndata_load_from_cache_does_not_affect_first_anndata_load(
    modified_config,
):
    def load():
        return io.load_fold_embeddings(
            fold_id=0,
            fold_label="train_smaller",
            gene_locus=GeneLocus.BCR,
            target_obs_column=TargetObsColumnEnum.disease,
        )

    # first load, cache miss
    adata1 = load()
    df1 = adata1.obs

    # second load, cache hit
    adata2 = load()
    adata2.obs.drop(columns=["disease"], inplace=True)

    assert "disease" in df1.columns
    assert "disease" in adata1.obs.columns
    assert "disease" not in adata2.obs.columns


def test_train_all_models(modified_config):
    training_fold_name = "train_smaller"
    validation_fold_name = "validation"
    testing_fold_name = "test"
    metamodel_fold_label_train = validation_fold_name
    metamodel_feature_prefixes = [
        "repertoire_stats",
        "convergent_cluster_model",
        "sequence_model",
    ]
    fold_ids = config.all_fold_ids

    gene_loci_used: GeneLocus = GeneLocus.BCR | GeneLocus.TCR
    # TODO add rest of TargetObsColumnEnum. The snapshot datasets don't have enough coverage of these classes to test them.
    target_obs_columns = [
        TargetObsColumnEnum.disease,
        TargetObsColumnEnum.disease_all_demographics_present,
        # TargetObsColumnEnum.age_group_binary_healthy_only,
        # TargetObsColumnEnum.covid_vs_healthy,
    ]
    expected_classes_by_target = {
        TargetObsColumnEnum.disease: ["Covid19", "HIV", "Healthy/Background"],
        TargetObsColumnEnum.disease_all_demographics_present: [
            "Covid19",
            "HIV",
            "Healthy/Background",
        ],
        # TargetObsColumnEnum.age_group_binary_healthy_only: ["50+", "under 50"],
        # TargetObsColumnEnum.covid_vs_healthy: ["Covid19", "Healthy/Background"],
    }

    # TODO: assert config.paths is what we expect
    print(config.paths)

    def _run_metamodel(
        gene_locus: GeneLocus, target_obs_column: TargetObsColumnEnum, fold_id: int
    ):
        for (
            metamodel_flavor,
            metamodel_config,
        ) in train_metamodel.get_metamodel_flavors(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            fold_id=fold_id,
            base_model_train_fold_name=training_fold_name,
        ).items():
            logger.info(
                f"Training metamodel for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_column}, metamodel flavor {metamodel_flavor}: {metamodel_config}"
            )
            # metamodel_config may change during training, e.g. regress_out_pipeline might get set.
            # we make sure to update to the latest metamodel_config to load in the metamodel correctly later.
            (
                experiment_set,
                metamodel_config,
            ) = train_metamodel.run_classify_with_all_models(
                fold_id=fold_id,
                gene_locus=gene_locus,
                target_obs_column=target_obs_column,
                metamodel_flavor=metamodel_flavor,
                metamodel_config=metamodel_config,
                base_model_train_fold_name=training_fold_name,
                metamodel_fold_label_train=metamodel_fold_label_train,
                # Disable evaluation because global fold doesn't have a test set
                metamodel_fold_label_test=(
                    testing_fold_name if fold_id != -1 else None
                ),
                chosen_models=["lasso_cv", "xgboost"],
                n_jobs=1,
                # control fold_id and cache manually so that we limit repetitive I/O
                clear_cache=False,
                fail_on_error=True,  # this is for CI only
            )

            # Confirm feature names.
            # First, generate the expected feature names, depending on the metamodel flavor.

            disease_features = []
            if not metamodel_flavor.startswith("demographics_only"):
                # Add disease features, unless we have a demographics-only metamodel.
                expected_classes: List[str] = expected_classes_by_target[
                    target_obs_column
                ]
                if len(expected_classes) == 2:
                    # Binary metamodels only include a feature for one of the classes:
                    # e.g. Covid-vs-healthy metamodel only includes BCR:repertoire_stats:Healthy/Background, BCR:convergent_cluster_model:Healthy/Background, etc.
                    # but does not include BCR:repertoire_stats:Covid19, BCR:convergent_cluster_model:Covid19, etc.

                    # Still keep this as a list, so we can iterate over it
                    expected_classes = [expected_classes[1]]

                for single_gene_locus in gene_locus:  # might be composite
                    for feature_prefix in metamodel_feature_prefixes:
                        for disease in expected_classes:
                            # this generates e.g.:
                            # [
                            #     "BCR:repertoire_stats:Covid19",
                            #     "BCR:repertoire_stats:HIV",
                            #     "BCR:repertoire_stats:Healthy/Background",
                            #     "BCR:convergent_cluster_model:Covid19",
                            #     "BCR:convergent_cluster_model:HIV",
                            #     "BCR:convergent_cluster_model:Healthy/Background",
                            #     "BCR:sequence_model:Covid19",
                            #     "BCR:sequence_model:HIV",
                            #     "BCR:sequence_model:Healthy/Background",
                            # ]
                            disease_features.append(
                                f"{single_gene_locus.name}:{feature_prefix}:{disease}"
                            )

            demographics_features = []
            if (
                metamodel_flavor == "with_demographics_columns"
                or metamodel_flavor.startswith("demographics_only")
            ):
                if metamodel_flavor in [
                    "with_demographics_columns",
                    "demographics_only",
                    "demographics_only_age",
                ]:
                    demographics_features.append("demographics:age")

                # Metamodel features also depend on OneHotEncoder choices, which are determined by:

                # a) the order of binary demographic column values in the metamodel's training set, like "sex":
                if metamodel_flavor in [
                    "with_demographics_columns",
                    "demographics_only",
                    "demographics_only_sex",
                ]:
                    one_hot_encoder_chosen_sex = (
                        metamodel_config.extra_metadata_featurizers[
                            "demographics"
                        ].one_hot_encoder_.encoder_dict_["sex"][0]
                    )
                    # assert one_hot_encoder_chosen_sex in [
                    #     "F",
                    #     "M",
                    # ], f"Expected one hot encoder to choose between 'F' and 'M', but got {one_hot_encoder_chosen_sex}"
                    # Update: we now force a specific choice here by setting sex to have an ordered categorical dtype.
                    # Confirm that the one hot encoder picks that up:
                    assert one_hot_encoder_chosen_sex == "F"

                    demographics_features.append(
                        f"demographics:sex_{one_hot_encoder_chosen_sex}"
                    )

                # b) which samples are abstained on - some ethnicity values may disappear
                if metamodel_flavor in [
                    "with_demographics_columns",
                    "demographics_only",
                    "demographics_only_ethnicity_condensed",
                ]:
                    one_hot_encoder_chosen_ethnicities = (
                        metamodel_config.extra_metadata_featurizers[
                            "demographics"
                        ].one_hot_encoder_.encoder_dict_["ethnicity_condensed"]
                    )
                    possible_ethnicities = [
                        "African",
                        "Asian",
                        "Caucasian",
                        "Hispanic/Latino",
                    ]
                    assert set(one_hot_encoder_chosen_ethnicities) <= set(
                        possible_ethnicities
                    ), f"One hot encoder chose ethnicites {one_hot_encoder_chosen_ethnicities}, which is not a subset of our dataset's possible ethnicities {possible_ethnicities}"

                    demographics_features.extend(
                        [
                            f"demographics:ethnicity_condensed_{ethnicity}"
                            for ethnicity in one_hot_encoder_chosen_ethnicities
                        ]
                    )

            interaction_terms = []
            if metamodel_flavor == "with_demographics_columns":
                # Also add interaction terms, e.g.:

                # interaction|BCR:repertoire_stats:Covid19|demographics:age
                # interaction|BCR:repertoire_stats:Covid19|demographics:ethnicity_condensed_African
                # interaction|BCR:repertoire_stats:Covid19|demographics:ethnicity_condensed_Hispanic/Latino
                # interaction|BCR:repertoire_stats:Covid19|demographics:ethnicity_condensed_Caucasian
                # interaction|BCR:repertoire_stats:Covid19|demographics:sex_M
                # interaction|BCR:repertoire_stats:Covid19|demographics:ethnicity_condensed_Asian
                # interaction|BCR:convergent_cluster_model:Covid19|demographics:age
                # interaction|BCR:convergent_cluster_model:Covid19|demographics:ethnicity_condensed_African
                # interaction|BCR:convergent_cluster_model:Covid19|demographics:ethnicity_condensed_Hispanic/Latino
                # [...]

                interaction_terms = [
                    f"interaction|{disease_feature}|{demographic_feature}"
                    for disease_feature in disease_features
                    for demographic_feature in demographics_features
                ]

            # all expected feature names
            expected_feature_names = (
                disease_features + demographics_features + interaction_terms
            )

            if metamodel_flavor == "isotype_counts_only":
                # special case: hijack and replace
                if (
                    gene_locus == GeneLocus.BCR
                    or gene_locus == GeneLocus.BCR | GeneLocus.TCR
                ):
                    # note: no entry for TCR
                    expected_feature_names = [
                        "isotype_counts:isotype_proportion:IGHG",
                        "isotype_counts:isotype_proportion:IGHA",
                        "isotype_counts:isotype_proportion:IGHD-M",
                    ]
                else:
                    raise ValueError(
                        f"Should not have isotype_counts_only metamodel flavor for {gene_locus}"
                    )

            # Load trained metamodel, two ways.
            for metamodel_name in ["lasso_cv", "xgboost"]:
                for clf in [
                    BlendingMetamodel(
                        fold_id=fold_id,
                        metamodel_name=metamodel_name,
                        base_model_train_fold_name=training_fold_name,
                        metamodel_fold_label_train=metamodel_fold_label_train,
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_column,
                        metamodel_config=metamodel_config,
                        metamodel_base_dir=BlendingMetamodel._get_metamodel_base_dir(
                            gene_locus=gene_locus,
                            target_obs_column=target_obs_column,
                            metamodel_flavor=metamodel_flavor,
                        ),
                    ),
                    BlendingMetamodel.from_disk(
                        fold_id=fold_id,
                        metamodel_name=metamodel_name,
                        base_model_train_fold_name=training_fold_name,
                        metamodel_fold_label_train=metamodel_fold_label_train,
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_column,
                        metamodel_flavor=metamodel_flavor,
                    ),
                ]:
                    # Confirm feature names match
                    observed_metamodel_feature_names = clf.feature_names_in_
                    nl = "\n"  # can't use backslash in fstring
                    assert np.array_equal(
                        observed_metamodel_feature_names, expected_feature_names
                    ), f"""
    Expected metamodel flavor {metamodel_flavor} ({target_obs_column}, {gene_locus}) feature names to be:
{nl.join(expected_feature_names)},
    but got:
{nl.join(observed_metamodel_feature_names)}"""

                    # TODO: run predictions with these models - based on supervised_embedding code - to check that we can load them from disk

    for gene_locus in gene_loci_used:
        # Control fold_id and cache manually so that we limit repetitive I/O
        for fold_id in fold_ids:
            for target_obs_col in target_obs_columns:
                # Train model 1
                logger.info(
                    f"Training model 1 for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_col}."
                )
                train_repertoire_stats_model.run_classify_with_all_models(
                    gene_locus=gene_locus,
                    fold_label_train=training_fold_name,
                    fold_label_test=validation_fold_name,
                    chosen_models=[
                        config.metamodel_base_model_names.model_name_overall_repertoire_composition,
                    ],
                    n_jobs=1,
                    target_obs_column=target_obs_col,
                    # control fold_id and cache manually so that we limit repetitive I/O
                    fold_ids=[fold_id],
                    clear_cache=False,
                    fail_on_error=True,  # this is for CI only
                )

                logger.info(
                    f"Tuning model 1 on validation set for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_col}."
                )
                # Load model 1
                clf = RepertoireClassifier(
                    fold_id=fold_id,
                    model_name=config.metamodel_base_model_names.model_name_overall_repertoire_composition,
                    fold_label_train="train_smaller",
                    target_obs_column=target_obs_col,
                    gene_locus=gene_locus,
                )
                # TODO: expose final pipeline step's feature_names_in_ as a property
                feature_names = clf._inner[:-1].get_feature_names_out()
                feature_names = [s for s in feature_names if "pca" in s.lower()]
                # expect f"{isotype_group}:pca_{n_pc}" from 1 to n_pcs+1 for each isotype_group, then other features from obs
                # n_pcs_effective doesn't always equal RepertoireClassifier.n_pcs (see train_repertoire_stats_model), so use inequalities
                assert (
                    2
                    <= len(feature_names)
                    <= RepertoireClassifier.n_pcs
                    * len(helpers.isotype_groups_kept[gene_locus])
                )

                # Tune on validation set
                clf_tuned = clf.tune_model_decision_thresholds_to_validation_set(
                    validation_set=None
                )

                # Type checks
                assert isinstance(clf, ImmuneClassifierMixin)
                assert isinstance(clf, RepertoireClassifier)
                assert not isinstance(clf, AdjustedProbabilitiesDerivedModel)

                assert not isinstance(clf_tuned, ImmuneClassifierMixin)
                assert not isinstance(clf_tuned, RepertoireClassifier)
                assert isinstance(clf_tuned, AdjustedProbabilitiesDerivedModel)

                for one_clf in [clf, clf_tuned]:
                    # Confirm clf and clf_tuned have the right classes_
                    assert np.array_equal(
                        one_clf.classes_, expected_classes_by_target[target_obs_col]
                    )

                ##########

                # Train model 2
                logger.info(
                    f"Training model 2 for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_col}."
                )
                train_convergent_cluster_classifier.run_classify_with_all_models(
                    gene_locus=gene_locus,
                    fold_label_train=training_fold_name,
                    fold_label_test=validation_fold_name,
                    chosen_models=[
                        config.metamodel_base_model_names.model_name_convergent_clustering,
                    ],
                    n_jobs=2,  # this parameter controls p-value thresholding tuning parallelization
                    target_obs_column=target_obs_col,
                    # control fold_id and cache manually so that we limit repetitive I/O
                    fold_ids=[fold_id],
                    clear_cache=False,
                    # reduced list of p-values for CI
                    # all are deliberately high to reduce potential for lots of abstentions on our tiny snapshot test dataset,
                    # which could cause tests to fail due to the resulting model training sets being too small / not having all classes included
                    p_values=[0.5, 0.8],
                    fail_on_error=True,  # this is for CI only
                )

                logger.info(
                    f"Tuning model 2 on validation set for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_col}."
                )
                # Load model 2
                clf = ConvergentClusterClassifier(
                    fold_id=fold_id,
                    model_name=config.metamodel_base_model_names.model_name_convergent_clustering,
                    fold_label_train="train_smaller",
                    target_obs_column=target_obs_col,
                    gene_locus=gene_locus,
                )
                assert np.array_equal(
                    clf.feature_names_in_, expected_classes_by_target[target_obs_col]
                )
                # Tune on validation set
                clf_tuned = clf.tune_model_decision_thresholds_to_validation_set(
                    validation_set=None
                )

                # Type checks
                assert isinstance(clf, ImmuneClassifierMixin)
                assert isinstance(clf, ConvergentClusterClassifier)
                assert not isinstance(clf, AdjustedProbabilitiesDerivedModel)

                assert not isinstance(clf_tuned, ImmuneClassifierMixin)
                assert not isinstance(clf_tuned, ConvergentClusterClassifier)
                assert isinstance(clf_tuned, AdjustedProbabilitiesDerivedModel)

                for one_clf in [clf, clf_tuned]:
                    # Confirm clf and clf_tuned have the right feature_names_in_
                    assert np.array_equal(
                        one_clf.feature_names_in_,
                        expected_classes_by_target[target_obs_col],
                    )
                    # Confirm clf and clf_tuned have the right classes_
                    assert np.array_equal(
                        one_clf.classes_, expected_classes_by_target[target_obs_col]
                    )

                ##########

                # Train exact matches benchmark
                logger.info(
                    f"Training exact matches benchmark for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_col}."
                )
                train_exact_matches_classifier.run_classify_with_all_models(
                    gene_locus=gene_locus,
                    fold_label_train=training_fold_name,
                    # for CI, we don't do p-value cross validation on validation fold,
                    # because we did not construct our snapshot datasets to have many repeated sequences
                    fold_label_test=training_fold_name,  # validation_fold_name,
                    chosen_models=[
                        config.metamodel_base_model_names.model_name_convergent_clustering,
                    ],
                    n_jobs=2,  # this parameter controls p-value thresholding tuning parallelization
                    target_obs_column=target_obs_col,
                    # control fold_id and cache manually so that we limit repetitive I/O
                    fold_ids=[fold_id],
                    clear_cache=False,
                    p_values=[0.1, 0.5],  # reduced list for CI
                    fail_on_error=True,  # this is for CI only
                )

                # Load exact matches model
                clf = ExactMatchesClassifier(
                    fold_id=fold_id,
                    model_name=config.metamodel_base_model_names.model_name_convergent_clustering,
                    fold_label_train="train_smaller",
                    target_obs_column=target_obs_col,
                    gene_locus=gene_locus,
                )

                # Type checks
                assert isinstance(clf, ImmuneClassifierMixin)
                assert isinstance(clf, ExactMatchesClassifier)

                # Confirm clf has the right feature_names_in_
                assert np.array_equal(
                    clf.feature_names_in_,
                    expected_classes_by_target[target_obs_col],
                )
                # Confirm clf has the right classes_
                assert np.array_equal(
                    clf.classes_, expected_classes_by_target[target_obs_col]
                )

                ##########

                # Train model 3
                sample_weight_strategy = SampleWeightStrategy.ISOTYPE_USAGE
                logger.info(
                    f"Training model 3 for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_col}, sample_weight_strategy={sample_weight_strategy}."
                )
                train_sequence_model.run_classify_with_all_models(
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_col,
                    sample_weight_strategy=sample_weight_strategy,
                    fold_label_train=training_fold_name,
                    fold_label_test=validation_fold_name,
                    chosen_models=[
                        config.metamodel_base_model_names.model_name_sequence_disease,
                    ],
                    n_jobs=1,
                    # control fold_id and cache manually so that we limit repetitive I/O
                    fold_ids=[fold_id],
                    clear_cache=False,
                    fail_on_error=True,  # this is for CI only
                )

                # Model 3 rollup:
                # Theoretically we don't need to run this model3 rollup before training metamodel,
                # because metamodel will generate the rollup from scratch (rather than loading from disk),
                # but let's do it anyway because the rollup + tune is an important part of the project.
                logger.info(
                    f"Rolling up model 3 (+ tuning rollup decision thresholds) for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_col}, sample_weight_strategy={sample_weight_strategy}."
                )
                generate_rollups_on_all_classification_targets(
                    fold_ids=[fold_id],  # range(config.n_folds),
                    targets=[
                        (target_obs_col, SampleWeightStrategy.ISOTYPE_USAGE)
                        # (target_obs_column, SampleWeightStrategy.ISOTYPE_USAGE)
                        # for target_obs_column in config.classification_targets
                    ],
                    gene_locus=gene_locus,
                    fold_label_test=testing_fold_name,
                    chosen_models=[
                        config.metamodel_base_model_names.model_name_sequence_disease,
                    ],
                    fold_label_train="train_smaller",
                    also_tune_decision_thresholds=True,
                    clear_cache=False,
                    fail_on_error=True,  # this is for CI only
                )

                # TODO: Load model3 rollup and rollup_tuned
                # TODO: confirm both clf and clf_tuned have the right classes_

                ##########

                # Metamodel for single gene locus
                _run_metamodel(
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_col,
                    fold_id=fold_id,
                )

                ##########

                # Miscellaneous checks:

                # Confirm that the loaded "sex" binary metadata column has an ordered categorical dtype.
                # This is relied on in DemographicsFeaturizer's one hot encoding to ensure a consistent feature name choice for all folds.
                adata = io.load_fold_embeddings(
                    fold_id=fold_id,
                    fold_label=training_fold_name,
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_col,
                    sample_weight_strategy=sample_weight_strategy,
                )
                assert adata.obs[
                    "sex"
                ].cat.ordered, f"Expected sex to have ordered categorical dtype; instead got {adata.obs['sex'].dtype}"

            # manage fold ID and cache manually:
            # now that we are done with all TargetObsColumns from this fold,
            # clear cache
            io.clear_cached_fold_embeddings()
            gc.collect()

        # For this gene locus, run model interpretations
        logger.info(
            f"Running sequence model interpretations for gene_locus={gene_locus}, target_obs_columns={target_obs_columns}."
        )
        interpretation.rank_entire_locus_sequences(
            gene_locus=gene_locus,
            target_obs_columns=target_obs_columns,
            main_output_base_dir=(
                config.paths.model_interpretations_output_dir / gene_locus.name
            ),
            highres_output_base_dir=(
                config.paths.high_res_outputs_dir
                / "model_interpretations"
                / gene_locus.name
            ),
            cdr3_size_threshold=5,  # Lower because our simulated dataset is very sparse
        )

    # Metamodel with all gene locuses together:
    if len(gene_loci_used) > 1:
        # Control fold_id and cache manually so that we limit repetitive I/O
        for fold_id in fold_ids:
            for target_obs_col in target_obs_columns:
                _run_metamodel(
                    gene_locus=gene_loci_used,
                    target_obs_column=target_obs_col,
                    fold_id=fold_id,
                )

            # manage fold ID and cache manually:
            # now that we are done with all TargetObsColumns from this fold,
            # clear cache
            io.clear_cached_fold_embeddings()
            gc.collect()
