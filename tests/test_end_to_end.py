import copy
import gc
import logging
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import anndata
import pytest
import scanpy as sc

from malid import (
    config,
    helpers,
    io,
)
from malid.datamodels import (
    GeneLocus,
    SampleWeightStrategy,
    TargetObsColumnEnum,
    healthy_label,
)
from malid.external.adjust_model_decision_thresholds import (
    AdjustedProbabilitiesDerivedModel,
)
from malid.train import (
    train_repertoire_stats_model,
    train_vj_gene_specific_sequence_model,
    train_vj_gene_specific_sequence_model_rollup,
    train_metamodel,
    train_convergent_cluster_classifier,
    train_exact_matches_classifier,
    model_definitions,
)
from malid.trained_model_wrappers import (
    SequenceSubsetStrategy,
    RepertoireClassifier,
    ConvergentClusterClassifier,
    ExactMatchesClassifier,
    BlendingMetamodel,
    VJGeneSpecificSequenceModelRollupClassifier,
)
from malid.trained_model_wrappers.immune_classifier_mixin import (
    ImmuneClassifierMixin,
)
from malid.trained_model_wrappers.vj_gene_specific_sequence_model_rollup_classifier import (
    AggregationStrategy,
)

logger = logging.getLogger(__name__)
sample_weight_strategy = (
    SampleWeightStrategy.ISOTYPE_USAGE  # | SampleWeightStrategy.CLONE_SIZE
    # TODO: Fix clone_size. Values in weight column created from num_clone_members all very small, and don't sum to 1.
)


@pytest.fixture()
def modified_config():
    """Modify the config for end to end tests."""
    ## Create test anndatas in temporary config.paths directory, set up in conftest.py

    # Use these three diseases because they reliably have 20 or more specimens in each split of all folds.
    # (A decent number of specimens is important because we will do nested/internal cross validation when training the Glmnet models, for example.)
    diseases = ["Lupus", "HIV", healthy_label]
    assert set(diseases) <= set(helpers.diseases)

    # Generate samples from a multivariate Gaussian distribution for each class.
    def define_class_parameters(classes: List[str], x_range=(0, 10), y_range=(0, 10)):
        x_coords = np.random.uniform(x_range[0], x_range[1], len(classes))
        y_coords = np.random.uniform(y_range[0], y_range[1], len(classes))
        centers = list(zip(x_coords, y_coords))

        class_params = {}

        for i, class_name in enumerate(classes):
            mu = centers[i]
            # cov = np.array([
            #     [np.random.uniform(0.5, 2), np.random.uniform(-0.5, 0.5)],
            #     [np.random.uniform(-0.5, 0.5), np.random.uniform(0.5, 2)]
            # ])
            # Generate a valid covariance matrix: must be symmetric and positive-semidefinite.
            # you can achieve that from any matrix by taking outer product with itself:
            # random_matrix = np.random.rand(2, 2) # or smaller values for tighter clusters: np.random.uniform(0.1, 0.3, size=(2, 2))
            # cov = np.dot(random_matrix, random_matrix.T)

            # Update: Generate a near-diagonal covariance matrix for circular blobs
            spread = np.random.uniform(0.1, 0.3)
            cov = np.identity(2) * spread

            class_params[class_name] = {"mu": mu, "cov": cov}

        return class_params

    def generate_samples(class_params: dict, class_label: str, n_samples: int):
        mu = class_params[class_label]["mu"]
        cov = class_params[class_label]["cov"]
        return np.random.multivariate_normal(mu, cov, n_samples)

    # Define the class distributions up-front, then sample repeatedly from them later.
    # Generate these patterns upfront, so they're shared across subsets of the same fold (but actually currently shared across all folds and loci)
    np.random.seed(42)
    disease_gaussian_mixture_params = define_class_parameters(diseases)

    for gene_locus in GeneLocus:
        # Generate these patterns upfront, so they're shared across subsets of the same fold (but actually also shared across all folds)
        amino_acid_alphabet: List[str] = list("ACDEFGHIKLMNPQRSTVWY")
        common_disease_specific_sequence = {
            disease: "".join(
                np.random.choice(
                    amino_acid_alphabet, size=np.random.randint(8, 12), replace=True
                )
            )
            for disease in diseases
        }
        # Associate a specific set of V genes with each disease and healthy.
        # (Note that model 1 will remove the bottom half of rare V genes)
        disease_associated_v_genes = {
            disease: helpers.all_observed_v_genes()[gene_locus][ix * 10 : ix * 10 + 10]
            for ix, disease in enumerate(diseases)
        }
        assert all(len(lst) == 10 for lst in disease_associated_v_genes.values())
        available_j_genes = helpers.all_observed_j_genes()[gene_locus][:3]

        # Precompute all V-J gene combinations (see below for usage)
        all_v_genes = set().union(*disease_associated_v_genes.values())  # set union
        all_vj_combinations = [(v, j) for v in all_v_genes for j in available_j_genes]

        for fold_id in config.all_fold_ids:
            for fold_label in ["train_smaller", "validation", "test"]:
                if fold_id == -1 and fold_label == "test":
                    # skip global fold test set: does not exist
                    continue

                all_X = []
                all_obs = []
                for disease in diseases:
                    n_specimens = np.random.randint(15, 21)
                    n_sequences = np.random.randint(4000, 5000)

                    X = generate_samples(
                        class_params=disease_gaussian_mixture_params,
                        class_label=disease,
                        n_samples=n_sequences,
                    )
                    all_X.append(X)

                    # Use real specimen labels associated with this disease, so all metadata loading works properly
                    specimen_labels = helpers.get_all_specimen_cv_fold_info()
                    specimen_labels = (
                        specimen_labels[
                            (specimen_labels["fold_id"] == int(fold_id))
                            & (specimen_labels["fold_label"] == fold_label)
                            & (specimen_labels["disease"] == disease)
                        ][["specimen_label", "participant_label"]]
                        .drop_duplicates()
                        .iloc[:n_specimens]
                    )

                    # Make obs dataframe with columns ["specimen_label", "participant_label"]
                    obs = specimen_labels.sample(n_sequences, replace=True).reset_index(
                        drop=True
                    )
                    obs["disease"] = disease
                    obs["amplification_label"] = obs["specimen_label"]  # for simplicity

                    # Set other obs columns
                    # Note on this np.random.choice pattern: designed so that when disease == healthy_label, values to choose from are identical
                    obs["cdr3_seq_aa_q_trim"] = np.random.choice(
                        [
                            common_disease_specific_sequence[healthy_label],
                            common_disease_specific_sequence[disease],
                        ],
                        p=[0.1, 0.9],
                        size=obs.shape[0],
                        replace=True,
                    )

                    # Ensure at least one sequence for each disease for each V-J combination:
                    # This will guarantee that each disease will have at least one sequence associated with each (V gene, J gene) combination in the obs dataframe.
                    # As a result, when we split into V-J subsets to train submodels, each subset will have data from more than one class.
                    # Without this, we would have some V-J subsets with only one class, and those submodels would fail to train.
                    # Implementation: Start with a dataframe holding all possible V-J combinations, repeated several times:
                    # (This is a small predetermined dataframe vis-a-vis the total size of obs)
                    all_vj_combinations_df = pd.DataFrame(
                        all_vj_combinations * 5, columns=["v_gene", "j_gene"]
                    )

                    # Fill the remaining rows (the majority of obs) with random V-J combinations depending on which disease we're in
                    remaining_shape = obs.shape[0] - all_vj_combinations_df.shape[0]
                    assert remaining_shape > 0  # Sanity check

                    def normalize(arr) -> np.ndarray:
                        arr = np.array(arr)
                        return arr / np.sum(arr)

                    more_v_genes = np.random.choice(
                        np.hstack(
                            [
                                # Each is a list of gene names
                                disease_associated_v_genes[healthy_label],
                                disease_associated_v_genes[disease],
                            ]
                        ),
                        p=normalize(
                            # We want disease-specific V genes to be prevalent, since model 1 will filter out the rarest 50% of V genes.
                            np.hstack(
                                [
                                    normalize(
                                        [0.1]
                                        * len(disease_associated_v_genes[healthy_label])
                                    ),
                                    normalize(
                                        [0.9] * len(disease_associated_v_genes[disease])
                                    ),
                                ]
                            )
                        ),
                        size=remaining_shape,
                        replace=True,
                    )

                    more_j_genes = np.random.choice(
                        available_j_genes, size=remaining_shape, replace=True
                    )

                    # Concatenate with the original precomputed V-J combinations lists, shuffle, and add to obs
                    obs["v_gene"] = np.hstack(
                        [all_vj_combinations_df["v_gene"], more_v_genes]
                    )
                    obs["j_gene"] = np.hstack(
                        [all_vj_combinations_df["j_gene"], more_j_genes]
                    )

                    all_obs.append(obs)

                # Create adata
                adata = anndata.AnnData(
                    X=np.vstack(all_X),
                    obs=pd.concat(all_obs, axis=0),
                    uns={
                        "embedded": config.embedder.name,
                        "embedded_fine_tuned_on_fold_id": fold_id,
                        "embedded_fine_tuned_on_gene_locus": gene_locus.name,
                    },
                )
                adata.obs_names_make_unique()  # make index unique again

                # Sanity check: all V-J subsets should include every disease (see above for why this is important)
                assert (
                    adata.obs.groupby(["v_gene", "j_gene"], observed=True)[
                        "disease"
                    ].nunique()
                    == len(diseases)
                ).all()

                # Generate other columns
                adata.obs["disease_subtype"] = adata.obs["disease"]
                adata.obs["cdr3_aa_sequence_trim_len"] = adata.obs[
                    "cdr3_seq_aa_q_trim"
                ].str.len()
                adata.obs["cdr1_seq_aa_q_trim"] = adata.obs["cdr3_seq_aa_q_trim"].copy()
                adata.obs["cdr2_seq_aa_q_trim"] = adata.obs["cdr3_seq_aa_q_trim"].copy()
                adata.obs["num_clone_members"] = np.random.poisson(
                    lam=50, size=adata.shape[0]
                )
                adata.obs["isotype_supergroup"] = np.random.choice(
                    helpers.isotype_groups_kept[gene_locus],
                    size=adata.shape[0],
                    replace=True,
                )
                # generate v_mut between 0 and 0.2 for BCR, or always 0 for TCR
                adata.obs["v_mut"] = (
                    np.random.uniform(low=0, high=0.2, size=adata.shape[0])
                    if gene_locus == GeneLocus.BCR
                    else 0.0
                )

                adata.raw = adata
                sc.pp.scale(adata)
                assert not adata.obs_names.duplicated().any()

                destination_dir = config.paths.scaled_anndatas_dir / gene_locus.name
                destination_dir.mkdir(parents=True, exist_ok=True)
                fname = f"fold.{fold_id}.{fold_label}.h5ad"
                adata.write(destination_dir / fname, compression="gzip")

    ## Change n_lambda used in hyperparameter-tuned linear models
    # save original value for later
    old_n_lambda = model_definitions.DEFAULT_N_LAMBDAS_FOR_TUNING
    model_definitions.DEFAULT_N_LAMBDAS_FOR_TUNING = 3

    ## Set desired sample weight strategy
    # save original value for later
    old_sample_weight_strategy = config.sample_weight_strategy
    config.sample_weight_strategy = sample_weight_strategy

    ## Set all_fold_ids and cross_validation_fold_ids to a smaller set for faster testing
    # save original values for later
    old_all_fold_ids = config.all_fold_ids
    old_cross_validation_fold_ids = config.cross_validation_fold_ids
    config.all_fold_ids = [0, -1]
    config.cross_validation_fold_ids = [0]

    # run tests
    yield

    # reset to original values
    model_definitions.DEFAULT_N_LAMBDAS_FOR_TUNING = old_n_lambda
    config.sample_weight_strategy = old_sample_weight_strategy
    config.all_fold_ids = old_all_fold_ids
    config.cross_validation_fold_ids = old_cross_validation_fold_ids


@pytest.mark.parametrize(
    "target_obs_column",
    [
        # This doesn't do any subsetting:
        TargetObsColumnEnum.disease,
        # This one does subsetting:
        TargetObsColumnEnum.lupus_vs_healthy,
    ],
)
def test_first_anndata_load_does_not_affect_second_anndata_load_from_cache(
    modified_config, target_obs_column
):
    def load():
        return io.load_fold_embeddings(
            fold_id=0,
            fold_label="train_smaller",
            gene_locus=GeneLocus.BCR,
            target_obs_column=target_obs_column,
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


@pytest.mark.parametrize(
    "target_obs_column",
    [
        # This doesn't do any subsetting:
        TargetObsColumnEnum.disease,
        # This one does subsetting:
        TargetObsColumnEnum.lupus_vs_healthy,
    ],
)
def test_modfiying_second_anndata_load_from_cache_does_not_affect_first_anndata_load(
    modified_config, target_obs_column
):
    def load():
        return io.load_fold_embeddings(
            fold_id=0,
            fold_label="train_smaller",
            gene_locus=GeneLocus.BCR,
            target_obs_column=target_obs_column,
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


def test_load_train_smaller_further_splits(modified_config):
    # train_smaller1 and train_smaller2 are further splits of train_smaller, and have a slightly different io.load_fold_embeddings path for efficiency
    def _load(fold_label):
        return io.load_fold_embeddings(
            fold_id=0,
            fold_label=fold_label,
            gene_locus=GeneLocus.BCR,
            target_obs_column=TargetObsColumnEnum.disease,
        ).obs["specimen_label"]

    specimens1 = _load("train_smaller1")
    specimens2 = _load("train_smaller2")
    assert (
        len(set(specimens1).intersection(set(specimens2))) == 0
    ), "train_smaller1 and train_smaller2 should have completely separate sets of specimen_labels"


# This is a very slow test, so run it as the last test in the suite.
@pytest.mark.order("last")
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
    fold_ids = [0, -1]  # config.all_fold_ids
    # Special fold names for "Model 3 separated by V and J genes":
    training_split_further_fold_name_base_sequence_model_train = (
        "train_smaller"
        # Technically this should be "train_smaller1", but for our small end-to-end test, this would cut out too many samples.
    )
    training_split_further_fold_name_rollup_model_train = "train_smaller"  # Technically this should be "train_smaller2", but for our small end-to-end test, this would cut out too many samples.

    gene_loci_used: GeneLocus = GeneLocus.BCR | GeneLocus.TCR
    # TODO add rest of TargetObsColumnEnum. The snapshot datasets don't have enough coverage of these classes to test them.
    target_obs_columns = [
        TargetObsColumnEnum.disease,
        TargetObsColumnEnum.disease_all_demographics_present,
        # TargetObsColumnEnum.age_group_binary_healthy_only,
        # TargetObsColumnEnum.covid_vs_healthy,
    ]
    expected_classes_by_target = {
        TargetObsColumnEnum.disease: ["HIV", "Healthy/Background", "Lupus"],
        TargetObsColumnEnum.disease_all_demographics_present: [
            "HIV",
            "Healthy/Background",
            "Lupus",
        ],
        # TargetObsColumnEnum.age_group_binary_healthy_only: ["50+", "under 50"],
        # TargetObsColumnEnum.covid_vs_healthy: ["Lupus", "Healthy/Background"],
    }

    # TODO: assert config.paths is what we expect
    print(config.paths)

    def _run_metamodel(
        gene_locus: GeneLocus, target_obs_column: TargetObsColumnEnum, fold_id: int
    ):
        metamodel_flavors = train_metamodel.get_metamodel_flavors(
            gene_locus=gene_locus,
            target_obs_column=target_obs_column,
            fold_id=fold_id,
            base_model_train_fold_name=training_fold_name,
            base_model_train_fold_name_for_sequence_model=training_split_further_fold_name_base_sequence_model_train,
            base_model_train_fold_name_for_aggregation_model=training_split_further_fold_name_rollup_model_train,
        )

        # Confirm all expected metamodel flavors were generated
        expected_metamodel_flavors_by_target: Dict[TargetObsColumnEnum, Set[str]] = {
            TargetObsColumnEnum.disease: {
                "default",
                "subset_of_submodels_repertoire_stats",
                "subset_of_submodels_convergent_cluster_model",
                "subset_of_submodels_sequence_model",
                "subset_of_submodels_repertoire_stats_convergent_cluster_model",
                "subset_of_submodels_repertoire_stats_sequence_model",
                "subset_of_submodels_convergent_cluster_model_sequence_model",
            },
            TargetObsColumnEnum.disease_all_demographics_present: {
                "default",
                "with_demographics_columns",
                "demographics_regressed_out",
                "demographics_only",
                "demographics_only_with_interactions",
                "demographics_only_age",
                "demographics_only_sex",
                "demographics_only_ethnicity_condensed",
            },
            # TargetObsColumnEnum.age_group_binary_healthy_only: {"default"},
            # TargetObsColumnEnum.covid_vs_healthy: {"default"},
        }
        if GeneLocus.BCR in gene_locus:
            expected_metamodel_flavors_by_target[TargetObsColumnEnum.disease].add(
                "isotype_counts_only"
            )
        assert (
            set(metamodel_flavors.keys())
            == expected_metamodel_flavors_by_target[target_obs_column]
        )

        for (
            metamodel_flavor,
            metamodel_config,
        ) in metamodel_flavors.items():
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
                chosen_models=["elasticnet_cv", "xgboost"],
                n_jobs=1,
                # control fold_id and cache manually so that we limit repetitive I/O
                clear_cache=False,
                # this is for CI only: don't swallow training errors
                fail_on_error=True,
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

                if metamodel_flavor.startswith("subset_of_submodels"):
                    # Special case: including only a subset of submodels
                    # The metamodel flavor name will contain the submodel names
                    metamodel_feature_prefixes_filtered = [
                        feature_prefix
                        for feature_prefix in metamodel_feature_prefixes
                        if feature_prefix in metamodel_flavor
                    ]
                else:
                    # Default: all submodels used
                    metamodel_feature_prefixes_filtered = metamodel_feature_prefixes

                for single_gene_locus in gene_locus:  # might be composite
                    for feature_prefix in metamodel_feature_prefixes_filtered:
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
                    "demographics_only_with_interactions",
                    "demographics_only_age",
                ]:
                    demographics_features.append("demographics:age")

                # Metamodel features also depend on OneHotEncoder choices, which are determined by:

                # a) the order of binary demographic column values in the metamodel's training set, like "sex":
                if metamodel_flavor in [
                    "with_demographics_columns",
                    "demographics_only",
                    "demographics_only_with_interactions",
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
                    "demographics_only_with_interactions",
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
            elif metamodel_flavor == "demographics_only_with_interactions":
                # Also add interaction terms, e.g.:

                # 'interaction|demographics:sex_F|demographics:age',
                # 'interaction|demographics:ethnicity_condensed_African|demographics:age',
                # 'interaction|demographics:ethnicity_condensed_Caucasian|demographics:age',
                # 'interaction|demographics:ethnicity_condensed_Hispanic/Latino|demographics:age',
                # 'interaction|demographics:ethnicity_condensed_Asian|demographics:age',
                # 'interaction|demographics:ethnicity_condensed_African|demographics:sex_F',
                # 'interaction|demographics:ethnicity_condensed_Caucasian|demographics:sex_F',
                # [...]

                interaction_terms = [
                    # See cartesian_product note about how the feature names are generated in reverse order when feeding in the same features_left and features_right
                    f"interaction|{demographics_features[j]}|{demographics_features[i]}"
                    # all pairs
                    for i in range(len(demographics_features))
                    for j in range(i + 1, len(demographics_features))
                    # except for combinations of ethnicity categorical dummy variables
                    # e.g. avoid 'interaction|demographics:ethnicity_condensed_Asian|demographics:ethnicity_condensed_Hispanic/Latino'
                    if not (
                        demographics_features[i].startswith(
                            "demographics:ethnicity_condensed_"
                        )
                        and demographics_features[j].startswith(
                            "demographics:ethnicity_condensed_"
                        )
                    )
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
            for metamodel_name in ["elasticnet_cv", "xgboost"]:
                for clf in [
                    BlendingMetamodel(
                        fold_id=fold_id,
                        metamodel_name=metamodel_name,
                        base_model_train_fold_name=training_fold_name,
                        metamodel_fold_label_train=metamodel_fold_label_train,
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_column,
                        metamodel_flavor=metamodel_flavor,
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

                    # TODO: run predictions with these models to check that we can load them from disk

    for gene_locus in gene_loci_used:
        # Control fold_id and cache manually so that we limit repetitive I/O
        for fold_id in fold_ids:
            for target_obs_col in target_obs_columns:
                # Train model 1
                logger.info(
                    f"Training model 1 for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_col}, sample_weight_strategy={sample_weight_strategy}."
                )
                train_repertoire_stats_model.run_classify_with_all_models(
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_col,
                    sample_weight_strategy=sample_weight_strategy,
                    fold_label_train=training_fold_name,
                    fold_label_test=validation_fold_name,
                    chosen_models=[
                        config.metamodel_base_model_names.model_name_overall_repertoire_composition[
                            gene_locus
                        ],
                    ],
                    n_jobs=1,
                    # control fold_id and cache manually so that we limit repetitive I/O
                    fold_ids=[fold_id],
                    clear_cache=False,
                    # this is for CI only: don't swallow training errors
                    fail_on_error=True,
                    # this is for CI only: we don't have sequences.sampled.parquet in CI; load our generated anndatas instead
                    load_obs_only=False,
                )

                logger.info(
                    f"Tuning model 1 on validation set for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_col}, sample_weight_strategy={sample_weight_strategy}."
                )
                # Load model 1
                clf = RepertoireClassifier(
                    fold_id=fold_id,
                    model_name=config.metamodel_base_model_names.model_name_overall_repertoire_composition[
                        gene_locus
                    ],
                    fold_label_train="train_smaller",
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_col,
                    sample_weight_strategy=sample_weight_strategy,
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
                    fold_label_train=training_split_further_fold_name_base_sequence_model_train,
                    fold_label_test=training_split_further_fold_name_rollup_model_train,
                    chosen_models=[
                        config.metamodel_base_model_names.model_name_convergent_clustering[
                            gene_locus
                        ],
                    ],
                    n_jobs=2,  # this parameter controls p-value thresholding tuning parallelization
                    target_obs_column=target_obs_col,
                    # control fold_id and cache manually so that we limit repetitive I/O
                    fold_ids=[fold_id],
                    clear_cache=False,
                    # reduced list of p-values for CI
                    # all are deliberately high to reduce potential for lots of abstentions on our tiny snapshot test dataset,
                    # which could cause tests to fail due to the resulting model training sets being too small / not having all classes included
                    # TODO: Remove this customization for CI? Or try [0.001, 0.01, 0.1]?
                    p_values=[0.5, 0.8],
                    # this is for CI only: don't swallow training errors
                    fail_on_error=True,
                    # this is for CI only: we don't have sequences.sampled.parquet in CI; load our generated anndatas instead
                    load_obs_only=False,
                )

                logger.info(
                    f"Tuning model 2 on validation set for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_col}."
                )
                # Load model 2
                clf = ConvergentClusterClassifier(
                    fold_id=fold_id,
                    model_name=config.metamodel_base_model_names.model_name_convergent_clustering[
                        gene_locus
                    ],
                    fold_label_train=training_split_further_fold_name_base_sequence_model_train,
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
                    fold_label_train=training_split_further_fold_name_base_sequence_model_train,
                    # Note that in CI, this is effectively not doing p-value cross validation on a separate held-out fold, because these two fold labels are identical (though it should work either way because our artificial snapshot datasets are designed with repeated sequences):
                    fold_label_test=training_split_further_fold_name_rollup_model_train,
                    chosen_models=[
                        config.metamodel_base_model_names.model_name_convergent_clustering[
                            gene_locus
                        ],
                    ],
                    n_jobs=2,  # this parameter controls p-value thresholding tuning parallelization
                    target_obs_column=target_obs_col,
                    # control fold_id and cache manually so that we limit repetitive I/O
                    fold_ids=[fold_id],
                    clear_cache=False,
                    p_values=[0.1, 0.5],  # reduced list for CI
                    # this is for CI only: don't swallow training errors
                    fail_on_error=True,
                    # this is for CI only: we don't have sequences.sampled.parquet in CI; load our generated anndatas instead
                    load_obs_only=False,
                )

                # Load exact matches model
                clf = ExactMatchesClassifier(
                    fold_id=fold_id,
                    model_name=config.metamodel_base_model_names.model_name_convergent_clustering[
                        gene_locus
                    ],
                    fold_label_train=training_split_further_fold_name_base_sequence_model_train,
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

                ### Train Model 3, separated by V gene + isotype, which is the default configured in config.py:
                sequence_subset_strategy: SequenceSubsetStrategy = (
                    config.metamodel_base_model_names.base_sequence_model_subset_strategy
                )
                v_gene_isotype_specific_sequence_models_to_train = list(
                    {
                        config.metamodel_base_model_names.base_sequence_model_name[
                            gene_locus
                        ],
                        # Train some other models to cover our bases
                        "elasticnet_cv_patient_level_optimization",
                    }
                )
                logger.info(
                    f"Training V-gene+isotype specific model 3 for gene_locus={gene_locus}, fold_id={fold_id}, target={target_obs_col}, sample_weight_strategy={sample_weight_strategy}."
                )
                train_vj_gene_specific_sequence_model.run_classify_with_all_models(
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_col,
                    sample_weight_strategy=sample_weight_strategy,
                    fold_label_train=training_split_further_fold_name_base_sequence_model_train,
                    fold_label_test=training_split_further_fold_name_rollup_model_train,
                    chosen_models=v_gene_isotype_specific_sequence_models_to_train,
                    n_jobs=1,
                    # control fold_id and cache manually so that we limit repetitive I/O
                    fold_ids=[fold_id],
                    clear_cache=False,
                    # this is for CI only: don't swallow training errors
                    fail_on_error=True,
                    # exclude rare V genes (default)
                    exclude_rare_v_genes=True,
                    # *This is the key parameter*:
                    sequence_subset_strategy=sequence_subset_strategy,
                )

                # Test reload - and will be used to test the specialized rollup below
                base_model = (
                    sequence_subset_strategy.base_model
                )  # e.g. VGeneIsotypeSpecificSequenceClassifier
                v_gene_isotype_specific_sequence_clf = base_model(
                    fold_id=fold_id,
                    model_name_sequence_disease=config.metamodel_base_model_names.base_sequence_model_name[
                        gene_locus
                    ],
                    fold_label_train=training_split_further_fold_name_base_sequence_model_train,
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_col,
                    sample_weight_strategy=sample_weight_strategy,
                )

                # Train specialized rollup that is split by V gene + isotype (or as configured in config.py defaults)
                # Which rollup models to train:
                def _strip_all_model_name_suffixes(name: str):
                    # Remove any model name suffixes corresponding to special customizations created inside the upcoming train.
                    # For example, convert "elasticnet_cv_mean_aggregated_as_binary_ovr" to "elasticnet_cv".
                    # Then the "_mean_aggregated_as_binary_ovr" variant will be trained within this upcoming train call.
                    # (These strings are stored in an enum in train_vj_gene_specific_sequence_model_rollup.py)
                    for (
                        suffix
                    ) in train_vj_gene_specific_sequence_model_rollup.ModelNameSuffixes:
                        name = name.replace(suffix.value, "")

                    for aggregation_strategy in AggregationStrategy:
                        name = name.replace(aggregation_strategy.model_name_suffix, "")

                    # Also, remove customizations from modify_fitted_model_lambda_setting() - these strings should match that function:
                    name = name.replace("lambda1se", "").replace("lambdamax", "")

                    return name

                rollup_models_to_train = {
                    # Train the model that will be used inside the metamodel.
                    # Remove any model name suffixes corresponding to special customizations created inside the upcoming train:
                    _strip_all_model_name_suffixes(
                        config.metamodel_base_model_names.aggregation_sequence_model_name[
                            gene_locus
                        ]
                    ),
                    #
                    # Also train xgboost because it has special handling of missing values:
                    "xgboost",
                }
                train_vj_gene_specific_sequence_model_rollup.run_classify_with_all_models(
                    gene_locus=gene_locus,
                    target_obs_column=target_obs_col,
                    sample_weight_strategy=sample_weight_strategy,
                    fold_label_train=training_split_further_fold_name_rollup_model_train,
                    fold_label_test=validation_fold_name,
                    # Which rollup models to train:
                    chosen_models=list(rollup_models_to_train),
                    n_jobs=1,
                    # Provide sequence model name here (rollup model will be trained on top of this model):
                    base_model_name=config.metamodel_base_model_names.base_sequence_model_name[
                        gene_locus
                    ],
                    base_model_fold_label_train=training_split_further_fold_name_base_sequence_model_train,
                    # Configure aggregation strategies to train.
                    # Let's train them all, because config.metamodel_base_model_names.aggregation_sequence_model_name[gene_locus] may use any of them,
                    # so we will want to have it available for metamodel training later in this end-to-end run.
                    aggregation_strategies=list(AggregationStrategy),
                    # control fold_id and cache manually so that we limit repetitive I/O
                    fold_ids=[fold_id],
                    clear_cache=False,
                    # this is for CI only: don't swallow training errors
                    fail_on_error=True,
                    # *This is the key parameter*:
                    sequence_subset_strategy=sequence_subset_strategy,
                )
                # Test the trained model.
                num_features = {}
                for rollup_model_name in {
                    f"{s}_mean_aggregated_as_binary_ovr" for s in rollup_models_to_train
                } | {
                    # Include the version of xgboost with special handling of missing values (it leaves them in instead of doing fillna).
                    "xgboost_mean_aggregated_as_binary_ovr_with_nans",
                    # Include reweighting by subset frequencies
                    "xgboost_mean_aggregated_as_binary_ovr_reweighed_by_subset_frequencies",
                }:
                    # Load and run this specialized rollup
                    v_gene_isotype_specific_rollup_clf_specialized = VJGeneSpecificSequenceModelRollupClassifier(
                        fold_id=fold_id,
                        # Provide sequence model name here (rollup model will be trained on top of this model):
                        base_sequence_model_name=config.metamodel_base_model_names.base_sequence_model_name[
                            gene_locus
                        ],
                        base_model_train_fold_label=training_split_further_fold_name_base_sequence_model_train,
                        # Rollup model details:
                        rollup_model_name=rollup_model_name,
                        fold_label_train=training_split_further_fold_name_rollup_model_train,
                        gene_locus=gene_locus,
                        target_obs_column=target_obs_col,
                        sample_weight_strategy=sample_weight_strategy,
                        # We pass in the sequence classifier explicitly here, but we could instead pass sequence_subset_strategy
                        sequence_classifier=v_gene_isotype_specific_sequence_clf,
                    )
                    v_gene_isotype_specific_rollup_featurized_specialized = (
                        v_gene_isotype_specific_rollup_clf_specialized.featurize(
                            io.load_fold_embeddings(
                                fold_id=fold_id,
                                fold_label=validation_fold_name,
                                gene_locus=gene_locus,
                                target_obs_column=target_obs_col,
                                sample_weight_strategy=sample_weight_strategy,
                            )
                        )
                    )
                    # Confirm classes match
                    # However, because train_smaller1 is a subset,
                    # we may not really see all the classes.
                    assert set(
                        v_gene_isotype_specific_rollup_clf_specialized.classes_
                    ) <= set(expected_classes_by_target[target_obs_col])
                    assert set(
                        v_gene_isotype_specific_rollup_clf_specialized.classes_
                    ) == set(
                        io.load_fold_embeddings(
                            fold_id=fold_id,
                            fold_label=training_split_further_fold_name_rollup_model_train,
                            gene_locus=gene_locus,
                            target_obs_column=target_obs_col,
                            sample_weight_strategy=sample_weight_strategy,
                        ).obs[target_obs_col.value.obs_column_name]
                    )
                    # Confirm predictions shape n_specimens x n_classes
                    v_gene_isotype_specific_rollup_preds_specialized = (
                        v_gene_isotype_specific_rollup_clf_specialized.predict_proba(
                            v_gene_isotype_specific_rollup_featurized_specialized.X
                        )
                    )
                    assert v_gene_isotype_specific_rollup_preds_specialized.shape == (
                        len(
                            v_gene_isotype_specific_rollup_featurized_specialized.sample_names
                        ),
                        len(v_gene_isotype_specific_rollup_clf_specialized.classes_),
                    )
                    num_features[
                        rollup_model_name
                    ] = v_gene_isotype_specific_rollup_clf_specialized.n_features_in_
                # Confirm that the number of features is the same for all rollup models
                assert all(
                    nf == next(iter(num_features.values()))
                    for nf in num_features.values()
                )

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
