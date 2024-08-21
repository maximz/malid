# %% [markdown]
# # Mal-ID Tutorial
#
# Mal-ID uses B cell receptor (BCR) and T cell receptor (TCR) sequencing data to classify disease or immune state.
#
# In the Mal-ID framework, we train three BCR and three TCR disease classifiers with different ways of extracting feature from immune receptor sequencing data.
# Then we train an ensemble metamodel, a classifier that uses the predicted disease probabilities from the six base models to make a final prediction of disease status.
#
# #### This tutorial covers:
#
# 1. Mal-ID configuration settings
# 2. Accessing sample metadata
# 3. Loading sequence data
# 4. Loading Models 1, 2, and 3 and making predictions with these base models
# 5. Loading the ensemble metamodel and making predictions
#
# We will cover all of the components in this schematic:
#
# <div>
# <img src="../schematic.png" width="800"/>
# </div>
#
# This tutorial assumes you've already followed the "runbook": the commands in the readme to load the data and train the models.
#
# **([Main repo here](https://github.com/maximz/malid))**

# %%

# %% [markdown]
# ## Start with some necessary imports

# %%
import pandas as pd
from malid import config, io, helpers
from malid.trained_model_wrappers import (
    RepertoireClassifier,
    ConvergentClusterClassifier,
    VJGeneSpecificSequenceModelRollupClassifier,
    BlendingMetamodel,
)
from malid.datamodels import GeneLocus, TargetObsColumnEnum

# %%

# %% [markdown]
# ## Review Mal-ID configuration settings

# %% [markdown]
# The inclusion criteria for the dataset — meaning which samples get divided into cross validation folds — are defined as a `CrossValidationSplitStrategy` object in `malid/datamodels.py`.
#
# The default strategy is `CrossValidationSplitStrategy.in_house_peak_disease_timepoints`, which includes peak disease timepoints from our primary in-house dataset. Indeed, that's the active cross validation split strategy:

# %%
config.cross_validation_split_strategy

# %%

# %% [markdown]
# The data is divided into three folds: `fold_id` can be 0, 1, or 2.
#
# Each fold has a `train_smaller`, `validation`, and `test` set, referred to as a `fold_label`. (The `train_smaller` set is further subdivided into `train_smaller1` and `train_smaller2`.)
#
# Each sample will be in one test set. We also make sure that all samples from the same person have the same `fold_label`.
#
# Finally, we define a special `fold_id=-1` "global" fold that does not have a `test` set. All the data is instead used in the `train_smaller` and `validation` fold labels. (The `train_smaller` to `validation` proportion is the same as for other fold IDs, but both sets are larger than usual.)
#
# <div>
# <img src="../cross_validation.png" width="600"/>
# </div>
#
#
# The list of all fold IDs is therefore 0, 1, 2, and -1:

# %%
config.all_fold_ids

# %%

# %% [markdown]
# The language model being used is ESM-2, applied to the CDR3 region:

# %%
config.embedder.name  # The name

# %%
config.embedder.embedder_sequence_content  # The sequence region being embedded

# %%

# %% [markdown]
# Our models are configured to use both BCR and TCR data:

# %%
config.gene_loci_used

# %% [markdown]
# Just to clarify, `GeneLocus.BCR|TCR` means the union of BCR and TCR — both are active. This is the same as writing `GeneLocus.BCR | GeneLocus.TCR`:

# %%
GeneLocus.BCR | GeneLocus.TCR

# %%

# %%

# %% [markdown]
# ## Load metadata
#
# Here's a Pandas DataFrame with all of the samples in our database.
#
# The key fields are:
#
# - `participant_label`: the patient ID
# - `specimen_label`: the sample ID
# - `disease`

# %%
metadata = helpers.get_all_specimen_info()
metadata

# %%

# %% [markdown]
# Each sample is identified by a `specimen_label` and has a boolean column named `in_training_set`, indicating whether a sample met the requirements for inclusion in the cross validation divisions and passed QC requirements (see the readme for more details).
#
# Let's look at only the samples that passed those filters:

# %%
metadata = metadata[metadata["in_training_set"]]
metadata

# %%

# %% [markdown]
# The metadata also includes `disease_subtype`, `study_name`, `age`, `sex`, and `ethnicity_condensed` (ancestry) for a particular sample, as available.

# %%

# %% [markdown]
# ## Load data
#
# Let's choose a specific fold ID we will work with. As described above, our options are 0, 1, and 2 for the cross validation folds, and -1 for the global fold. Let's choose the first fold:

# %%
fold_id = 0

# %%

# %% [markdown]
# Classification targets are defined in an enumeration called `TargetObsColumnEnum`:

# %%
[t.name for t in TargetObsColumnEnum]

# %%

# %% [markdown]
# Each classification target is associated with a metadata field. Let's focus on `TargetObsColumnEnum.disease`, which is our main classification goal to predict the `disease` metadata column:

# %%
target_obs_column = TargetObsColumnEnum.disease

# %%

# %% [markdown]
# This target is associated with the "disease" metadata field. You can tell by looking at the `obs_column_name` attribute:

# %%
target_obs_column.value.obs_column_name

# %%

# %% [markdown]
# Here are the values of that metadata field:

# %%
metadata["disease"].value_counts()

# %%

# %% [markdown]
# At this point, the data have been split into cross-validation folds and the sequences have been transformed into language model embeddings.
#
# BCR and TCR data can be loaded with separate calls to `io.load_fold_embeddings()`:

# %%
adata_bcr = io.load_fold_embeddings(
    fold_id=fold_id,
    fold_label="test",  # Load the held out data
    gene_locus=GeneLocus.BCR,
    target_obs_column=target_obs_column,
)
adata_bcr

# %%
adata_tcr = io.load_fold_embeddings(
    fold_id=fold_id,
    fold_label="test",
    gene_locus=GeneLocus.TCR,  # Same as previous code block, except now we are loading TCR data
    target_obs_column=TargetObsColumnEnum.disease,
)
adata_tcr

# %%

# %% [markdown]
# In Mal-ID, we store data in [AnnData containers](https://anndata.readthedocs.io/en/latest/). We often use the variable name `adata` for these objects.
#
# AnnData containers have a `.X` property with the language model embedding vector for each sequence, and a `.obs` property with the metadata for each sequence:

# %%
adata_bcr.X

# %%
adata_bcr.obs

# %%

# %% [markdown]
# `adata_bcr.X` has one row per sequence and one column per language model embedding dimension. `adata_bcr.obs` has one row per sequence and one column per metadata field. The same is true for `adata_tcr`.

# %%
adata_bcr.X.shape  # samples x embedding dimensions

# %%
adata_bcr.obs.shape  # samples x metadata fields

# %%

# %% [markdown]
# Sequence-level metadata fields include:
#
# - The `participant_label` (patient ID) and `specimen_label` (sample ID) from which the sequence originated
# - `v_gene`: IGHV or TRBV gene segment name
# - `j_gene`: IGHJ or TRBJ gene segment name
# - `cdr3_seq_aa_q_trim`: CDR3 amino acid sequence
# - `v_mut`: Somatic hypermutation rate, specifically the fraction of V region nucleotides that are mutated (BCR only)
# - `isotype_supergroup`: either `IGHG`, `IGHA`, `IGHD-M` (combining IgD and IgM), or `TCRB`. The name "supergroup" refers to the fact that we are overloading the definition of isotypes by combining IgD and IgM, and by labeling all TCR data as `isotype_supergroup="TCRB"` for convenience, even though TCRs don't have isotypes by definition. (Note that IgE is filtered out in the subsampling step, as are unmutated IgD and IgM sequences which represent naive B cells.)

# %%

# %% [markdown]
# The choice of `target_obs_column` argument in the data loading `io.load_fold_embeddings()` call matters. The AnnData object is filtered down to samples that are "in scope" for a particular classification target. For example, if you choose `target_obs_column=TargetObsColumnEnum.sex_healthy_only`, the AnnDatas will only have samples from healthy individuals for whom the sex is known. (See `malid/datamodels.py` for the exact definition of each `TargetObsColumnEnum` option.)

# %%

# %% [markdown]
# > *Aside*
# >
# > When we ran `io.load_fold_embeddings()` above, you may have noticed some log messages about caching the data. When the large AnnData object is first loaded, it's copied from its original file path — usually on a network-mounted file system — to scratch storage on the machine where you're running this notebook. This speeds up further data loading.
# >
# > Another type of caching happens silently: the imported data is cached in memory, so that repeated calls to `io.load_fold_embeddings` with the same arguments don't result in slow loads of the same data over and over. The data is cached before we run the filtering for a particular classification target, so we can still leverage the cached version even if we switch the `target_obs_column` argument. This helps in our training loop, so we can train the model for many classification targets seamlessly. The cache is capped at four entries, and older data is automatically removed from the cache to make room for what's being used now. (You can also manually clear the cache with `io.clear_cached_fold_embeddings()`, or disable the cache altogether by setting the environment variable `MALID_DISABLE_IN_MEMORY_CACHE=true` before loading any Mal-ID Python code.)

# %%

# %% [markdown]
# ## Model 1

# %% [markdown]
# Model 1 uses overall summary statistics of the BCR or TCR repertoire to predict disease status.
#
# This way of generating features can be tied into many classification algorithms, e.g. logistic regression or random forests. We try a bunch of classification algorithms and choose the one with highest performance on the validation set, which is not seen during training. The same is true for Models 2 and 3.
#
# We've recorded our choices of the chosen "model name" for Model 1, Model 2, and Model 3 in `config.metamodel_base_model_names`. The ensemble metamodel will use these versions of the base Mal-ID components:

# %%
config.metamodel_base_model_names

# %%

# %% [markdown]
# Let's look specifically at the version of Model 1 chosen for BCR data:

# %%
config.metamodel_base_model_names.model_name_overall_repertoire_composition[
    GeneLocus.BCR
]

# %%

# %% [markdown]
# The model name is `elasticnet_cv0.25`, which is elastic net regularized logistic regression with an L1-L2 ratio of 0.25. (The exact definition is in `malid/train/model_definitions.py`.)
#
# > _Aside_:
# > >
# > Models with `_cv` in the name use internal (nested) cross validation to tune their hyperparameters.

# %%

# %% [markdown]
# Let's load this version of Model 1 for BCR data:

# %%
clf1 = RepertoireClassifier(
    fold_id=fold_id,
    # Load "elasticnet_cv0.25"
    model_name=config.metamodel_base_model_names.model_name_overall_repertoire_composition[
        GeneLocus.BCR
    ],
    fold_label_train="train_smaller",  # Indicates which part of the data the model was trained on
    gene_locus=GeneLocus.BCR,  # A different model is trained for each sequencing locus
    target_obs_column=target_obs_column,  # A different model is trained for each classification target
)
clf1

# %%

# %% [markdown]
# We've loaded a `RepertoireClassifier`, which is a wrapper around a scikit-learn model stored in `_inner`.
#
# In this case, it's a wrapper around a scikit-learn Pipeline:

# %%
type(clf1._inner)

# %%
clf1._inner

# %%

# %% [markdown]
# The scikit-learn Pipeline passes V-J gene use counts through `log1p`, scaling, and PCA transformations. This happens for IgG, IgA, and IgM separately, and is coordinated by a `ColumnTransformer` step.
#
# The resulting PCs — along with the somatic hypermutation features marked "remainder" and "passthrough" — then go through standardization (`StandardScalerThatPreservesInputType`) and logistic regression (`GlmnetLogitNetWrapper`).
#
#
# Here are the original input feature names. Notice how there's a feature for each V gene, J gene, and isotype combination here, which then gets reduced into a smaller set of features by PCA:

# %%
clf1.feature_names_in_  # Attribute access is passed through to the _inner scikit-learn Pipeline

# %%

# %% [markdown]
# And now here is the reduced set of features coming out of the `ColumnTransformer` step. Notice how the V gene/J gene count features have turned into 15 PCs per isotype:

# %%
clf1.named_steps["columntransformer"].get_feature_names_out()

# %%

# %% [markdown]
# Here is the final step of the pipeline. As expected based on the `elasticnet_cv0.25` model name we specified when loading the trained model from disk, it's a elasticnet logistic regression with an 0.25 L1-L2 ratio:

# %%
clf1.steps[-1]

# %%

# %% [markdown]
# **Mal-ID models have the following API:**
#
# - **`clf.featurize(adata)`**: this function accepts an AnnData object and generates features specific for the model. The features and metadata are returned in a `FeaturizedData` container, which we'll explore below. (The features themselves are in the `.X` attribute of the `FeaturizedData` container.)
# - **`clf.predict_proba(features)`**: this function accepts features and returns predicted class probabilities by running the model.
# - **`clf.predict(features)`**: this function accepts features and returns predicted class labels by running the model.
#
# **_Common pattern:_** `predicted_class_probabilities = clf.predict_proba(clf.featurize(adata).X)`.
#
# Let's walk through this with Model 1. First, let's generate features from the held-out BCR data:

# %%
featurized_model1_data = clf1.featurize(adata_bcr)
type(featurized_model1_data)

# %%

# %% [markdown]
# We now have a `FeaturizedData` container. Let's unpack its contents:

# %%
# Features
featurized_model1_data.X

# %%
# Ground truth
featurized_model1_data.y

# %%
# Sample names
featurized_model1_data.sample_names

# %%
# Sample metadata
featurized_model1_data.metadata

# %%

# %% [markdown]
# Now run the model to predict the per-class probabilities:

# %%
clf1.predict_proba(featurized_model1_data.X)

# %%

# %% [markdown]
# To make this easier to read, let's bring in row and column names:

# %%
# Row names
featurized_model1_data.sample_names

# %%
# Column names
clf1.classes_

# %%
# Table of predicted class probabilities
pd.DataFrame(
    clf1.predict_proba(featurized_model1_data.X),
    index=featurized_model1_data.sample_names,
    columns=clf1.classes_,
)

# %%

# %% [markdown]
# **We just generated a table of predicted class probabilities for Model 1, using `featurize()` and `predict_proba()`.**
#
# Alternatively, we can use the model to generate a single predicted label for each sample:

# %%
clf1.predict(featurized_model1_data.X)

# %%

# %% [markdown]
# ## Model 2
#
# Model 2 uses clustering to identify shared groups of sequences across individuals with the same disease. Then we predict disease using the number of disease-associated cluster hits per sample.
#
# As before, let's first check which version is used in the ensemble metamodel. It's the ridge logistic regression version of Model 2:

# %%
# Chosen model names for Models 1, 2, and 3
config.metamodel_base_model_names

# %%
# The model name chosen for Model 2 - BCR
config.metamodel_base_model_names.model_name_convergent_clustering[GeneLocus.BCR]

# %%

# %% [markdown]
# Now let's load that version:

# %%
clf2 = ConvergentClusterClassifier(
    fold_id=fold_id,
    model_name=config.metamodel_base_model_names.model_name_convergent_clustering[
        GeneLocus.BCR
    ],
    fold_label_train="train_smaller1",  # The model was trained on train_smaller1, with hyperparameter tuning on train_smaller2.
    gene_locus=GeneLocus.BCR,
    target_obs_column=target_obs_column,
)
clf2

# %%

# %% [markdown]
# We get a `ConvergentClusterClassifier` object.
#
# Just as with `RepertoireClassifier` for Model 1, `ConvergentClusterClassifier` is a wrapper around a scikit-learn Pipeline:

# %%
type(clf2._inner)

# %%

# %% [markdown]
# But this time, the Pipeline is simpler:

# %%
clf2._inner

# %%

# %% [markdown]
# What's happening here: The pipeline confirms that the expected feature names are present, it standardizes the features, and then it runs logistic regression.

# %%

# %% [markdown]
# To run the model, let's start by featurizing the dataset, just as we did with Model 1:

# %%
featurized_model2_data = clf2.featurize(adata_bcr)
type(featurized_model2_data)

# %%

# %% [markdown]
# The `FeaturizedData` container can be unpacked the same way:

# %%
# Features (disease-associated cluster hit counts)
featurized_model2_data.X

# %%
# Ground truth
featurized_model2_data.y

# %%
# Sample names
featurized_model2_data.sample_names

# %%
# Sample metadata
featurized_model2_data.metadata

# %%

# %% [markdown]
# But this time, the `FeaturizedData` container also has some extra fields. Unlike the other components of Mal-ID, Model 2 abstains from prediction if none of the sequences in a sample match any disease-associated clusters. These abstentions are also stored in the `FeaturizedData` container too:

# %%
# These samples had no features generated — they are abstentions:
featurized_model2_data.abstained_sample_names

# %%
# Ground truth for abstained samples
featurized_model2_data.abstained_sample_y

# %%
# Metadata for abstained samples
featurized_model2_data.abstained_sample_metadata

# %%

# %% [markdown]
# Let's run the model to predict the per-class probabilities for the samples that are not abstentions:

# %%
# Table of predicted class probabilities
pd.DataFrame(
    clf2.predict_proba(featurized_model2_data.X),
    index=featurized_model2_data.sample_names,
    columns=clf2.classes_,
)

# %%

# %% [markdown]
# The `clf2.predict_proba(clf2.featurize(adata_bcr).X)` pattern is the same as what we saw above for Model 1.

# %%

# %% [markdown]
# ## Model 3

# %% [markdown]
# Model 3 uses language model embeddings of BCR and TCR sequences to predict disease state, in two stages:
#
# * **Sequence stage**: Predict which type of patient a sequence comes from, based on the language model embedding for that sequence. _This is a sequence-level model._ It's actually trained separately for each V gene and isotype.
# * **Aggregation stage**: Predict the disease status for an entire sample, based on the per-sequence predictions from the sequence stage. _This is a sample-level or person-level model._
#
# Each stage is trained separately. The sequence stage is trained on the `train_smaller1` fold, and the aggregation stage is trained on the `train_smaller2` fold.
#
# The selected model names are recorded in `config.metamodel_base_model_names` as `base_sequence_model_name` and `aggregation_sequence_model_name` for the two stages, respectively:

# %%
config.metamodel_base_model_names

# %%

# %% [markdown]
# Let's load the selected versions of both stages.
#
# Both stages can be loaded together through the `VJGeneSpecificSequenceModelRollupClassifier` class, which represents the aggregation stage:

# %%
clf3 = VJGeneSpecificSequenceModelRollupClassifier(
    # First, all the usual parameters, like fold ID, sequencing locus, and classification target:
    fold_id=fold_id,
    gene_locus=GeneLocus.BCR,
    target_obs_column=target_obs_column,
    #
    # Model 3 includes a seqeunce stage and an aggregation stage.
    # The aggregation stage is trained on top of the sequence stage, so to speak.
    # First, provide the sequence stage model name:
    base_sequence_model_name=config.metamodel_base_model_names.base_sequence_model_name[
        GeneLocus.BCR
    ],
    # The sequence stage was trained on train_smaller1:
    base_model_train_fold_label="train_smaller1",
    #
    # Next, provide the aggregation stage model name here:
    rollup_model_name=config.metamodel_base_model_names.aggregation_sequence_model_name[
        GeneLocus.BCR
    ],
    # The aggregation stage was trained on train_smaller2:
    fold_label_train="train_smaller2",
)
clf3

# %%

# %% [markdown]
# The sequence stage is automatically loaded and stored inside of the `VJGeneSpecificSequenceModelRollupClassifier`:

# %%
clf3.sequence_classifier

# %%

# %% [markdown]
# The sequence stage, a `VGeneIsotypeSpecificSequenceClassifier` object, is actually a collection of models trained separately for each V gene and isotype:

# %%
clf3.sequence_classifier.models_

# %%

# %%

# %% [markdown]
# Returning to the aggregation stage, it accepts the following features:

# %%
clf3.feature_names_in_

# %%

# %% [markdown]
# **How do we get features like `Influenza_IGHV3-23_IGHA`, which represents the average probability of the Influenza class across the IGHV3-23, IgA sequences in each sample?**
#
# First, a vector of per-class predicted probabilities is generated for each sequence, using the model associated with the V gene and isotype the sequence belongs to.
#
# Then the probabilities are aggregated across sequences from the same sample. Probabilities are only comparable between sequences scored by the same model, so the aggregation happens separately for each V gene and isotype group. For BCR, the aggregation strategy used is just an average:

# %%
clf3.aggregation_strategy

# %%

# %% [markdown]
# Once we have those features, the "aggregation stage" involves running this scikit-learn Pipeline:

# %%
clf3._inner

# %%

# %% [markdown]
# You may recognize the first two steps of the Pipeline from before:
#
# * `MatchVariables` confirms the expected feature names are present.
# * `StandardScalerThatPreservesInputType` standardizes the features.
#
# But the third step, `BinaryOvRClassifierWithFeatureSubsettingByClass`, is new.
#
# What's happening here is that the aggregation stage model is fitted in a one-versus-rest fashion, with one model for each class (e.g. Covid-19 vs rest):

# %%
clf3.named_steps["binaryovrclassifierwithfeaturesubsettingbyclass"].estimators_

# %%

# %% [markdown]
# Additionally, the submodel for each class is trained _only with features corresponding to that class_.
#
# For example, the sequence-level model generates predicted class probabilities `P(Covid-19)`, `P(HIV)`, `P(Lupus)`, and so forth for every sequence. But when making predictions of Covid-19 using the "Covid-19 vs rest" model, we should only look at the `P(Covid-19)` values. Sequence-level probabilities for the other classes, like `P(HIV)`, should have no bearing.
#
# Let's confirm that only Covid-19 specific features enter the "Covid-19 vs rest" model. All of these are `P(Covid-19)` features:

# %%
clf3.named_steps["binaryovrclassifierwithfeaturesubsettingbyclass"].estimators_[
    0
].clf.feature_names_in_

# %%

# %% [markdown]
# Now that we have Model 3 loaded, let's featurize the dataset to be able to use the model.
#
# This means every sequence is run through its associated sequence-level classifier, then the sequence probabilities are aggregated into sample-level features:

# %%
featurized_model3_data = clf3.featurize(adata_bcr)
type(featurized_model3_data)

# %% [markdown]
# > _Aside:_
# >
# > The log message about "VJGeneSpecificSequenceModelRollupClassifier featurization matrix N/As due to specimens not having any sequences with particular V/J gene pairs" is a bit misleading.
# >
# > What it's actually referring to is missing values due to some samples having no sequences to score and aggregate for a particular V gene and isotype. For these samples, the `P(Covid-19)`, `P(HIV)`, `P(Lupus)`, and so on are set to the value `1 / n_classes` for the V gene and isotype combination that are missing.

# %%

# %% [markdown]
# Let's review the features that were generated:

# %%
# Features (in this case, they've already been standardized)
featurized_model3_data.X

# %%
# Ground truth
featurized_model3_data.y

# %%
# Sample names
featurized_model3_data.sample_names

# %%
# Sample metadata
featurized_model3_data.metadata

# %%

# %% [markdown]
# Finally, let's run the model to predict the per-class probabilities at a _sample level_:

# %%
# Table of predicted class probabilities
pd.DataFrame(
    clf3.predict_proba(featurized_model3_data.X),
    index=featurized_model3_data.sample_names,
    columns=clf3.classes_,
)

# %%

# %%

# %% [markdown]
# ## Ensemble metamodel combining Models 1, 2, and 3 and combining BCR and TCR

# %% [markdown]
# Finally, let's load the ensemble metamodel, which brings all the other model components together for a final prediction of disease status.
#
# We'll specify `metamodel_name="ridge_cv"` to load the ridge logistic regression version which we highlight in the paper. (The featurization uses the base models chosen in `config.metamodel_base_model_names` as we reviewed above, so the features will be the same regardless of `metamodel_name`. This parameter just controls which classification algorithm is used in the metamodel itself.)
#
# We'll also specify `metamodel_flavor="default"`, which refers to combining Models 1, 2, and 3. Other `metamodel_flavor` options include:
#
# - `subset_of_submodels_repertoire_stats` (Model 1 only)
# - `subset_of_submodels_convergent_cluster_model` (Model 2 only)
# - `subset_of_submodels_sequence_model` (Model 3 only)
# - `subset_of_submodels_repertoire_stats_convergent_cluster_model` (Models 1 and 2 only)
# - and so on.

# %%
clf_metamodel = BlendingMetamodel.from_disk(
    fold_id=fold_id,
    target_obs_column=target_obs_column,
    metamodel_name="ridge_cv",  # Which metamodel version to use
    base_model_train_fold_name="train_smaller",  # The base components are fitted on the train_smaller set
    metamodel_fold_label_train="validation",  # The metamodel is fitted on the validation set
    gene_locus=GeneLocus.BCR | GeneLocus.TCR,  # Use BCR and TCR components together
    metamodel_flavor="default",  # Use Models 1 + 2 + 3
)

clf_metamodel

# %%

# %% [markdown]
# Like we've seen with the other models, `BlendingMetamodel` is a wrapper around a scikit-learn Pipeline that confirms the expected features are present, standardizes them, and then runs ridge logistic regression:

# %%
clf_metamodel._inner

# %%

# %% [markdown]
# The features are per-class predicted probabilities from each submodel. For example, `BCR:repertoire_stats:Covid19` is `P(Covid-19)` according to the BCR version of Model 1, the repertoire summary statistics model. Here's the full list of features:

# %%
clf_metamodel.feature_names_in_

# %%

# %% [markdown]
# As expected, there are three BCR and three TCR submodels:

# %%
clf_metamodel.metamodel_config.submodels

# %%

# %% [markdown]
# Let's featurize our input data to use with the metamodel. This time, the call to `featurize()` requires wrapping the input AnnData as a `dict[GeneLocus, AnnData]` — meaning a dictionary that maps from a sequencing locus to an AnnData object. This is because the BCR+TCR metamodel requires both AnnDatas at the same time to generate the features.

# %%
featurized_metamodel_data = clf_metamodel.featurize(
    {GeneLocus.BCR: adata_bcr, GeneLocus.TCR: adata_tcr}
)
type(featurized_metamodel_data)

# %%

# %% [markdown]
# > _Aside:_
# >
# > There are two new log messages here worth highlighting:
# >
# > 1. The first line, with "dropping specimens from GeneLocus.BCR anndata": these samples are removed because they have BCR data only — no TCR data. The BCR+TCR metamodel only runs on samples that have both BCR and TCR data available.
# > 2. The last line, with "Abstained specimens": Model 2 abstained from prediction for these samples, because none of the sequences in these samples matched any of the disease-associated clusters. The abstention propogates up into the metamodel.

# %%

# %% [markdown]
# Let's unwrap the `FeaturizedData` container as usual. This time, the features are the predicted class probabilities from the base models:

# %%
# Features
featurized_metamodel_data.X

# %%
# Ground truth
featurized_metamodel_data.y

# %%
# Sample names
featurized_metamodel_data.sample_names

# %%
# Sample metadata
featurized_metamodel_data.metadata

# %%

# %% [markdown]
# And we'll run our final prediction, like with the other models:

# %%
# Table of predicted class probabilities
pd.DataFrame(
    clf_metamodel.predict_proba(featurized_metamodel_data.X),
    index=featurized_metamodel_data.sample_names,
    columns=clf_metamodel.classes_,
)

# %%

# %% [markdown]
# We'll end by calculating the AUROC for fold 0:

# %%
import sklearn.metrics

sklearn.metrics.roc_auc_score(
    y_true=featurized_metamodel_data.y,
    y_score=clf_metamodel.predict_proba(featurized_metamodel_data.X),
    multi_class="ovo",  # Multiclass AUC calculated in one-versus-one fashion
    average="weighted",  # Take class size-weighted average of the binary AUROC calculated for each pair of classes
)

# %%

# %% [markdown]
# See the readme for instructions about customizing Mal-ID for additional datasets and classification targets.

# %%

# %%
