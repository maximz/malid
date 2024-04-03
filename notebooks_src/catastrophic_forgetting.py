# %% [markdown]
# # Catastrophic Forgetting
#
# ## Reviewer request:
#
# - State the perplexity of the original LLM model [a specific metric we can compute]
#
# - State the perplexity on the test set of BCR sequences of the fine tuned model.
#
# - State the perplexity of the test set of non-BCR sequences (the original LLM test set) of the fine tuned model. These numbers are needed to assess whether the fine tuning was successful and no catastrophic forgetting
#
#
# ## Conclusions from this notebook
# - Finetuning improves the cross-entropy loss on the malid sequences as well as the uniref50 sequences.
# - The improvement is more significant on malid sequences than on uniref50 sequences
# - Finetuning does not introduce catastrophic forgetting of uniref50 training data
#
# ## Some notes
# - Implemented the same filtering of uniref50 data as the unirep [methods section](https://www.nature.com/articles/s41592-019-0598-1#Sec9):
#
# > "We removed proteins longer than 2,000 amino acids and records containing noncanonical amino-acid symbols (X, B, Z, J), randomly selected test and validation subsets for monitoring training (1% of the overall dataset each) and used the rest of the data (~24 million sequences) in training."

# %%
run_extra_analysis = False  # feature flag to do a bunch of extra computation
enable_gpu = True  # feature flag to enable/disable GPU usage

# %%
import choosegpu

choosegpu.configure_gpu(enable=enable_gpu)

# %%
import pandas as pd
from typing import Optional
from Bio import SeqIO
import gzip
import re
from malid.datamodels import GeneLocus
from malid import config, helpers, apply_embedding
from malid.embedders.base_embedder import BaseEmbedder
import genetools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from IPython.display import display

# %%
assert (
    config.embedder.is_fine_tuned
), "Current embedder is not fine tuned. The point of this notebook is to examine the fine tuning process."
print(config.embedder.name)

# %%

# %%
bad_chars = re.compile(".*[XBZJ].*")
# NOTE: don't limit to the first n rows of fasta or else you'll get a weird size distribution of sequences
uniref_sequences = []
uniref_taxonomies = []
with gzip.open("../data/uniref50.fasta.gz", "rt") as handler:
    fasta_sequences = SeqIO.parse(handler, "fasta")
    for sequence in fasta_sequences:
        name, seq = sequence.id, str(sequence.seq)
        if len(seq) < 2000 and not bad_chars.match(seq):
            uniref_sequences.append(seq)
            uniref_taxonomies.append(
                sequence.description.split("Tax=")[-1].split("TaxID")[0].strip()
            )

uniref_sequences = pd.Series(uniref_sequences)


# %%
# what are the most prevalent types of sequences in the uniref data?
uniref_taxonomies = pd.Series(uniref_taxonomies)
uniref_taxonomies.value_counts().head(30)

# %%
# what is the sequence length distribution in uniref50?

length_counts = uniref_sequences.apply(lambda x: len(x)).value_counts().sort_index()
window = 500

fig, ax = plt.subplots()
plt.scatter(x=length_counts.index, y=length_counts, s=1)
rect = patches.Rectangle(
    (window, 0),
    -window,
    length_counts.max(),
    linewidth=1,
    edgecolor="r",
    facecolor="none",
)
ax.add_patch(rect)
plt.xlabel("sequence length")
plt.ylabel("num sequences")

length_counts
fig, ax = plt.subplots()
plt.scatter(x=length_counts.index[:window], y=length_counts[:window], s=1)
plt.xlabel("sequence length")
plt.ylabel("num sequences")

# %%
gene_locus = next(iter(config.gene_loci_used))
gene_locus

# %%
# load malid seqs
malid_sequences = apply_embedding.load_sequence_embedding_content_for_fold(
    fold_id=1,
    fold_label="test",
    gene_locus=gene_locus,
    embedder_class=config.embedder,
)[0]

# %%
# what is the distribution of sequence lengths in the malid data?
length_counts = (
    pd.Series(malid_sequences).apply(lambda x: len(x)).value_counts().sort_index()
)
plt.scatter(length_counts.index, length_counts)

# %%
# Load and instantiate standard non-fine-tuned model.
non_fine_tuned_embedder: BaseEmbedder = config.embedder.non_fine_tuned_version()


def fetch_malid_fold_data(
    gene_locus: GeneLocus, fold_id: int, downsample: Optional[int] = None
):
    """
    For a given fold, load the fine-tuned model, as well as the sequence data for train/test/val splits
    Optionally downsample the reads
    """
    embedder = apply_embedding.load_embedding_model(
        gene_locus=gene_locus,
        fold_id=fold_id,
    )
    train_sequences, val_sequences, test_sequences = [
        apply_embedding.load_sequence_embedding_content_for_fold(
            fold_id=fold_id,
            fold_label=fold_label,
            gene_locus=gene_locus,
            embedder_class=config.embedder,
        )[0]
        for fold_label in ["train_smaller", "validation", "test"]
    ]

    if downsample:
        train_sequences = np.random.choice(train_sequences, downsample)
        val_sequences = np.random.choice(val_sequences, downsample)
        test_sequences = np.random.choice(test_sequences, downsample)

    return embedder, test_sequences, val_sequences, train_sequences


# %% [markdown]
# # Experiment 1: Explore all of the models from each cross validation fold
# This takes approximately 2 days to run, so hidden behind a feature flag for now

# %%
downsample = 10000

if run_extra_analysis:
    # keep uniref set consistent throughout analysis
    uniref_sample = np.random.choice(uniref_sequences, downsample)

    results = {}
    for gene_locus in config.gene_loci_used:
        for fold_id in config.cross_validation_fold_ids:
            (
                fine_tuned_embedder,
                test_sequences,
                val_sequences,
                train_sequences,
            ) = fetch_malid_fold_data(gene_locus, fold_id, downsample)
            results[
                (gene_locus.name, fold_id, "finetune_test")
            ] = fine_tuned_embedder.calculate_cross_entropy_loss(test_sequences)
            results[
                (gene_locus.name, fold_id, "finetune_val")
            ] = fine_tuned_embedder.calculate_cross_entropy_loss(val_sequences)
            results[
                (gene_locus.name, fold_id, "finetune_train")
            ] = fine_tuned_embedder.calculate_cross_entropy_loss(train_sequences)
            results[
                (gene_locus.name, fold_id, "finetune_uniref")
            ] = fine_tuned_embedder.calculate_cross_entropy_loss(
                uniref_sample,
                # Uniref sequences are very long, so we need to use lower batch size
                batch_size=10,
            )
            results[
                (gene_locus.name, fold_id, "base_test")
            ] = non_fine_tuned_embedder.calculate_cross_entropy_loss(test_sequences)
            results[
                (gene_locus.name, fold_id, "base_val")
            ] = non_fine_tuned_embedder.calculate_cross_entropy_loss(val_sequences)
            results[
                (gene_locus.name, fold_id, "base_train")
            ] = non_fine_tuned_embedder.calculate_cross_entropy_loss(train_sequences)
            results[
                (gene_locus.name, fold_id, "base_uniref")
            ] = non_fine_tuned_embedder.calculate_cross_entropy_loss(
                uniref_sample,
                # Uniref sequences are very long, so we need to use lower batch size
                batch_size=10,
            )


# %%
if run_extra_analysis:
    summary = (
        pd.Series(results)
        .reset_index()
        .rename(
            columns={
                "level_0": "gene_locus",
                "level_1": "fold",
                "level_2": "experiment",
                0: "loss",
            }
        )
    )
    summary["perplexity"] = np.exp(summary["loss"])
    summary.to_csv(
        config.paths.fine_tuned_embedding_dir / "cross_validation_loss_analysis.csv"
    )
    summary["model"] = summary["experiment"].apply(lambda x: x.split("_")[0])
    summary["dataset"] = summary["experiment"].apply(lambda x: x.split("_")[1])
    display(summary.drop("experiment", axis=1))
    display(
        summary.groupby(["gene_locus", "experiment"]).agg(
            {"loss": "mean", "perplexity": "mean"}
        )
    )

# %%
if run_extra_analysis:
    plt.figure()
    sns.stripplot(x="loss", y="experiment", hue="gene_locus", data=summary)
    plt.figure()
    sns.stripplot(
        y="perplexity",
        x="model",
        hue="dataset",
        data=summary,
        palette="Paired",
        order=["base", "finetune"],
    )
    plt.legend(bbox_to_anchor=(1, 1))

    plt.figure()
    sns.stripplot(
        y="loss",
        x="model",
        hue="dataset",
        data=summary,
        palette="Paired",
        order=["base", "finetune"],
    )
    plt.legend(bbox_to_anchor=(1, 1))
    plt.ylabel("cross entropy loss")

    plt.figure()
    sns.stripplot(
        y="loss",
        x="model",
        hue="gene_locus",
        data=summary,
        palette="Paired",
        order=["base", "finetune"],
    )
    plt.legend(bbox_to_anchor=(1, 1))
    plt.ylabel("cross entropy loss")

# %% [markdown]
# # Experiment 2: Only use -1 fold model
# - Use validation set of sequences from the -1 fold model
# - Add 20 bootstrap samples of 1000 sequences
# - Make figure with confidence intervals

# %%
results = {}
for gene_locus in config.gene_loci_used:
    fine_tuned_embedder = apply_embedding.load_embedding_model(
        gene_locus=gene_locus,
        fold_id=-1,
    )
    print(
        fine_tuned_embedder,
        fine_tuned_embedder.embedder_sequence_content,
        non_fine_tuned_embedder,
        non_fine_tuned_embedder.embedder_sequence_content,
    )
    val_sequences = apply_embedding.load_sequence_embedding_content_for_fold(
        fold_id=-1,
        fold_label="validation",
        gene_locus=gene_locus,
        embedder_class=config.embedder,
    )[0]
    for i in range(20):
        print(i)
        val_sample = np.random.choice(val_sequences, 1000)
        uniref_sample = np.random.choice(uniref_sequences, 1000)
        results[
            (gene_locus.name, i, "best", "validation")
        ] = fine_tuned_embedder.calculate_cross_entropy_loss(val_sample)
        results[
            (gene_locus.name, i, "best", "uniref")
        ] = fine_tuned_embedder.calculate_cross_entropy_loss(
            uniref_sample,
            # Uniref sequences are very long, so we need to use lower batch size
            batch_size=10,
        )
        results[
            (gene_locus.name, i, "base", "validation")
        ] = non_fine_tuned_embedder.calculate_cross_entropy_loss(val_sample)
        results[
            (gene_locus.name, i, "base", "uniref")
        ] = non_fine_tuned_embedder.calculate_cross_entropy_loss(
            uniref_sample,
            # Uniref sequences are very long, so we need to use lower batch size
            batch_size=10,
        )


# %%
bootstraps = (
    pd.Series(results)
    .reset_index()
    .rename(
        columns={
            "level_0": "locus",
            "level_1": "iteration",
            "level_2": "model",
            "level_3": "dataset",
            0: "cross_val_entropy",
        }
    )
)

# perplexity = 2^{cross entropy loss}
bootstraps["perplexity"] = np.exp2(bootstraps["cross_val_entropy"])

bootstraps["model"] = bootstraps["model"].map(
    {"best": "Fine-tuned LLM", "base": "Base LLM"}
)
bootstraps.loc[
    (bootstraps["locus"] == "BCR") & (bootstraps["dataset"] == "validation"), "Dataset"
] = "BCR sequences"
bootstraps.loc[
    (bootstraps["locus"] == "TCR") & (bootstraps["dataset"] == "validation"), "Dataset"
] = "TCR sequences"
bootstraps.loc[bootstraps["dataset"] == "uniref", "Dataset"] = "Uniref50 sequences"

# %%
bootstraps.to_csv(
    config.paths.output_dir / f"fine_tuning.catastrophic_forgetting_analysis.csv"
)

# %%

# %%
# We can resume from here by reloading from disk
bootstraps = pd.read_csv(
    config.paths.output_dir / f"fine_tuning.catastrophic_forgetting_analysis.csv",
    index_col=0,
)
bootstraps

# %%

# %%

# %%
bootstraps.groupby(["locus", "model", "dataset"]).agg("mean")

# %%
# borrow color mapping for disease color palette
colors = list(helpers.disease_color_palette.values())
color_mapping = dict(
    zip(
        bootstraps.Dataset.sort_values().unique(),
        colors[: bootstraps.Dataset.nunique()],
    )
)
color_mapping.keys()

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

if "BCR sequences" in color_mapping:
    sns.barplot(
        x="model",
        y="cross_val_entropy",
        hue="Dataset",
        order=["Base LLM", "Fine-tuned LLM"],
        hue_order=["BCR sequences", "Uniref50 sequences"],
        data=bootstraps.loc[bootstraps.locus == "BCR"],
        palette=[color_mapping[i] for i in ["BCR sequences", "Uniref50 sequences"]],
        ax=ax[0],
        errorbar=None,
    )
    ax[0].set_ylabel("Cross Entropy Loss")
    ax[0].set_xlabel("Model")
    ax[0].set_title("BCR finetuning")
    ax[0].get_legend().remove()

if "TCR sequences" in color_mapping:
    sns.barplot(
        x="model",
        y="cross_val_entropy",
        hue="Dataset",
        order=["Base LLM", "Fine-tuned LLM"],
        hue_order=["TCR sequences", "Uniref50 sequences"],
        data=bootstraps.loc[bootstraps.locus == "TCR"],
        palette=[color_mapping[i] for i in ["TCR sequences", "Uniref50 sequences"]],
        ax=ax[1],
        errorbar=None,
    )
    ax[1].set_ylabel("Cross Entropy Loss")
    ax[1].set_xlabel("Model")
    ax[1].set_title("TCR finetuning")
    ax[1].get_legend().remove()

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
indexes = [labels.index(x) for x in set(labels)]
labels = [labels[i] for i in indexes]
lines = [lines[i] for i in indexes]

fig.legend(lines, labels, bbox_to_anchor=(1.2, 0.5))

fig.tight_layout()
genetools.plots.savefig(
    fig,
    config.paths.output_dir
    / f"fine_tuning.catastrophic_forgetting_analysis_cross_entropy_loss.png",
    dpi=300,
)


# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

if "BCR sequences" in color_mapping:
    sns.barplot(
        x="model",
        y="perplexity",
        hue="Dataset",
        order=["Base LLM", "Fine-tuned LLM"],
        hue_order=["BCR sequences", "Uniref50 sequences"],
        data=bootstraps.loc[bootstraps.locus == "BCR"],
        palette=[color_mapping[i] for i in ["BCR sequences", "Uniref50 sequences"]],
        ax=ax[0],
        errorbar=None,
    )
    ax[0].set_ylabel("Perplexity")
    ax[0].set_xlabel("Model")
    ax[0].set_title("BCR finetuning")
    ax[0].get_legend().remove()

if "TCR sequences" in color_mapping:
    sns.barplot(
        x="model",
        y="perplexity",
        hue="Dataset",
        order=["Base LLM", "Fine-tuned LLM"],
        hue_order=["TCR sequences", "Uniref50 sequences"],
        data=bootstraps.loc[bootstraps.locus == "TCR"],
        palette=[color_mapping[i] for i in ["TCR sequences", "Uniref50 sequences"]],
        ax=ax[1],
        errorbar=None,
    )
    ax[1].set_ylabel("Perplexity")
    ax[1].set_xlabel("Model")
    ax[1].set_title("TCR finetuning")
    ax[1].get_legend().remove()

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
indexes = [labels.index(x) for x in set(labels)]
labels = [labels[i] for i in indexes]
lines = [lines[i] for i in indexes]

fig.legend(lines, labels, bbox_to_anchor=(1.2, 0.5))

fig.tight_layout()
genetools.plots.savefig(
    fig,
    config.paths.output_dir
    / f"fine_tuning.catastrophic_forgetting_analysis_perplexity.png",
    dpi=300,
)


# %%
