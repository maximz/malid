# %%
from summarynb import show, table, chunks
from malid import config
import pandas as pd
from IPython.display import display, Markdown

# %%
gene_loci = config.gene_loci_used
gene_loci


# %%
def run_summary(gene_locus):
    display(Markdown(f"## {gene_locus}"))
    base_dir = (
        config.paths.external_cohort_evaluation_output_dir / "default" / gene_locus.name
    )
    fname_results = base_dir / "compare_model_scores.tsv"
    if not fname_results.exists():
        print("Not run")
        return

    results = pd.read_csv(fname_results, sep="\t", index_col=0)
    show(table(results))
    for model_names in chunks(results.sort_index().index.tolist(), 3):
        show(
            [
                [
                    base_dir / f"confusion_matrix.{model_name}.png"
                    for model_name in model_names
                ],
                [
                    base_dir / f"classification_report.{model_name}.txt"
                    for model_name in model_names
                ],
            ],
            headers=model_names,
            max_width=200,
        )
    display(Markdown("---"))


# %%
for single_locus in gene_loci:
    # single locus
    run_summary(single_locus)
if len(gene_loci) > 1:
    # multi-loci
    run_summary(gene_loci)

# %%

# %%
