# %%

# %%

# %%
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns

# %%
import pandas as pd

# %%
import joblib
import glob
import genetools

# %%
import choosegpu
from malid import config
from malid.datamodels import GeneLocus

# do not use GPU - may be implicitly used by joblib imports of jax-unirep parameters?
choosegpu.configure_gpu(enable=False)


# %%

# %%

# %%
def get_fold_data(output_dir):
    all_results = [
        joblib.load(f) for f in glob.glob(f"{output_dir}/loss.epoch.*.joblib")
    ]
    if len(all_results) == 0:
        # this was not run yet
        return None

    all_results_df = pd.DataFrame(all_results)
    # At this point each row's training_loss and holdout_loss are actually numpy arrays with single items. Weird
    # TODO: fix upstream.
    # Workaround: get single elements out
    all_results_df["training_loss"] = all_results_df["training_loss"].apply(
        lambda i: i.item()
    )
    all_results_df["holdout_loss"] = all_results_df["holdout_loss"].apply(
        lambda i: i.item()
    )
    return all_results_df.sort_values("epoch").reset_index(drop=True)


# %%

# %%
for gene_locus in config.gene_loci_used:
    GeneLocus.validate_single_value(gene_locus)
    fig, axarr = plt.subplots(nrows=len(config.all_fold_ids), ncols=2, figsize=(10, 10))
    for fold_id in config.all_fold_ids:
        all_results_df = get_fold_data(
            config.paths.fine_tuned_embedding_dir / gene_locus.name / f"fold_{fold_id}"
        )
        if all_results_df is None:
            print(f"Not yet run: gene_locus={gene_locus}, fold={fold_id}")
            continue

        axarr[fold_id, 0].scatter(
            all_results_df["epoch"], all_results_df["training_loss"]
        )
        axarr[fold_id, 0].set_title(f"{gene_locus} fold {fold_id}, training loss")

        axarr[fold_id, 1].scatter(
            all_results_df["epoch"], all_results_df["holdout_loss"]
        )
        axarr[fold_id, 1].set_title(f"{gene_locus} fold {fold_id}, validation loss")

        # Add early stopping - choose params from epoch with lowest validation loss, before overfitting
        # get best row
        best_results = all_results_df.iloc[all_results_df["holdout_loss"].idxmin()]
        best_epoch = best_results["epoch"]
        max_epoch = all_results_df["epoch"].max()
        ready_to_stop = max_epoch >= best_epoch + 2  # heuristic
        print(
            f"{gene_locus} fold {fold_id}:\tbest epoch is {best_epoch:n} out of {max_epoch:n} epochs.\t{'Ready to stop' if ready_to_stop else 'Keep running'}."
        )
        best_params = joblib.load(
            config.paths.fine_tuned_embedding_dir
            / gene_locus.name
            / f"fold_{fold_id}"
            / f"params.epoch.{best_epoch:n}.joblib"
        )

        # save out best params
        joblib.dump(
            best_params,
            config.paths.fine_tuned_embedding_dir
            / gene_locus.name
            / f"fold_{fold_id}"
            / "best_params.joblib",
        )
    fig.tight_layout()
    genetools.plots.savefig(
        fig, config.paths.output_dir / f"fine_tuning.loss.{gene_locus.name}.png"
    )

# %%

# %%
