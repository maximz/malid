"""
Run fold -1 (global fold) fine-tuned language model on external validation cohorts,
and apply existing scaling and PCA transformations.

Recall that we have a separate fine-tuned language model for each train-smaller set.
Here we will treat external validation cohorts as fold -1's test set.
Apply the langauge model, scaling, and PCA transformations trained on fold -1's train-smaller set.

This writes to config.paths.scaled_anndatas_dir.

Usage examples:
    > python scripts/external_validation_cohort.run_embedding_fine_tuned.and_scale.py --help;
    > python scripts/external_validation_cohort.run_embedding_fine_tuned.and_scale.py;               # all loci
    > python scripts/external_validation_cohort.run_embedding_fine_tuned.and_scale.py --locus BCR; # BCR only
    > python scripts/external_validation_cohort.run_embedding_fine_tuned.and_scale.py --n_gpus 1; # use only 1 GPU
"""

import anndata
from typing import List, Union
import time
import gc
import pandas as pd
import os
import logging
import click
import multiprocessing
from multiprocessing.pool import Pool, AsyncResult
import psutil
import choosegpu

from malid import config, cli_utils, helpers, apply_embedding
from malid.datamodels import GeneLocus

logger = logging.getLogger(__name__)

# Create a multiprocessing.Pool with an initiializer and a queue as init-args
# In the queue, put gpu IDs
# Each process will get from that queue once, and thus it gets its GPU ID
# That way, we can use multiprocessing.Pool to manage all the processes for us
# see e.g. https://gist.github.com/dnozay/b2462798ca89fbbf0bf4#file-main-py-L38 and https://stackoverflow.com/a/41434133/130164 and https://rvprasad.medium.com/in-python-choose-builtin-process-pools-over-custom-process-pools-d43f019633f1

## Worker initialization.


def init_worker(
    initargs_queue: multiprocessing.Queue, _gene_locus: GeneLocus, _fold_id: int
):
    """
    Initialize multiprocessing pool worker.

    We pass a queue to this function.
    The function will grab exactly one GPU ID from the queue.
    Then the worker's state will be initialized with that GPU ID - we will load the language model accordingly.

    Alternatives considered:
    1) Don't use an initializer; just run a `pool.map(initializer, list(range(n_gpus)) + ['cpu' for cpu_id in range(n_cpus)])` after.
        But we are not guaranteed to run exactly once on each multiprocessing worker process.
    2) Manage multiprocessing processes ourselves with a shared input and results queue. Complex and error prone.
    """
    # We will store the individual worker's state in these variables
    # see https://thelaziestprogrammer.com/python/multiprocessing-pool-expect-initret-proposal
    # and https://superfastpython.com/multiprocessing-pool-initializer/#Example_of_Accessing_an_Initialized_Variable_in_a_Worker
    global embedder
    global transformations_to_apply
    global full_process_identifier
    global gene_locus
    global fold_id

    # Can't use the same names as global vars unfortunately.
    gene_locus = _gene_locus
    fold_id = _fold_id

    # Get one GPU ID from the queue.
    gpu_id = initargs_queue.get()

    _setup_worker_logging(worker_id=os.getpid())

    # initialize embedder once, and reuse
    if gpu_id == "cpu":
        choosegpu.configure_gpu(enable=False)
    else:
        choosegpu.configure_gpu(enable=True, gpu_device_ids=[gpu_id])

    embedder = apply_embedding.load_embedding_model(
        gene_locus=gene_locus, fold_id=fold_id
    )
    transformations_to_apply = apply_embedding.load_transformations(
        gene_locus=gene_locus, fold_id=fold_id
    )

    full_process_identifier = f"GPU #{gpu_id} (process #{os.getpid()})"
    logger.info(f"Initialized embedder: {full_process_identifier}")

    # This doesn't mark a particular task as done, i.e. it doesn't know about the specific item
    # It simply reduces the number of unfinished tasks
    # Used so we can queue.join(), i.e. wait for all to be finished
    initargs_queue.task_done()


def _setup_worker_logging(worker_id: int):
    import malid
    from notebooklog import setup_logger

    malid.logger, log_fname = setup_logger(
        log_dir=config.paths.log_dir, name=f"embedding_worker_{worker_id}"
    )
    malid.logger.info(log_fname)
    print(log_fname)


## Worker process.


def embed_single_participant(participant_label):
    """Run embedding in a child process."""
    # This is assigned to a particular worker in the multiprocessing.Pool.
    # We can directly use the global variables (specific to a particular worker) defined in init_worker.

    itime = time.time()
    adata = _process_participant(
        participant_label=participant_label,
        embedder=embedder,
        transformations_to_apply=transformations_to_apply,
        gene_locus=gene_locus,
        fold_id=fold_id,
    )
    elapsed_time = time.time() - itime

    if adata is None:
        # This was a bad sample. Skip
        logger.warning(
            f"Skipped {participant_label}, no anndata returned - {full_process_identifier}."
        )
        return None

    sequences_per_second_rate = adata.shape[0] / elapsed_time
    loading_debug_message = f"processed in {elapsed_time:0.0f} seconds ({sequences_per_second_rate:0.2f} sequences per second)"
    logger.info(
        f"Finished {participant_label} with {full_process_identifier}: {adata.shape} {loading_debug_message}. Percent RAM used: {psutil.virtual_memory().percent}%"
    )

    # fix obs names
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs_names_make_unique()

    # sanity check
    assert not adata.obs["specimen_label"].isna().any()
    assert not adata.obs["participant_label"].isna().any()

    # Make the participant/specimen identifiers all strings and then categoricals
    adata.obs["participant_label"] = (
        adata.obs["participant_label"].astype(str).astype("category")
    )
    adata.obs["specimen_label"] = (
        adata.obs["specimen_label"].astype(str).astype("category")
    )

    # Save
    output_dir = config.paths.external_data_embeddings / gene_locus.name
    output_dir.mkdir(parents=True, exist_ok=True)
    fname_out = output_dir / f"{participant_label}.h5"
    logger.info(
        f"Writing external cohort participant {participant_label} ({gene_locus}) using fold -1: {adata.shape[0]} sequences in {adata.obs['specimen_label'].nunique()} specimens -> {fname_out}"
    )
    adata.write(fname_out)

    # Garbage collect
    del adata
    gc.collect()

    return fname_out


def _process_participant(
    participant_label, embedder, transformations_to_apply, gene_locus, fold_id
) -> Union[anndata.AnnData, None]:
    # Load sequences
    fname = config.paths.external_processed_data / f"{participant_label}.parquet"
    if not fname.exists():
        # Expected when participant had no sequences pass our filters
        logger.warning(f"Skipping {participant_label} - no processed data found")
        return None

    # Read in file.
    df = pd.read_parquet(
        fname,
        columns=[
            "specimen_label",
            "extracted_isotype",
            "isotype_supergroup",
            "v_gene",
            "j_gene",
            "cdr1_seq_aa_q_trim",
            "cdr2_seq_aa_q_trim",
            "cdr3_seq_aa_q_trim",
            "cdr3_aa_sequence_trim_len",
            "participant_label",
            "disease",
            "disease_subtype",
            "igh_or_tcrb_clone_id",
            "v_mut",
            "is_peak",
        ],
    )
    if df.shape[0] == 0:
        # This is unexpected
        logger.warning(f"Skipping {participant_label} - no sequences found")
        return None

    # Filter down to this particular gene locus - certain isotype groups
    df = df.loc[df["isotype_supergroup"].isin(helpers.isotype_groups_kept[gene_locus])]
    if df.shape[0] == 0:
        logger.warning(
            f"{participant_label} had no sequences for {gene_locus}. Skipping."
        )
        return None

    # Filter down to peak timepoints
    df = df.loc[df["is_peak"]]
    if df.shape[0] == 0:
        logger.warning(
            f"{participant_label} had no peak timepoint sequences for {gene_locus}. Skipping."
        )
        return None

    logger.info(
        f"Loaded {df.shape[0]} sequences for {participant_label} - {gene_locus}"
    )

    # Make adata
    # Drop any columns we don't want to have saved in the anndata for space reasons
    adata = apply_embedding.run_embedding_model(
        embedder=embedder,
        df=df.drop("is_peak", axis=1),
        gene_locus=gene_locus,
        fold_id=fold_id,
    )
    adata = apply_embedding.transform_embedded_anndata(
        transformations_to_apply=transformations_to_apply,
        adata=adata,
    )
    return adata


## Get metadata.


def get_external_cohort_participants(gene_locus: GeneLocus):
    """Get metadata"""
    external_cohort_specimens = pd.read_csv(
        config.paths.metadata_dir / "generated.external_cohorts.all_specimens.tsv",
        sep="\t",
    )

    # filter down external_cohort_specimens:

    # must belong to this locus
    external_cohort_specimens = external_cohort_specimens[
        external_cohort_specimens["gene_locus"] == gene_locus.name
    ]

    # must exist
    external_cohort_specimens["fname"] = external_cohort_specimens[
        "participant_label"
    ].apply(
        lambda participant_label: config.paths.external_processed_data
        / f"{participant_label}.parquet"
    )
    external_cohort_specimens = external_cohort_specimens[
        external_cohort_specimens["fname"].apply(os.path.exists)
    ]

    # must be peak
    external_cohort_specimens = external_cohort_specimens[
        external_cohort_specimens["is_peak"] == True
    ]

    # make sure single row per participant
    external_cohort_specimens = external_cohort_specimens.groupby(
        ["participant_label", "study_name", "fname"]
    ).head(n=1)

    if (
        "ImmuneCode" in external_cohort_specimens["study_name"].values
        and "Emerson" in external_cohort_specimens["study_name"].values
    ):
        # subsample from the giant amount of Adaptive TCR healthy repertoires
        external_cohort_specimens = pd.concat(
            [
                external_cohort_specimens[
                    external_cohort_specimens["study_name"] != "Emerson"
                ],
                external_cohort_specimens[
                    external_cohort_specimens["study_name"] == "Emerson"
                ].sample(
                    n=(external_cohort_specimens["study_name"] == "ImmuneCode").sum(),
                    replace=False,
                    random_state=0,
                ),
            ],
            axis=0,
        )

    external_cohort_specimens = external_cohort_specimens.reset_index(drop=True)

    return external_cohort_specimens["participant_label"].unique()


## Run the multiprocessing operation


def error_callback(err):
    # raise err
    # Don't re-raise, because program will not terminate
    logger.exception(f"Multiprocessing error: {err}", exc_info=err)


def run_on_single_locus(gene_locus: GeneLocus, n_gpu_processes=2, n_cpu_processes=0):
    itime = time.time()
    GeneLocus.validate_single_value(gene_locus)
    fold_id = -1  # use global fold

    ## Set up GPUs/CPUs
    # How many processes we want
    # Each GPU process also has one core maxed out at 100% CPU for it.
    # Each CPU process is about 10x slower than a GPU process, and is actually multiprocessed over ~30 cores (maxed out to 100% CPU).
    # Interestingly, adding even a single CPU process seems to slow down the GPU processes a bit...
    # The math from early experiments suggests that 4GPU-0CPU > 4GPU-1CPU > 4GPU-2CPU >> 4GPU-10CPU when considered holistically
    # Though I'm curious how this would change with different CPU batching amount settings

    # Get available GPU IDs, starting from the back
    gpu_ids = [index for (index, uuid) in reversed(choosegpu.get_available_gpus())]
    assert len(gpu_ids) >= n_gpu_processes, "Not enough GPUs available"
    gpu_ids = gpu_ids[:n_gpu_processes] + ["cpu"] * n_cpu_processes

    # Pass worker-specific initializer arguments to subprocesses in a queue, so we are guaranteed each process is initialized exactly once.

    # Create a queue that supports task_done so we can track how many items have been processed
    # Use managed queues so that we avoid synchronization errors in standard multiprocessing.JoinableQueue (or multiprocessing.Queue). We kept seeing results_queue.empty() returning True when it obviously wasn't empty.
    # See the many notes and warnings at https://docs.python.org/3/library/multiprocessing.html#pipes-and-queues -- especially about the need for child processes to flush their queue writes. The manager takes care of all that for us - no more synchronization issues.
    # Under the hood, this passes a "proxy" of the queue (a reference address) to the child processes.

    queue_manager = multiprocessing.Manager()
    initargs_queue = queue_manager.Queue()
    for gpu_id in gpu_ids:
        initargs_queue.put(gpu_id)

    with Pool(
        len(gpu_ids),
        initializer=init_worker,
        initargs=(initargs_queue, gene_locus, fold_id),
    ) as pool:
        # Wait till initialization complete (requires calling task_done)
        initargs_queue.join()

        # Get tasks to process
        participant_labels = get_external_cohort_participants(gene_locus=gene_locus)
        logger.info(f"Enqueueing {len(participant_labels)} tasks for {gene_locus}")

        # To iterate over results as tasks are completed, use: `for result in pool.imap(func, iterable):` or `imap_unordered`
        # Or use this if we don't need the result:
        # issue tasks
        async_result: AsyncResult = pool.map_async(
            embed_single_participant, participant_labels, error_callback=error_callback
        )

        # wait for tasks to complete
        async_result.wait()

        etime = time.time()
        if async_result.successful():
            click.echo(f"{gene_locus} tasks complete. Time elapsed: {etime - itime}")
        else:
            logger.error(f"Error in {gene_locus} tasks. Time elapsed: {etime - itime}")
            try:
                # This will fail
                async_result.get()
            except Exception as err:
                # logger.exception(f"Multiprocessing pool failed: {err}")
                raise


@click.command()
@cli_utils.accepts_gene_loci
@click.option(
    "--n_gpus",
    default=2,
    type=int,
    show_default=True,
)
@click.option(
    "--n_cpus",
    default=0,
    type=int,
    show_default=True,
)
def run(gene_locus: List[GeneLocus], n_gpus=2, n_cpus=0):
    gene_locus: GeneLocus = GeneLocus.combine_flags_list_into_single_multiflag_value(
        gene_locus
    )
    click.echo(f"Selected gene_locus: {gene_locus}")
    GeneLocus.validate(gene_locus)  # packed (by OR) into single item here
    choosegpu.configure_gpu(enable=False)  # subprocesses will configure their own GPUs
    for single_gene_locus in gene_locus:
        click.echo(f"Running on {single_gene_locus}...")
        run_on_single_locus(
            gene_locus=single_gene_locus, n_gpu_processes=n_gpus, n_cpu_processes=n_cpus
        )


if __name__ == "__main__":
    run()
