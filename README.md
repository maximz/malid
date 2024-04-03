# Mal-ID [![codecov](https://codecov.io/gh/maximz/malid/branch/master/graph/badge.svg)](https://codecov.io/gh/maximz/malid)

**Preprint**: [Disease diagnostics using machine learning of immune receptors](https://www.biorxiv.org/content/10.1101/2022.04.26.489314)

## Installation, part 1: base environment

### Production: GPU conda environment (preferred)

If you don't already have Conda installed:

```bash
# install Mambaforge (https://github.com/conda-forge/miniforge#mambaforge)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash "Mambaforge-$(uname)-$(uname -m).sh" -b -p "$HOME/miniconda"

source ~/miniconda/bin/activate
conda init
# log out and log back in
```

If you do already have Conda installed, just install mamba:

```bash
# Install mamba to replace conda
conda install mamba -n base -c conda-forge
```

Now create a conda environment:

```bash
# Create env
# Based on https://docs.rapids.ai/install and https://github.com/rapidsai/cuml/blob/branch-23.06/conda/environments/all_cuda-118_arch-x86_64.yaml
mamba create -n cuda-env-py39 -c conda-forge python=3.9 -y;

# Activate env
conda activate cuda-env-py39;

which nvcc # should be blank - confirms that systemwide cuda env isn't visible

# Install environment.
# Somehow, putting this in an `environment.yml` and running `mamba env create -f environment.yml` breaks the solver.
# Based on https://docs.rapids.ai/install and https://github.com/rapidsai/cuml/blob/branch-23.06/conda/environments/all_cuda-118_arch-x86_64.yaml
# Install cudatoolkit along with cuda and cudnn libraries (don't use system ones)
# cudnn: Find cudnn version based on tensorflow GPU compatibility chart https://www.tensorflow.org/install/source#gpu and what's available at conda-forge (https://anaconda.org/conda-forge/cudnn/files). See also these docs: https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html
# hdf5: version requirement comes from libgdal
# graphviz, scanpy also installed
mamba install \
    -c rapidsai \
    -c conda-forge \
    -c nvidia \
    cuda=11.8 \
    cuda-nvcc=11.8 \
    cuda-python=11.8 \
    'conda-forge::cupy=12.1*' \
    cuda-version=11.8 \
    'conda-forge::cudnn=8.8*' \
    cudatoolkit=11.8 \
    rapids=23.06 \
    'gcc_linux-64=13.1.0' \
    'librmm==23.6.*' \
    c-compiler \
    cxx-compiler \
    'dask-cuda==23.6.*' \
    'dask-cudf==23.6.*' \
    'conda-forge::h5py=3.7.0' \
    'conda-forge::hdf5=1.12.1' \
    'conda-forge::python-graphviz' 'conda-forge::graphviz' \
    'conda-forge::python-snappy' 'conda-forge::snappy' \
    'bioconda::seqkit' \
    -y;

# We also need a specific Fortran compiler version for glmnet (but looks like we can have libgfortran5 installed simultaneously). Specifically, we need libgfortran.so.4.
# Including this package above breaks the solver, so we have to install separately. This will also downgrade libgfortran-ng to 7.5.0.
# mamba install -c conda-forge 'libgfortran4=7.5.0' -y;

# Update: the above line no longer works on its own either. We have to install directly. URL pulled from https://anaconda.org/conda-forge/libgfortran4/files:
mamba install 'https://anaconda.org/conda-forge/libgfortran4/7.5.0/download/linux-64/libgfortran4-7.5.0-h14aa051_20.tar.bz2'


# Check again
which nvcc # ~/anaconda3/envs/cuda-env-py39/bin/nvcc
nvcc --version # Cuda compilation tools, release 11.8, V11.8.89
```

Add this to `~/.bashrc`:

```bash
# Turn off Jax GPU memory preallocation
# Without this, Jax fills up the GPU completely and isn't cleared. Then any use of non-Jax GPU libraries will fail because GPU memory is full.
# See https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html and https://github.com/google/jax/issues/1222
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Solve Jax CUDA problem:
# external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_asm_compiler.cc:57] cuLinkAddData fails. This is usually caused by stale driver version.
# external/org_tensorflow/tensorflow/compiler/xla/service/gpu/gpu_compiler.cc:1320] The CUDA linking API did not work. Please use XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 to bypass it, but expect to get longer compilation time due to the lack of multi-threading.
# jaxlib.xla_extension.XlaRuntimeError: UNKNOWN: no kernel image is available for execution on the device
# see https://github.com/google/jax/issues/5723
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"
```

Apply:

```bash
source ~/.bashrc
conda activate cuda-env-py39
```

### Production: CPU-only conda environment (alternative)

This is for alternative production environments without GPUs. We still use conda though, to avoid reliance on system-wide dependencies. Otherwise it's basically the same as the local development environment below.

```bash
# see instructions above to install mamba
mamba create -n cuda-env-py39 -c conda-forge python=3.9 gcc gxx gfortran -y;

conda activate cuda-env-py39;

# see notes above about why this command is necessary:
mamba install 'https://anaconda.org/conda-forge/libgfortran4/7.5.0/download/linux-64/libgfortran4-7.5.0-h14aa051_20.tar.bz2';
```

### Local development: CPU-only virtualenv through pyenv

```bash
brew update && brew install pyenv; # on Mac

pyenv install 3.9
pyenv global 3.9
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

echo 'eval "$(pyenv init -)"' >> ~/.zshrc
# eval "$(pyenv virtualenv-init -)"
# the above makes every shell command very slow
# instead, try this (https://github.com/pyenv/pyenv-virtualenv/issues/259#issuecomment-1007432346):
echo 'eval "$(pyenv virtualenv-init - | sed s/precmd/chpwd/g)"' >> ~/.zshrc
exec "$SHELL" # restart shell

pyenv virtualenv 3.9 malid-3.9
echo 'malid-3.9' > .python-version
pyenv version # malid-3.9
```

## Installation, part two: core requirements

Run this on both CPU and GPU environments -- i.e. regardless of whether you used Conda or Pyenv above.

**If using CPU, uncomment the `requirements_cpu.txt` line, and comment out the `requirements_gpu.txt` line.**

```bash
which pip # make sure this points where you expect it: either your conda or pyenv environment
pip install --upgrade pip wheel

# General requirements (clone github repos into a sibling directory):
rm -r ../cuda-env-py39-pkgs
mkdir -p ../cuda-env-py39-pkgs
pip install -r requirements.txt --src ../cuda-env-py39-pkgs

# GPU specific requirements (skip if CPU only):
pip install -r requirements_gpu.txt --src ../cuda-env-py39-pkgs

# Or if running on CPU only:
# pip install -r requirements_cpu.txt --src ../cuda-env-py39-pkgs

# Further general requirements that rely on the CPU/GPU-specific requirements being installed:
pip install -r requirements_extra.txt --src ../cuda-env-py39-pkgs

pip check
# In the preferred production GPU environment, we expect:
# cudf 23.6.1 requires cupy-cuda11x>=12.0.0, which is not installed.
# cugraph 23.6.2 requires cupy-cuda11x>=12.0.0, which is not installed.
# cuml 23.6.0 requires cupy-cuda11x>=12.0.0, which is not installed.
# dask-cudf 23.6.1 requires cupy-cuda11x>=12.0.0, which is not installed.
# referencing 0.30.0 requires attrs>=22.2.0, but you have attrs 21.4.0 which is incompatible.

# The first four are expected errors (cupy 12.0 is already installed via conda, but pip isn't aware of that), but the attrs error is unexpected -- TODO: look into this.

# Add kernel to system-wide jupyter notebook
ipython kernel install --name py39-cuda-env --user
# Installed kernelspec py39-cuda-env in ~/.local/share/jupyter/kernels/py39-cuda-env
```

## Installation, part three: Mal-ID package

```bash
# Setup data dir
# on cluster:
ln -s /mnt/lab_data/kundaje/projects/malid data
# or locally:
# mkdir -p data

# Download PyIR reference data
# The first time we do this, we can fetch the data from IMGT - which is slow and has frequent timeouts:
# pyir setup
# rm -r data/pyir_cache
# mkdir -p data/pyir_cache/germlines
# cp -r "$(python -c 'import site; print(site.getsitepackages()[0])')/crowelab_pyir/data" data/pyir_cache

# But in practice, let's copy our saved version of the PyIR reference data
# (make sure to have the conda environment cuda-env-py39 active here still)
echo "$(python -c 'import crowelab_pyir; from pathlib import Path; print(Path(crowelab_pyir.__file__).parent / "data")')"
cp -r data/pyir_cache/germlines "$(python -c 'import crowelab_pyir; from pathlib import Path; print(Path(crowelab_pyir.__file__).parent / "data")')"

# Install fonts on cluster
# (make sure to have the conda environment cuda-env-py39 active here still)
echo "$(python -c 'import matplotlib; print(matplotlib.get_data_path())')/fonts/ttf"
cp data/fonts/*.ttf "$(python -c 'import matplotlib; print(matplotlib.get_data_path())')/fonts/ttf"
python -c 'import matplotlib.font_manager; matplotlib.font_manager._load_fontmanager(try_read_cache=False)'

# Install local package
pip install -e .

# Run all tests
pytest tests/test_dependencies.py # confirm core dependencies properly installed
pytest
pytest --gpu # unless CPU only

# Install pre-commit (see Development section below)
conda deactivate
pip install pre-commit
pre-commit install

# Snapshot resulting conda environment (unless CPU virtualenv):
conda env export --name cuda-env-py39 --no-builds | grep -v "prefix" > cudaenv.conda.yml

# If needed, here is how to delete an env we no longer use:
# conda deactivate
# rm -r ~/anaconda3/envs/cuda-env-py39
```

If we have upgraded from an older version by creating a new environment, change all notebooks and scripts to use that kernel (note this will also modify this readme):

```bash
## Edit with recursive sed:
# jupyter (kernel name)
grep -rl "py37-cuda-env" . --exclude-dir=.git | xargs sed -i "" -e 's/py37-cuda-env/py39-cuda-env/g'

# scripts (conda env name)
grep -rl "cuda-env-py37" . --exclude-dir=.git | xargs sed -i "" -e 's/cuda-env-py37/cuda-env-py39/g'
```

## Installation, optional extra: install R kernel too

```bash
conda create -n r-v41
conda activate r-v41

mamba install -c conda-forge r-recommended=4.1 r-irkernel=1.3 -y;
R --version # 4.1.3
Rscript --version # 4.1.3

mamba install -c conda-forge Jupyter -y;
# Fix zmq import error:
pip uninstall pyzmq
pip install pyzmq
# Register kernel
R -e 'IRkernel::installspec(name = "ir41", displayname = "R 4.1", user = TRUE)'
# Note remaining pip error: "jupyter 1.0.0 requires qtconsole, which is not installed."

# Install packages
R -e 'install.packages("tidyverse",repos = "http://cran.us.r-project.org")'
# ERROR: dependencies ‘googledrive’, ‘googlesheets4’, ‘reprex’ are not available for package ‘tidyverse’
R -e 'install.packages("gridExtra", repos = "http://cran.us.r-project.org")'

conda deactivate
jupyter kernelspec list
#   ir41                 /users/maximz/.local/share/jupyter/kernels/ir41
```


## Environment setup and everyday use

```bash
conda activate cuda-env-py39

## first time setup:
pre-commit install
# install package
pip install -e .
# only test that core dependencies are installed
pytest tests/test_dependencies.py

## everyday use:

# lint all files
make lint
# or lint staged files only
make lint-staged
# or lint files that have changed since upstream only
make lint-diff

# Run all tests with GPU disabled, skipping any tests marked with @pytest.mark.gpu
pytest # or simply: make test

# Run all tests with GPU enabled, including any tests marked with @pytest.mark.gpu, but skipping any tests marked @pytest.mark.skip_if_gpu.
pytest --gpu

# You can further narrow down the tests to CPU and fast ones only (for CI):
pytest -m "not slow"

# recreate all jupytext generated scripts
make regen-jupytext

# add this to .bashrc to log to Sentry:
# export SENTRY_API_KEY='https://REDACTED.ingest.sentry.io/REDACTED'
```

Logs are written to `paths.log_dir` from `config.py`.

When writing tests, don't call `config.configure_gpu()` manually. That's handled automatically by the test setup logic in `conftest.py` based on whether you run `pytest` or `pytest --gpu`.

Any tests that require GPU should be decorated with `@pytest.mark.gpu`. Any tests that are to be run with CPU only, and skipped if GPU is active, should be decorated with `@pytest.mark.skip_if_gpu`.

### Jupytext mirroring

Using jupytext, every `.ipynb` notebook in `notebooks/` has a paired `.py` script in `notebooks_src/`. This makes diffs much cleaner (nbdiff/nbdime is too slow and finnicky to be practical) and allows for bulk refactoring.

When you edit and save the `.ipynb` notebook in Jupyter Lab, Jupytext updates the `.py` paired script automatically. And when you edit the `.py` script in a text editor, reloading the paired `.ipynb` notebook in Jupyter Lab will sync the updates to the notebook.

Sometimes we fall back to the command line to sync these notebook/script pairs: after editing either element of the pair, you can `git add` it to the git staging area and then `make lint-staged` to run jupytext-sync (and other pre-commit hooks) on all staged files. We tend to do this after bulk refactors of `.py` scripts (`make lint-staged` will update the paired notebooks without having to open each one in Jupyter Lab) or after auto-generating notebooks from other template notebooks with `run_notebook_to_new_file.sh` (`make lint-staged` will then generate or update the paired script.)

### Best practices

I tend to work in feature branches with pull requests against master, until they're mature and pass all tests.

I tend to make microcommits with `git commit --fixup HEAD` (or use a commit hash from `git log` instead of `HEAD`).

When a branch is ready to merge, I squash those fixup commits into their parent:

```bash
git push origin my-branch-name # store a copy on origin (github) to be safe
git rebase -i origin/master --autosquash
git diff origin/my-branch-name..HEAD # should be empty / silent - confirm the rebase didn't change anything
```

To pull new changes from master into your feature branch:

```bash
git push origin my-branch-name # to be safe, backup your branch on origin (github)
git fetch --all
git rebase -i origin/master
```

You can analyze what the rebase did with commands like:

```bash
# compare [old master to old HEAD] to [new master to new HEAD]
diff <(git log master..origin/my-branch-name -p) <(git log origin/master..HEAD -p) | less
# or
git range-diff @{u} @{1} @
```

When ready: `git push --force origin my-branch-name` to save your changes to the Github copy too.

Monitor Github Actions CI jobs:

```bash
gh run list
gh run watch

# Sometimes merging to master sets off a cascade of Dependabot PR Github Actions runs.
# Here's how to cancel them all:
gh run list --json databaseId  -q '.[].databaseId' | xargs -IID gh run cancel ID
```

### Jupyter Lab extensions

Several extensions have automatically be installed through pip: Jupytext (see above) and the Jupyter Lab code formatter, which creates a toolbar button that runs Black on your notebook.

Some extra manual configuration is required for the code formatter extension:

In the Jupyter Lab menu bar, go to: "Settings" > "Settings Editor" > "JSON Settings Editor" (top right) > "Jupyterlab Code Formatter". Under "User Preferences", add:

```json
{
    "preferences": {
        "default_formatter": {
            "python": "black",
            "R": "styler"
        }
    }
}
```

## Configuration

### Modeling feature flags

In `config.py`:

- `_default_dataset_version` and `_default_cross_validation_split_strategy`: which data to use
- `_default_embedder`: language model to use. Or override with `EMBEDDER_TYPE` environment variable, which accepts `name` property values from any embedder.
- `_default_classification_targets`
- `sequence_identity_thresholds` (for model2)
- `gene_loci_used` (include BCR, TCR, or both)
- `metamodel_base_model_names` (base model versions used in metamodel)

### Configure file paths

View current path configuration: `./config.py`

Update `config.py`. For each file path, `rm -r FILEPATH` it if we are regenerating.

To create all the necessary directories (including intermediates -- this is like `mkdir -p`): run `python scripts/make_dirs.py`

### Configure diseases

To add new diseases, modify:

- `assemble_etl_metadata.ipynb`
- `etl.ipynb`
- `datamodels.py`'s disease list
- `datamodels.CrossValidationSplitStrategy`'s `diseases_to_keep_all_subtypes` and `subtypes_keep`
- `datamodels.TargetObsColumnEnum` and `config.classification_targets`
- `rm -r out && mkdir -p out`
- `config.py`'s `dataset_version`, then run `python scripts/make_dirs.py`

## Design

### Sequence data

Raw sequences are stored in Parquet format. We must be consistent about which conserved prefixes/suffixes to the CDR regions are kept or removed. We use IgBlast-formatted CDR1/2/3 regions from IMGT numbering. V3-30 example:

- CDR1: `GFTFSSYG`
- CDR2: `ISYDGSNK`
- CDR3: `ARDGIVGATGLDY`, `AKEMFKYGSGVSSDGFDV`, etc. Notice the leading `C` and trailing `W` are removed.

Our TCR data is missing CDR1 and CDR2 annotations. But these are deterministic since TCRs don't have somatic hypermutations. We fill in the CDR1 and CDR2 regions using downloaded reference sequences.

### Embeddings

Embedded sequence vectors coming from a language model are stored in AnnData objects as `.X`, with metadata in `.obs`.

BCR and TCR sequences are stored together in Parquet, but pass through separate language model embeddings and land in separate AnnData objects. This is because the two types of sequences are selected to bind to different targets (antibodies can bind almost any epitope structure, whereas TCRs bind peptides in the context of MHC). We expect correlations within BCRs and correlations within TCRs, not between the two.

These AnnData objects are further divided by cross validation folds: `fold_id` can be 0, 1, or 2, and `fold_label` can be `train_smaller` (also further divided into `train_smaller1` and `train_smaller2`), `validation`, or `test`. There's also a special `fold_id=-1` "global" fold that does not have a `test` set. The data is instead used in the other two fold labels; the `train_smaller` to `validation` proportion is the same as for other fold IDs, but both sets are larger than usual.

### Models

We train all the models using the AnnData objects divided into cross validation folds. (We do this so that all models get exactly the same sets of sequences - even though some models will ignore `.X` and only use `.obs`.)

## Runbook, for a single embedder

These environment variables are available to override the defaults in `config.py`:

- `MALID_DATASET_VERSION`
- `MALID_CV_SPLIT`
- `EMBEDDER_TYPE`
- `MALID_DISABLE_IN_MEMORY_CACHE` (disables in-memory LRU cache of sequence embedding anndatas)

### ETL

(These ETL steps are not embedder-specific.)

Add Boydlab participant tables directory paths to `notebooks/etl.ipynb`.

Run the instructions in `adaptive_runbook.md` and `notebooks/airr_external_data/readme.md`. Confirm that the CDR1/2/3 format matches our data, e.g. no prefix and suffix in CDR3.

```bash
# rm -r and mkdir -p the config.py paths

conda activate cuda-env-py39;

python scripts/make_dirs.py

## Preparation
# extract TCR-B reference annotations
python scripts/get_tcr_v_gene_annotations.py;

## ETL: Boydlab data
# ETL output is not embedding type-specific, so it's just written to `out/`, rather than an embedding type-specific subdirectory.

./run_notebooks.sh \
    notebooks/assemble_etl_metadata.ipynb \
    notebooks/airr_external_data/covid_tcr_immunecode_metadata.ipynb \
    notebooks/airr_external_data/adaptive_cohorts.metadata.ipynb \
    notebooks/airr_external_data/metadata_per_patient.ipynb \
    notebooks/airr_external_data/metadata_shomuradova.ipynb \
    notebooks/airr_external_data/metadata_britanova.ipynb \
    notebooks/airr_external_data/combine_external_cohort_metadata.ipynb \
    notebooks/etl.ipynb \
    notebooks/participant_specimen_disease_map.ipynb \
    notebooks/read_counts_per_specimen.ipynb \
    notebooks/get_all_v_genes.ipynb; # generates a list of all observed V genes in our dataset so we can add V gene dummy variables to our models

# Review out/reads_per_specimen_and_isotype.tsv carefully.

############

# Narrow down from "all data" to "data that has survived quality filters and has been subsampled".

# Apply QC filters and downsample to roughly one sequence per clone per specimen.
# (It's not exactly that. See this notebook malid/sample_sequences.py for details.)
./run_notebooks.sh notebooks/sample_sequences.ipynb;

# Review log messages carefully: grep 'malid.sample_sequences' notebooks/sample_sequences.ipynb
```

### Narrow down to data for a particular cross validation split strategy

All sections from here on are specific to the currently active cross validation split strategy, configured in `malid/config.py`. This determines which samples are included in the cross-validation training and testing sets.

Cross valdiation split strategies include:

- `CrossValidationSplitStrategy.in_house_peak_disease_timepoints`: default, 3 folds
- `CrossValidationSplitStrategy.in_house_peak_disease_leave_one_cohort_out`: one single fold with specific study names in the hold-out test set
- `CrossValidationSplitStrategy.adaptive_peak_disease_timepoints`: 3 folds, like the default but for Adaptive cohorts (TCR only)
- `CrossValidationSplitStrategy.adaptive_peak_disease_timepoints_leave_some_cohorts_out`: one single fold with specific study names in the training set and the hold-out test set

### Make cross validation folds

Now, we will narrow down from "data that has survived quality filters and has been subsampled", to "data that has been further selected for the currently selected cross validation split strategy", which is defined in `malid/config.py`.

```bash
# Make CV splits, subsetting to specimens for the currently selected cross validation split strategy, such as "in-house samples at peak disease timepoints only".
# This will create train/validation/test splits, along with a global fold that does not have a test set. (It's actually a bit more granular than that; see this notebook for more details.)
./run_notebooks.sh notebooks/make_cv_folds.ipynb;
```

Review `out/$CROSS_VALIDATION_SPLIT_STRATEGY_NAME/all_data_combined.participants_by_disease_subtype.tsv` carefully.

### Prepare for embedding

Review embedder selection in malid/config.py. It determines the base model and the sequence regions to be embedded (e.g. CDR1+2+3).

Then make the directories: `python scripts/make_dirs.py`

### Fine-tune the language model, if enabled in `config.py`

This will use the default embedder configured in `malid/config.py`.

Fine tune a language model separately for each fold, using a subset of train-smaller set (choosing the training epoch with best performance against a subset of validation set).

```bash
## Run fine-tuning on all folds and gene loci:
# python scripts/fine_tune_language_model.py
# You can also override --num_epochs, which defaults to 40.

# Or, split up between machines:
python scripts/fine_tune_language_model.py --locus BCR --fold_id 0 2>&1 | tee "data/logs/fine_tune_language_model.fold0.bcr.log"
python scripts/fine_tune_language_model.py --locus TCR --fold_id 0 2>&1 | tee "data/logs/fine_tune_language_model.fold0.tcr.log"
#
python scripts/fine_tune_language_model.py --locus BCR --fold_id 1 2>&1 | tee "data/logs/fine_tune_language_model.fold1.bcr.log"
python scripts/fine_tune_language_model.py --locus TCR --fold_id 1 2>&1 | tee "data/logs/fine_tune_language_model.fold1.tcr.log"
#
python scripts/fine_tune_language_model.py --locus BCR --fold_id 2 2>&1 | tee "data/logs/fine_tune_language_model.fold2.bcr.log"
python scripts/fine_tune_language_model.py --locus TCR --fold_id 2 2>&1 | tee "data/logs/fine_tune_language_model.fold2.tcr.log"

python scripts/fine_tune_language_model.py --locus BCR --fold_id -1 2>&1 | tee "data/logs/fine_tune_language_model.globalfold.bcr.log"
python scripts/fine_tune_language_model.py --locus TCR --fold_id -1 2>&1 | tee "data/logs/fine_tune_language_model.globalfold.tcr.log"

# Some embedders can be monitored with Tensorboard:
tensorboard --logdir="$(python -c 'import malid.config; print(malid.config.paths.fine_tuned_embedding_dir)')" --port=$PORT;

# Extract parameters at epoch with best validation loss:
./run_notebooks.sh notebooks/fine_tune.analyze_over_epochs.ipynb;

# Fetch uniref 50 raw data
wget -O - https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/README # print readme to describe uniref data
wget -O ./data/uniref50.fasta.gz  https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz # download ref data
! zcat ./data/uniref50.fasta.gz | echo $((`wc -l`/4)) # 83360861 total sequences

# Assess perplexity/cross validation loss on uniref50 + TCR/BCR sequences
./run_notebooks.sh notebooks/catastrophic_forgetting.ipynb;
```

### Run off-the-shelf or fine-tund language model

Create embeddings with the configured language model, then scale the resulting anndatas:

```bash
# Run on all folds and loci:
# python scripts/run_embedding.py
# python scripts/scale_embedding_anndatas.py

# OR split up between machines:
python scripts/run_embedding.py --locus BCR --fold_id 0 2>&1 | tee "data/logs/run_embedding.fine_tuned.fold0.bcr.log"
python scripts/scale_embedding_anndatas.py --locus BCR --fold_id 0 2>&1 | tee "data/logs/scale_anndatas_created_with_finetuned_embedding.fold0.bcr.log"

python scripts/run_embedding.py --locus TCR --fold_id 0 2>&1 | tee "data/logs/run_embedding.fine_tuned.fold0.tcr.log"
python scripts/scale_embedding_anndatas.py --locus TCR --fold_id 0 2>&1 | tee "data/logs/scale_anndatas_created_with_finetuned_embedding.fold0.tcr.log"

#

python scripts/run_embedding.py --locus BCR --fold_id 1 2>&1 | tee "data/logs/run_embedding.fine_tuned.fold1.bcr.log"
python scripts/scale_embedding_anndatas.py --locus BCR --fold_id 1 2>&1 | tee "data/logs/scale_anndatas_created_with_finetuned_embedding.fold1.bcr.log"

python scripts/run_embedding.py --locus TCR --fold_id 1 2>&1 | tee "data/logs/run_embedding.fine_tuned.fold1.tcr.log"
python scripts/scale_embedding_anndatas.py --locus TCR --fold_id 1 2>&1 | tee "data/logs/scale_anndatas_created_with_finetuned_embedding.fold1.tcr.log"

#

python scripts/run_embedding.py --locus BCR --fold_id 2 2>&1 | tee "data/logs/run_embedding.fine_tuned.fold2.bcr.log"
python scripts/scale_embedding_anndatas.py --locus BCR --fold_id 2 2>&1 | tee "data/logs/scale_anndatas_created_with_finetuned_embedding.fold2.bcr.log"

python scripts/run_embedding.py --locus TCR --fold_id 2 2>&1 | tee "data/logs/run_embedding.fine_tuned.fold2.tcr.log"
python scripts/scale_embedding_anndatas.py --locus TCR --fold_id 2 2>&1 | tee "data/logs/scale_anndatas_created_with_finetuned_embedding.fold2.tcr.log"

#

python scripts/run_embedding.py --locus BCR --fold_id -1 2>&1 | tee "data/logs/run_embedding.fine_tuned.globalfold.bcr.log"
python scripts/scale_embedding_anndatas.py --locus BCR --fold_id -1 2>&1 | tee "data/logs/scale_anndatas_created_with_finetuned_embedding.globalfold.bcr.log"

python scripts/run_embedding.py --locus TCR --fold_id -1 2>&1 | tee "data/logs/run_embedding.fine_tuned.globalfold.tcr.log"
python scripts/scale_embedding_anndatas.py --locus TCR --fold_id -1 2>&1 | tee "data/logs/scale_anndatas_created_with_finetuned_embedding.globalfold.tcr.log"
```

### Train and analyze base models

```bash
## Model 1: Repertoire statistics classifiers:
# Compute repertoire-level statistics, then run classifiers on them, using same fold splits as in modeling above.
# Train new repertoire stats model on train_smaller too, first evaluating on validation set, then evaluating on test set (with and without tuning on validation set)
# Feature standardization is built-in.
python scripts/train_repertoire_stats_models.py 2>&1 | tee "data/logs/train_repertoire_stats_models.log";
./run_notebooks.sh \
    notebooks/analyze_repertoire_stats_models.ipynb \
    notebooks/repertoire_stats_classifiers.tune_model_decision_thresholds_on_validation_set.ipynb \
    notebooks/summary.repertoire_stats_classifiers.ipynb \
    notebooks/interpret_model1.ipynb;

###########

## Model 2: Convergent sequence cluster classifiers for disease classification
# Run convergent clustering classifiers on train-smaller too, so that comparable to our models above, along with tuning on validation set.
# train on all folds, loci, and targets
# or pass in specific settings, see: python scripts/train_convergent_clustering_models.py --help
# for example: python scripts/train_convergent_clustering_models.py --target_obs_column "disease" --n_jobs 8
# Note that --n_jobs here corresponds to the parallelization level for p-value threshold tuning.
# For larger datasets, reduce the n_jobs parallelization level to reduce memory pressure.
python scripts/train_convergent_clustering_models.py --n_jobs 8 2>&1 | tee "data/logs/train_convergent_clustering_models.log";
./run_notebooks.sh \
    notebooks/analyze_convergent_clustering_models.ipynb \
    notebooks/convergent_clustering_models.tune_model_decision_thresholds_on_validation_set.ipynb \
    notebooks/summary.convergent_sequence_clusters.ipynb;

###########

## Benchmark: Exact matches classifier.
# Train on disease task only.
# Note that --n_jobs here corresponds to the parallelization level for p-value threshold tuning.
# For larger datasets, reduce the n_jobs parallelization level to reduce memory pressure.
python scripts/train_exact_matches_models.py \
    --target_obs_column "disease" \
    --n_jobs 8 2>&1 | tee "data/logs/train_exact_matches_models.log";
./run_notebooks.sh notebooks/analyze_exact_matches_models.ipynb;

###########

## Model 3:
# Our example code below is for "disease" target only.
# We want to run this for all target_obs_columns. We usually split into separate jobs.
# You could construct a loop in bash like so:
# for TARGET_OBS_COLUMN in $(python scripts/target_obs_column_names.py)
# do
#     echo "python scripts/my_command.py --target_obs_column \"$TARGET_OBS_COLUMN\" --other-parameters 2>&1 | tee \"data/logs/my_command.$TARGET_OBS_COLUMN.log\""
# done
# unset TARGET_OBS_COLUMN;

# Here's how these scripts work:
# - If no `--target_obs_column` is supplied, it trains on all target_obs_columns one by one. You can supply multiple `--target_obs_column` values.
# - Same for `--fold_id` and `--locus`.

# Train sequence-level models split by V gene and isotype.
# Note that --n_jobs here corresponds to the parallelization level across V-J gene pair subsets.
# For larger datasets, reduce the n_jobs parallelization level to reduce memory pressure.
# This command can also be separated by locus and fold using command line arguments --fold_id and --locus.
# Add --resume to recover from a failed run.
# Add --train-preferred-model-only to train only the preferred model name used downstream in the metamodel. Pairs with the --use-preferred-base-model-only option for scripts/train_vj_gene_specific_sequence_model_rollup.py.
python scripts/train_vj_gene_specific_sequence_model.py \
    --target_obs_column "disease" \
    --n_jobs 8 \
    --sequence-subset-strategy split_Vgene_and_isotype \
    2>&1 | tee "data/logs/train_vj_gene_specific_sequence_model.disease.split_Vgene_and_isotype.log";
# Add --fold_id or --locus (multiple usage allowed) to further split the work across machines.

# Monitor how many split_keys have been started by parsing the JSON logs:
# cat data/logs/train_vj_gene_specific_sequence_model.disease.split_Vgene_and_isotype.log | grep '^\{"split_key' | jq -r '.split_key' --compact-output | sort | uniq -c | wc -l

# Train rollup model that aggregates sequence scores to the patient (really the specimen) level.
# This will produce a separate rollup model trained on top of each base model name that was configured in scripts/train_vj_gene_specific_sequence_model.py
# Add --use-preferred-base-model-only to train only against the preferred base (sequence-level) model name used downstream in the metamodel. Pairs with the --train-preferred-model-only option for scripts/train_vj_gene_specific_sequence_model.py.
python scripts/train_vj_gene_specific_sequence_model_rollup.py \
    --target_obs_column "disease" \
    --sequence-subset-strategy split_Vgene_and_isotype \
    2>&1 | tee "data/logs/train_vj_gene_specific_sequence_model_rollup.disease.split_Vgene_and_isotype.log";

# Analyze aggregation stage model
SEQUENCE_SUBSET_STRATEGY=split_Vgene_and_isotype ./run_notebook_to_new_file.sh notebooks/analyze_vj_gene_specific_sequence_model_rollup_classifier.ipynb notebooks/analyze_subset_specific_sequence_model_rollup_classifier.split_Vgene_and_isotype.generated.ipynb;

# SHAP analysis of the aggregation stage model
# Note: this notebook only supports Vgene-isotype splits, not other SEQUENCE_SUBSET_STRATEGY values:
./run_notebooks.sh notebooks/model3_aggregation_feature_importances.ipynb;
```

Note that we have chopped up the training set further into `train_smaller1` and `train_smaller2`. One part is used to train the base sequence model (a wrapper around a bunch of V-J gene specific sequence models). The other part is used to train the model to aggregate mean-sequence-scores-by-Vgene-and-isotype to the patient/specimen level. That aggregation model is then evaluated on the validation set. It would be a mistake to train the second stage aggregation model on the same dataset as we trained the base sequence model: the sequence classifiers would emit unrealistic predicted probabilities. We want to expose the aggregation model to what probabilities will look like on held out data, so that it generalizes well.

### Train and analyze ensemble metamodel using base models trained above

```bash
# train on all folds, loci, and targets
# or pass in specific settings, see: python scripts/train_metamodels.py --help
# for example: python scripts/train_metamodels.py --target_obs_column "disease"
python scripts/train_metamodels.py 2>&1 | tee "data/logs/train_metamodels.log";
# review the log for "Failed to create 'default' or dependent metamodel flavor" or similar errors, which means the metamodel training failed.

# Analyze:
./run_notebooks.sh \
    notebooks/analyze_metamodels.ipynb \
    notebooks/plot_metamodel_per_class_aucs.ipynb \
    notebooks/summary.metamodels.ipynb \
    notebooks/summary.metamodels.per_class_scores.ipynb \
    notebooks/summary.metamodels.succinct.ipynb;

# Overall summary
./run_notebooks.sh notebooks/summary.ipynb;

# Extract lupus vs rest; plot its specificity vs sensitivity
./run_notebooks.sh notebooks/sensitivity_specificity_lupus.ipynb;
```

### Analyze highly-ranked sequences and intersect with Cov-AbDab / MIRA

First process known binder datasets with our IgBlast version.

#### CoV-AbDab IgBlast

Unlike other datasets, we have amino acid entries (`VHorVHH` column). We need to switch to `igblastp` instead of the usual `igblastn`. The version of IgBlast we use predates the introduction of the AIRR output format, and the legacy parser doesn't support `igblastp`, so we have to roll our own parser too. Ultimately we get V gene calls and CDR1+2 sequences out of this, but use the original CDR3 sequence and J gene call from CoV-AbDab.

```bash
# Choose sequences of interest
./run_notebooks.sh notebooks/cov_abdab.ipynb;

# Export to fasta
python scripts/export_sequences_to_fasta.py \
    --input "data/CoV-AbDab_130623.filtered.tsv" \
    --output "data/CoV-AbDab_130623.filtered.fasta" \
    --name "covabdab" \
    --separator $'\t' \
    --column "VHorVHH";

# Chunk the fasta files.
# CoV-AbDab_130623.filtered.fasta --> CoV-AbDab_130623.filtered.fasta.part_001.fasta
seqkit split2 "data/CoV-AbDab_130623.filtered.fasta" -O "data/cov_abdab_fasta_split" --by-size 10000 --by-size-prefix "CoV-AbDab_130623.filtered.fasta.part_"

# Install igblastp
# Download it from https://ftp.ncbi.nih.gov/blast/executables/igblast/release/1.3.0/ncbi-igblast-1.3.0-x64-linux.tar.gz

# Result:
$HOME/boydlab/igblast/igblastp -version
# igblastp: 1.0.0
# Package: igblast 1.3.0, build Mar 11 2014 10:16:49

# Matches version of:
$HOME/boydlab/igblast/igblastn -version
# igblastn: 1.0.0
# Package: igblast 1.3.0, build Mar 26 2014 14:46:28

# Run igblast
# data/cov_abdab_fasta_split/CoV-AbDab_130623.filtered.fasta.part_001.fasta -> data/cov_abdab_fasta_split/CoV-AbDab_130623.filtered.fasta.part_001.fasta.parse.txt
tmpdir_igblast=$(mktemp -d)
echo "$tmpdir_igblast"
pushd "$tmpdir_igblast"
cp $HOME/boydlab/pipeline/run_igblastp_igh.sh .;
cp $HOME/boydlab/igblast/human_gl* .;
cp -r $HOME/boydlab/igblast/internal_data/ .;

num_processors=200 # 55

# use -print0 and -0 to handle spaces in filenames
# _ is a dummy value for $0 (the script name)
# $1 in the sh -c command will be the filename
find $HOME/code/immune-repertoire-classification/data/cov_abdab_fasta_split/ -name "*.part_*.fasta" -print0 | xargs -0 -I {} -n 1 -P "$num_processors" sh -c './run_igblastp_igh.sh "$1"' _ {}
echo $? # exit code

popd
echo "$tmpdir_igblast"
rm -r "$tmpdir_igblast"

# Monitor: these numbers must match
find data/cov_abdab_fasta_split/ -name "*.part_*.fasta" | wc -l
find data/cov_abdab_fasta_split/ -name "*.part_*.fasta.parse.txt" | wc -l


# Back to our python environment
for fname in data/cov_abdab_fasta_split/*.part_*.fasta; do
    echo $fname;
    python scripts/parse_igblastp.py \
        --fasta "$fname" \
        --parse "$fname.parse.txt" \
        --output "$fname.parsed.tsv" \
        --separator $'\t';
done
echo $?

# Monitor: these numbers must match
find data/cov_abdab_fasta_split/ -name "*.part_*.fasta.parse.txt" | wc -l
find data/cov_abdab_fasta_split/ -name "*.part_*.fasta.parsed.tsv" | wc -l


# Merge IgBlast parsed output to the original CoV-AbDab data
./run_notebooks.sh notebooks/covabdab_add_igblast_annotations.ipynb;
```

#### MIRA IgBlast

```bash
python scripts/export_sequences_to_fasta.py \
    --input "data/external_cohorts/raw_data/immunecode_all/mira/ImmuneCODE-MIRA-Release002.1/peptide-detail-ci.csv" \
    --output "data/external_cohorts/raw_data/immunecode_all/mira/ImmuneCODE-MIRA-Release002.1/peptide-detail-ci.fasta" \
    --name "peptide-detail-ci" \
    --column "TCR Nucleotide Sequence";

python scripts/export_sequences_to_fasta.py \
    --input "data/external_cohorts/raw_data/immunecode_all/mira/ImmuneCODE-MIRA-Release002.1/peptide-detail-cii.csv" \
    --output "data/external_cohorts/raw_data/immunecode_all/mira/ImmuneCODE-MIRA-Release002.1/peptide-detail-cii.fasta" \
    --name "peptide-detail-cii" \
    --column "TCR Nucleotide Sequence";

python scripts/export_sequences_to_fasta.py \
    --input "data/external_cohorts/raw_data/immunecode_all/mira/ImmuneCODE-MIRA-Release002.1/minigene-detail.csv" \
    --output "data/external_cohorts/raw_data/immunecode_all/mira/ImmuneCODE-MIRA-Release002.1/minigene-detail.fasta" \
    --name "minigene-detail" \
    --column "TCR Nucleotide Sequence";

# Chunk the fasta files.
# Use gnu split command because we know the fastas we generated always have each sequence on a single line, i.e. number of lines is divisible by two (not all fasta are like this!)
# TODO: Update this to use same seqkit split as above.
cd data/external_cohorts/raw_data/immunecode_all/mira/ImmuneCODE-MIRA-Release002.1
rm -r splits
mkdir -p splits
for fname in *.fasta; do
  split -l 10000 --verbose --numeric-suffixes=1 --suffix-length=10 --additional-suffix=".fasta" "$fname" "splits/$fname.part"
done

# Run igblast
cp $HOME/boydlab/pipeline/run_igblast_command_tcr.sh .;
cp $HOME/boydlab/igblast/human_gl* .;
cp -r $HOME/boydlab/igblast/internal_data/ .;
find splits -name "*.part*.fasta" | xargs -I {} -n 1 -P 55 sh -c "./run_igblast_command_tcr.sh {}"

# Monitor
find splits -name "*.part*.fasta" | wc -l
find splits -name "*.parse.txt" | wc -l

# Parse to file
conda deactivate
source ~/boydlab/pyenv/activate
# $HOME/boydlab/pipeline/load_igblast_parse.ireceptor_data.to_file.py --locus TCRB splits/*.parse.txt
# parallelize in chunk size of 50 parses x 40 processes:
find splits -name "*.parse.txt" | xargs -x -n 50 -P 40 $HOME/boydlab/pipeline/load_igblast_parse.ireceptor_data.to_file.py --locus "TCRB"

# Monitor
find splits -name "*.parse.txt" | wc -l
find splits -name "*.parse.txt.parsed.tsv" | wc -l
```

Merge IgBlast parsed output to the original data (for MIRA) and subset to sequences of interest:

```bash
./run_notebooks.sh notebooks/mira.ipynb;
```

#### Flu known binders

```bash
# choose sequences of interest
./run_notebooks.sh notebooks/flu_known_binders.ipynb;

# Export to fasta
python scripts/export_sequences_to_fasta.py \
    --input "data/flu_known_binders.filtered.tsv" \
    --output "data/flu_known_binders.filtered.fasta" \
    --name "flu" \
    --separator $'\t' \
    --column "VH_nuc";

# Chunk the fasta files.
# flu_known_binders.filtered.fasta --> flu_known_binders.filtered.fasta.part_001.fasta
seqkit split2 "data/flu_known_binders.filtered.fasta" -O "data/flu_known_binders_fasta_split" --by-size 10000 --by-size-prefix "flu_known_binders.filtered.fasta.part_"

# Run igblast
# data/flu_known_binders_fasta_split/flu_known_binders.filtered.fasta.part_001.fasta -> data/flu_known_binders_fasta_split/flu_known_binders.filtered.fasta.part_001.fasta.parse.txt
tmpdir_igblast=$(mktemp -d)
echo "$tmpdir_igblast"
pushd "$tmpdir_igblast"
cp $HOME/boydlab/pipeline/run_igblast_command.sh .;
cp $HOME/boydlab/igblast/human_gl* .;
cp -r $HOME/boydlab/igblast/internal_data/ .;

num_processors=200 # 55

# use -print0 and -0 to handle spaces in filenames
# _ is a dummy value for $0 (the script name)
# $1 in the sh -c command will be the filename
find $HOME/code/immune-repertoire-classification/data/flu_known_binders_fasta_split/ -name "*.part_*.fasta" -print0 | xargs -0 -I {} -n 1 -P "$num_processors" sh -c './run_igblast_command.sh "$1"' _ {}
echo $? # exit code

popd
echo "$tmpdir_igblast"
rm -r "$tmpdir_igblast"

# Monitor: these numbers must match
find data/flu_known_binders_fasta_split/ -name "*.part_*.fasta" | wc -l
find data/flu_known_binders_fasta_split/ -name "*.part_*.fasta.parse.txt" | wc -l

# Parse to file (uses python2.7 pipeline code)
conda deactivate
source ~/boydlab/pyenv/activate
# $HOME/boydlab/pipeline/load_igblast_parse.ireceptor_data.to_file.py --locus IgH splits/*.parse.txt
# parallelize in chunk size of 50 parses x 40 processes:
num_processors=200 # 40
# use -print0 and -0 to handle spaces in filenames
find data/flu_known_binders_fasta_split/ -name "*.part_*.fasta.parse.txt" -print0 | xargs -0 -x -n 50 -P "$num_processors" $HOME/boydlab/pipeline/load_igblast_parse.ireceptor_data.to_file.py --locus "IgH"
echo $?

# Monitor: these numbers must match
find data/flu_known_binders_fasta_split/ -name "*.part_*.fasta.parse.txt" | wc -l
find data/flu_known_binders_fasta_split/ -name "*.part_*.fasta.parse.txt.parsed.IgH.tsv" | wc -l

# Merge IgBlast parsed output to the original flu binder data
./run_notebooks.sh notebooks/flu_known_binders_add_igblast_annotations.ipynb;
```

#### Run analysis

```bash
./run_notebooks.sh \
    notebooks/embed_known_binders.ipynb \
    notebooks/ranks_of_known_binders_vs_healthy_donor_sequences.ipynb;
```

### Wrap up confounders checks

```bash
# isotype_stats: generate isotype proportion counts per specimen.
./run_notebooks.sh \
    notebooks/isotype_stats.ipynb \
    notebooks/vgene_usage_stats.ipynb \
    notebooks/size_of_each_disease_batch.ipynb \
    notebooks/confounder_model.ipynb \
    notebooks/summary.confounders.ipynb;

## Measure number of overlapping sequences between fold labels
./run_notebooks.sh notebooks/compute_fold_overlaps.ipynb;

## Measure batch mixing with kBET reimplementation
./run_notebooks.sh notebooks/kbet_batch_evaluation.ipynb;
```

### Analyze lupus misclassifications

`./run_notebooks.sh notebooks/adult_lupus_misclassification_disease_activity_scores.ipynb` evaluates whether adult lupus patients (on treatment) who are misclassified as healthy have lower clinical disease activity scores than those who are correctly classified as lupus.

### Dotplots on raw data (pre-sampling)

```bash
./run_notebooks.sh notebooks/make_dotplot_data.ipynb;

# Fill in these paths using config.paths.dataset_specific_metadata, config.paths.dotplots_input, and config.paths.dotplots_output:
# (TODO: make a python script autogenerate these paths, and pipe to bash)
conda activate r-v41;
Rscript --vanilla scripts/plot_dot_plots.r \
    "data/data_v_$VERSION/metadata/participant_specimen_disease_map.tsv" \
    "data/data_v_$VERSION/dotplots/input" \
    "data/data_v_$VERSION/dotplots/output";
conda deactivate;
# Go back to our main conda environment
conda activate cuda-env-py39;
```

### Validation on external cohorts

_This section is only for `CrossValidationSplitStrategy.in_house_peak_disease_timepoints`._

The external cohorts were loaded in the ETL process. We have only-BCR and only-TCR external specimens so far, so we'll use those metamodel variants. Later we might find BCR+TCR external cohorts for a full-fledged evaluation. But for now, the BCR and TCR `fold_id=-1, fold_label="external"` specimens are completely separate sets.


```bash
# Embed with global fold (fold -1)
python scripts/run_embedding.py --fold_id -1 --external-cohort 2>&1 | tee "data/logs/external_validation_cohort_embedding.log"
python scripts/scale_embedding_anndatas.py --fold_id -1 --external-cohort 2>&1 | tee "data/logs/external_validation_cohort_scaling.log"


# Evaluate
./run_notebooks.sh \
    notebooks/evaluate_external_cohorts.ipynb \
    notebooks/summary.external_cohort_validation.ipynb;

```

### Compare in-house and Adaptive TCR data V gene use

We ran the following above for in-house data and for Adaptive:

```bash
./run_notebooks.sh notebooks/vgene_usage_stats.ipynb;
MALID_CV_SPLIT="adaptive_peak_disease_timepoints" ./run_notebook_to_new_file.sh notebooks/vgene_usage_stats.ipynb notebooks/vgene_usage_stats.generated.adaptive.ipynb;
```

Now run: `./run_notebooks.sh notebooks/airr_external_data/compare.vgene.use.ipynb`.

### Healthy resequencing sample analysis

We resequenced a number of healthy donors, so their specimens now have two replicates each.

Run the pipeline with `MALID_CV_SPLIT="in_house_peak_disease_leave_one_cohort_out"`.

Then do extra analysis of the replicates split up:

```bash
./run_notebooks.sh \
    notebooks/paired_sample_batch_effects.ipynb \
    notebooks/leave_one_cohort_out_metamodel.remove_broken_replicate.ipynb;
```
