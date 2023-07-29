# configuring multiprocessing is necessary to prevent tests that use multiprocessing from hanging in CI.
# without this, test_parallel_apply.py hangs indefinitely when running in github actions.
# see https://github.com/scipy/scipy/issues/11835 and https://github.com/scipy/scipy/pull/11836/
# and https://stackoverflow.com/q/64095876/130164 and https://github.com/qtile/qtile/issues/2068 and https://github.com/qtile/qtile/commit/57f0963fbd63f1c32b383412bc9bbc093b38dc22
# and maybe https://pythonspeed.com/articles/python-multiprocessing/
import multiprocessing
import copy

import numpy as np
import pytest

import choosegpu

multiprocessing.set_start_method("spawn")


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="Run GPU tests")


# See also https://stackoverflow.com/a/43938191/130164 for an alternative


def pytest_configure(config):
    # Configure GPU upfront so that test methods aren't constantly rewriting the config.
    # It's necessary to be consistent because GPU config at time of first Torch or Tensorflow import will determine its behavior for the remainder of the session.
    # Avoid calling config.configure_gpu() in any tests.
    # This code runs before any tests. It looks for the "--gpu" command line argument, and enables GPU and configures which tests run accordingly.
    # We should run "pytest" and "pytest --gpu" as two separate sessions.

    from malid import config as malid_config_module

    if config.getoption("--gpu"):
        choosegpu.configure_gpu(enable=True, memory_pool=True)
    else:
        choosegpu.configure_gpu(enable=False)

    ## Change config.paths to be based at tmp_path

    # save original values for later
    # copy is not necessary for config.paths, because make_paths() generates new objects, but copying just to be safe.
    old_paths = copy.copy(malid_config_module.paths)

    # modify
    # we can't use tmp_path fixture directly in pytest_configure, so make it manually
    # per https://github.com/pytest-dev/pytest/blob/4b77638ba88f6dc7bf1125deeb8e4d4bfd795ad5/src/_pytest/tmpdir.py#L183
    tmp_dir_plugin = config.pluginmanager.get_plugin("tmpdir")
    temp_path_factory = tmp_dir_plugin.TempPathFactory.from_config(
        config, _ispytest=True
    )
    tmp_path = temp_path_factory.mktemp("test_temp_path", numbered=True)
    malid_config_module.paths = malid_config_module.make_paths(
        embedder=malid_config_module.embedder,
        base_data_dir="data",
        base_output_dir="out",
        base_scratch_dir="scratch",
        relative_to_path=tmp_path,
        dataset_version="test_snapshots",
    )
    malid_config_module.make_dirs()

    # point some paths to existing locations
    malid_config_module.paths.tests_snapshot_dir = old_paths.tests_snapshot_dir
    malid_config_module.paths.dataset_specific_metadata = (
        old_paths.tests_snapshot_dir / "dataset_specific_metadata"
    )
    malid_config_module.paths.metadata_dir = old_paths.metadata_dir


def pytest_runtest_setup(item):
    # Called for each test function to determine whether to run it,
    # based on global config (specifically --gpu command line argument) and the test function's decorators (markers).
    if "gpu" in item.keywords and not item.config.getoption("--gpu"):
        pytest.skip("Need --gpu option to run this test marked @pytest.mark.gpu")
    if item.config.getoption("--gpu") and "skip_if_gpu" in item.keywords:
        pytest.skip(
            "Skipping because --gpu enabled while marked @pytest.mark.skip_if_gpu"
        )


@pytest.fixture
def sample_data():
    """Make multiclass train and test data"""
    n_features = 5
    covid_data = np.random.randn(100, n_features) + np.array([5] * n_features)
    hiv_data = np.random.randn(100, n_features) + np.array([15] * n_features)
    healthy_data = np.random.randn(100, n_features) + np.array([-5] * n_features)
    # add a fourth class, so that coefs_ for OvO and OvR models have different shapes.
    ebola_data = np.random.randn(100, n_features) + np.array([-15] * n_features)
    X_train = np.vstack([covid_data, hiv_data, healthy_data, ebola_data])
    y_train = np.hstack(
        [
            ["Covid"] * covid_data.shape[0],
            ["HIV"] * hiv_data.shape[0],
            ["Healthy"] * healthy_data.shape[0],
            ["Ebola"] * ebola_data.shape[0],
        ]
    )
    X_test = np.array(
        [
            [5] * n_features,
            [15] * n_features,
            [-5] * n_features,
            [6] * n_features,
            [-15] * n_features,
        ]
    )
    y_test = np.array(["Covid", "HIV", "Healthy", "Covid", "Ebola"])
    return (X_train, y_train, X_test, y_test)


@pytest.fixture
def sample_data_two(sample_data):
    """Same train data, different test data"""
    (X_train, y_train, X_test, y_test) = sample_data
    n_features = X_train.shape[1]
    X_test2 = np.array(
        [
            [6] * n_features,
            [14] * n_features,
            [-4] * n_features,
            [4] * n_features,
            [-14] * n_features,
        ]
    )
    y_test2 = np.array(["Covid", "HIV", "Healthy", "Covid", "Ebola"])
    return (X_train, y_train, X_test2, y_test2)


@pytest.fixture
def sample_data_binary():
    n_features = 5
    covid_data = np.random.randn(100, n_features) + np.array([5] * n_features)
    healthy_data = np.random.randn(100, n_features) + np.array([-5] * n_features)
    X_train = np.vstack([covid_data, healthy_data])
    y_train = np.hstack(
        [
            ["Covid"] * covid_data.shape[0],
            ["Healthy"] * healthy_data.shape[0],
        ]
    )
    X_test = np.array([[5] * n_features, [-5] * n_features, [3] * n_features])
    y_test = np.array(["Covid", "Healthy", "Covid"])
    return (X_train, y_train, X_test, y_test)
