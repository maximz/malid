import numpy as np
import pandas as pd
import pytest

from malid import helpers
from malid.datamodels import GeneLocus
from malid.trained_model_wrappers import SequenceClassifier


@pytest.fixture
def X():
    """Mock 5-dimensional vectors for 4 examples."""
    return np.random.randn(4, 5)


def test_add_dummy_variables_respects_X_dtype(X):

    isotype_groups = ["IGHG", "IGHA", "IGHD-M", "IGHD-M"]
    v_genes = ["IGHV4-34", "IGHV3-53", "IGHV3-53", "IGHV3-13"]

    # Confirm test assumptions: we assume these three isotype groups are whitelisted
    assert all(i in helpers.isotype_groups_kept[GeneLocus.BCR] for i in isotype_groups)
    # And confirm test assumptions: these V genes are available

    for dtype in [np.float16, np.float32]:
        X_added = SequenceClassifier._add_extra_columns_to_embedding_vectors(
            data_X=X.astype(dtype),
            isotype_groups=isotype_groups,
            v_genes=v_genes,
            gene_locus=GeneLocus.BCR,
            include_v_gene_as_dummy_variable=True,
            include_isotype_as_dummy_variable=True,
        )
        assert X_added.dtype == dtype
        assert X_added.shape == (
            X.shape[0],
            X.shape[1]
            + len(set(helpers.isotype_groups_kept[GeneLocus.BCR]))
            + len(set(helpers.all_observed_v_genes()[GeneLocus.BCR])),
        )
