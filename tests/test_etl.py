import numpy as np
import pandas as pd

from malid.datamodels import GeneLocus
from malid.etl import (
    dtypes_read_in,
    dtypes_expected_after_preprocessing,
    preprocess_each_participant_table,
    _trim_sequences,
)


def test_etl_with_empty_sample_dataframe():
    for gene_locus in GeneLocus:
        # confirm that works with an empty dataframe
        orig = pd.DataFrame(
            {c: pd.Series(dtype=t) for c, t in dtypes_read_in[gene_locus].items()}
        )
        processed = preprocess_each_participant_table(
            df=orig,
            gene_locus=gene_locus,
            final_dtypes=dtypes_expected_after_preprocessing,
        )
        assert processed.shape[0] == 0
        assert processed.shape[1] > 5


def test_trim_sequences():
    trimmed = _trim_sequences(pd.Series(["V Y Y T G S T", " N A ", "    ", " ... "]))
    # complex series equals comparison to allow nan==nan
    assert trimmed.equals(pd.Series(["VYYTGST", "NA", np.nan, np.nan]))
