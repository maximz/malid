import numpy as np
import pandas as pd
import pytest
from malid.embedders.base_embedder import (
    run_in_batches,
    BaseEmbedder,
    BaseFineTunedEmbedder,
)
import functools
from malid.datamodels import EmbedderSequenceContent
import logging

logger = logging.getLogger(__name__)


class MockClass:
    @run_in_batches(
        num_sequences_per_batch=50
    )  # We'll override this with self.batch_size
    def mock_method(self, arr):
        # Simply wrap the batch in a numpy array
        return np.array([arr])


# A standalone function for comparison
@run_in_batches(num_sequences_per_batch=50)
def standalone_mock_function(arr: np.ndarray) -> np.ndarray:
    # Simply wrap the batch in a numpy array
    return np.array([arr])


@pytest.mark.parametrize(
    "input_data, expected_number_of_batches",
    [
        (
            np.arange(100),
            2,
        ),  # 100 elements, batch size of 50, should result in 2 batches
        (np.arange(25), 1),  # 25 elements, batch size of 50, should result in 1 batch
    ],
)
@pytest.mark.parametrize("use_keyword_arg", [False, True])
def test_run_in_batches_with_class(
    input_data, expected_number_of_batches: int, use_keyword_arg: bool
):
    expected_batch_size = 50
    instance = MockClass()

    if use_keyword_arg:
        # Using keyword args
        output_data = instance.mock_method(arr=input_data)
    else:
        # Using positional args
        output_data = instance.mock_method(input_data)

    assert (
        len(output_data)
        == (
            len(input_data) // expected_batch_size
            + (len(input_data) % expected_batch_size > 0)
        )
        == expected_number_of_batches
    )
    assert np.array_equal(output_data.flatten(), input_data)


@pytest.mark.parametrize(
    "input_data, expected_number_of_batches",
    [
        (
            np.arange(100),
            2,
        ),  # 100 elements, batch size of 50, should result in 2 batches
        (np.arange(25), 1),  # 25 elements, batch size of 50, should result in 1 batch
    ],
)
@pytest.mark.parametrize("use_keyword_arg", [False, True])
def test_run_in_batches_with_standalone_function(
    input_data, expected_number_of_batches: int, use_keyword_arg: bool
):
    expected_batch_size = 50

    if use_keyword_arg:
        # Using keyword args
        output_data = standalone_mock_function(arr=input_data)
    else:
        # Using positional args
        output_data = standalone_mock_function(input_data)

    # Check the number of batches processed
    assert (
        len(output_data)
        == (
            len(input_data) // expected_batch_size
            + (len(input_data) % expected_batch_size > 0)
        )
        == expected_number_of_batches
    )

    # Check the content of the output
    assert np.array_equal(output_data.flatten(), input_data)


@pytest.mark.parametrize(
    "input_data, desired_batch_size, expected_number_of_batches",
    [
        (
            np.arange(100),
            10,
            10,
        ),  # 100 elements, batch size of 10, should result in 10 batches
        (
            np.arange(25),
            5,
            5,
        ),  # 25 elements, batch size of 5, should result in 5 batches
    ],
)
@pytest.mark.parametrize("use_keyword_arg", [False, True])
def test_run_in_batches_with_standalone_function_override(
    input_data,
    desired_batch_size: int,
    expected_number_of_batches: int,
    use_keyword_arg: bool,
):
    # Unwrap and rewrap
    mock_function_with_batch = run_in_batches(
        num_sequences_per_batch=desired_batch_size
    )(standalone_mock_function.__wrapped__)

    if use_keyword_arg:
        # Using keyword args
        output_data = mock_function_with_batch(arr=input_data)
    else:
        # Using positional args
        output_data = mock_function_with_batch(input_data)

    # Check the number of batches processed
    assert (
        len(output_data)
        == (
            len(input_data) // desired_batch_size
            + (len(input_data) % desired_batch_size > 0)
        )
        == expected_number_of_batches
    )

    # Check the content of the output
    assert np.array_equal(output_data.flatten(), input_data)


def test_run_in_batches_numpy_input_with_1d_output():
    @run_in_batches(num_sequences_per_batch=2)
    def func(arr: np.ndarray) -> np.ndarray:
        return np.array(list(reversed(arr)))

    # Notice that we reverse every pair of 2
    arr = np.array([0, 1, 2, 3, 4])
    assert np.array_equal(func(arr), [1, 0, 3, 2, 4])


def test_run_in_batches_numpy_input_with_2d_output():
    @run_in_batches(num_sequences_per_batch=2)
    def func(arr: np.ndarray) -> np.ndarray:
        newarr = np.array(list(reversed(arr)))
        # add more dimensions
        return np.vstack([newarr, np.zeros(newarr.shape[0])]).T

    # Notice that we reverse every pair of 2, and we make the output 2D
    arr = np.array([0, 1, 2, 3, 4])
    assert np.array_equal(func(arr), np.array([[1, 0], [0, 0], [3, 0], [2, 0], [4, 0]]))


def test_run_in_batches_pandas_input():
    @run_in_batches(num_sequences_per_batch=2)
    def func(df: pd.DataFrame) -> np.ndarray:
        return np.array(list(reversed(df["series1"].values)))

    df = pd.DataFrame(
        {"series1": [0, 1, 2, 3, 4], "series2": ["a", "b", "c", "d", "e"]}
    )
    assert np.array_equal(func(df), [1, 0, 3, 2, 4])


@pytest.fixture
def sequences():
    df = pd.DataFrame(
        {
            "fr1_seq_aa_q_trim": ["A" * 8, "A" * 10],
            "cdr1_seq_aa_q_trim": ["B" * 8, "B" * 10],
            "fr2_seq_aa_q_trim": ["C" * 8, "C" * 10],
            "cdr2_seq_aa_q_trim": ["D" * 8, "D" * 10],
            "fr3_seq_aa_q_trim": ["E" * 8, "E" * 10],
            "cdr3_seq_aa_q_trim": ["F" * 8, "F" * 10],
            "post_seq_aa_q_trim": ["G" * 8, "G" * 10],
        }
    )
    chars_all = ["A", "B", "C", "D", "E", "F", "G"]
    chars_subset = ["B", "D", "F"]
    return df, chars_all, chars_subset


def test_get_sequences(sequences):
    df, chars_all, chars_subset = sequences

    expected_outputs = {
        EmbedderSequenceContent.FR1ThruFR4: (
            [
                "".join([char * 8 for char in chars_all]),
                "".join([char * 10 for char in chars_all]),
            ],
            None,
            None,
        ),
        EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3: (
            [
                "".join([char * 8 for char in chars_all]),
                "".join([char * 10 for char in chars_all]),
            ],
            [8 * 5, 10 * 5],
            [8 * 6, 10 * 6],
        ),
        EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3_middle: (
            [
                "".join([char * 8 for char in chars_all]),
                "".join([char * 10 for char in chars_all]),
            ],
            [8 * 5 + 3, 10 * 5 + 3],
            [8 * 6 - 3, 10 * 6 - 3],
        ),
        EmbedderSequenceContent.CDR123: (
            [
                "".join([char * 8 for char in chars_subset]),
                "".join([char * 10 for char in chars_subset]),
            ],
            None,
            None,
        ),
        EmbedderSequenceContent.CDR123_but_only_return_CDR3: (
            [
                "".join([char * 8 for char in chars_subset]),
                "".join([char * 10 for char in chars_subset]),
            ],
            [8 * 2, 10 * 2],
            None,
        ),
        EmbedderSequenceContent.CDR123_but_only_return_CDR3_middle: (
            [
                "".join([char * 8 for char in chars_subset]),
                "".join([char * 10 for char in chars_subset]),
            ],
            [8 * 2 + 3, 10 * 2 + 3],
            [-3, -3],
        ),
        EmbedderSequenceContent.CDR3: (["F" * 8, "F" * 10], None, None),
    }

    # Confirm we are running this test for all possible EmbedderSequenceContent settings
    assert all(
        embedder_sequence_content in expected_outputs.keys()
        for embedder_sequence_content in EmbedderSequenceContent
    )

    for embedder_sequence_content, expected in expected_outputs.items():
        observed = BaseEmbedder._get_sequences(
            df=df, embedder_sequence_content=embedder_sequence_content
        )
        assert len(observed) == len(expected)
        for ix, (o, e) in enumerate(zip(observed, expected)):
            assert np.array_equal(
                o, e
            ), f"For {embedder_sequence_content}, field {ix}: observed {o}, expected {e}"


@pytest.mark.xfail(raises=ValueError)
def test_get_sequences_doesnt_allow_nulls():
    df = pd.DataFrame(
        {
            "cdr1_seq_aa_q_trim": ["1", "11", "111"],
            "cdr2_seq_aa_q_trim": ["2", "22", np.nan],
            "cdr3_seq_aa_q_trim": ["3", "33", "333"],
        }
    )
    BaseEmbedder._get_sequences(
        df=df, embedder_sequence_content=EmbedderSequenceContent.CDR123
    )


@pytest.mark.xfail(raises=ValueError)
def test_get_sequences_doesnt_allow_empty_strings():
    df = pd.DataFrame(
        {
            "cdr1_seq_aa_q_trim": ["1", "11", "111"],
            "cdr2_seq_aa_q_trim": ["2", "22", ""],
            "cdr3_seq_aa_q_trim": ["3", "33", "333"],
        }
    )
    BaseEmbedder._get_sequences(
        df=df, embedder_sequence_content=EmbedderSequenceContent.CDR123
    )


def test_partial_pooling():
    # imagine three sequences, 5-7 amino acids each, and each amino acid has a 2d embedding
    # embeddings is a list of one ndarray per sequence
    embeddings = [
        np.array([[1, -1], [2, -2], [5, -5], [6, -6], [3, -3]]),
        np.array([[1, -1], [2, -2], [0, 0], [5, -5], [6, -6], [3, -3]]),
        np.array([[1, -1], [2, -2], [0, 0], [5, -5], [6, -6], [7, -7], [3, -3]]),
    ]
    options = [
        # Try inputs corresponding to different EmbedderSequenceContent settings:
        # EmbedderSequenceContent.CDR123 or EmbedderSequenceContent.CDR3
        (
            (None, None),
            np.array(
                [
                    [np.mean([1, 2, 5, 6, 3]), np.mean([-1, -2, -5, -6, -3])],
                    [np.mean([1, 2, 0, 5, 6, 3]), np.mean([-1, -2, 0, -5, -6, -3])],
                    [
                        np.mean([1, 2, 0, 5, 6, 7, 3]),
                        np.mean([-1, -2, 0, -5, -6, -7, -3]),
                    ],
                ]
            ),
        ),
        # EmbedderSequenceContent.CDR123_but_only_return_CDR3
        (
            ([2, 3, 3], None),
            np.array(
                [
                    [np.mean([5, 6, 3]), np.mean([-5, -6, -3])],
                    [np.mean([5, 6, 3]), np.mean([-5, -6, -3])],
                    [np.mean([5, 6, 7, 3]), np.mean([-5, -6, -7, -3])],
                ]
            ),
        ),
        # EmbedderSequenceContent.CDR123_but_only_return_CDR3_middle
        (
            ([2, 3, 3], [4, 5, 6]),
            np.array(
                [
                    [np.mean([5, 6]), np.mean([-5, -6])],
                    [np.mean([5, 6]), np.mean([-5, -6])],
                    [np.mean([5, 6, 7]), np.mean([-5, -6, -7])],
                ]
            ),
        ),
    ]
    for (sequence_cut_start, sequence_cut_end), expected in options:
        observed = BaseEmbedder._partial_pooling(
            embeddings, sequence_cut_start, sequence_cut_end
        )
        assert np.array_equal(observed, expected)


def test_make_weights(sequences):
    df, _, _ = sequences

    if all(
        embedder_sequence_content.cdr3_region_weight_in_loss_computation == 1
        for embedder_sequence_content in EmbedderSequenceContent
    ):
        logger.warning(
            "test_make_weights is meaningless if all EmbedderSequenceContent.cdr3_region_weight_in_loss_computation are 1"
        )

    expected_outputs = {
        EmbedderSequenceContent.FR1ThruFR4: [np.ones(8 * 7), np.ones(10 * 7)],
        EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3: [
            np.hstack(
                [
                    np.ones(8 * 5),
                    np.full(
                        8,
                        EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3.cdr3_region_weight_in_loss_computation,
                    ),
                    np.ones(8),
                ]
            ),
            np.hstack(
                [
                    np.ones(10 * 5),
                    np.full(
                        10,
                        EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3.cdr3_region_weight_in_loss_computation,
                    ),
                    np.ones(10),
                ]
            ),
        ],
        EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3_middle: [
            np.hstack(
                [
                    np.ones(8 * 5 + 3),
                    np.full(
                        8 - 3 - 3,
                        EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3_middle.cdr3_region_weight_in_loss_computation,
                    ),
                    np.ones(3),
                    np.ones(8),
                ]
            ),
            np.hstack(
                [
                    np.ones(10 * 5 + 3),
                    np.full(
                        10 - 3 - 3,
                        EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3_middle.cdr3_region_weight_in_loss_computation,
                    ),
                    np.ones(3),
                    np.ones(10),
                ]
            ),
        ],
        EmbedderSequenceContent.CDR123: [
            np.ones(8 * 3),
            np.ones(10 * 3),
        ],
        EmbedderSequenceContent.CDR3: [np.ones(8), np.ones(10)],
        EmbedderSequenceContent.CDR123_but_only_return_CDR3: [
            np.hstack(
                [
                    np.ones(8 * 2),
                    np.full(
                        8,
                        EmbedderSequenceContent.CDR123_but_only_return_CDR3.cdr3_region_weight_in_loss_computation,
                    ),
                ]
            ),
            np.hstack(
                [
                    np.ones(10 * 2),
                    np.full(
                        10,
                        EmbedderSequenceContent.CDR123_but_only_return_CDR3.cdr3_region_weight_in_loss_computation,
                    ),
                ]
            ),
        ],
        EmbedderSequenceContent.CDR123_but_only_return_CDR3_middle: [
            np.hstack(
                [
                    np.ones(8 * 2 + 3),
                    np.full(
                        8 - 3 - 3,
                        EmbedderSequenceContent.CDR123_but_only_return_CDR3_middle.cdr3_region_weight_in_loss_computation,
                    ),
                    np.ones(3),
                ]
            ),
            np.hstack(
                [
                    np.ones(10 * 2 + 3),
                    np.full(
                        10 - 3 - 3,
                        EmbedderSequenceContent.CDR123_but_only_return_CDR3_middle.cdr3_region_weight_in_loss_computation,
                    ),
                    np.ones(3),
                ]
            ),
        ],
    }

    # Confirm we are running this test for all possible EmbedderSequenceContent settings
    assert all(
        embedder_sequence_content in expected_outputs.keys()
        for embedder_sequence_content in EmbedderSequenceContent
    )

    for embedder_sequence_content, expected in expected_outputs.items():
        extracted_sequences = BaseEmbedder._get_sequences(
            df=df, embedder_sequence_content=embedder_sequence_content
        )
        observed_weight_arrays_per_sequence = (
            BaseFineTunedEmbedder._make_weights_for_sequence_positions(
                *extracted_sequences,
                embedder_sequence_content=embedder_sequence_content,
            )
        )
        assert (
            len(observed_weight_arrays_per_sequence)
            == len(expected)
            == len(extracted_sequences[0])
        )
        for ix, (o, e, raw_sequence) in enumerate(
            zip(observed_weight_arrays_per_sequence, expected, extracted_sequences[0])
        ):
            # No matter what, the shapes should be same as shapes in: [np.ones(len(s)) for s in sequences]
            # Sanity check that our expected values have the right shapes
            assert len(o) == len(e) == len(raw_sequence)

            # Confirm we match the expected values
            assert np.array_equal(
                o, e
            ), f"For example {ix} and {embedder_sequence_content}: observed {o}, expected {e}, raw sequence was {raw_sequence}"
