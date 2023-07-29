#!/usr/bin/env python

import numpy as np
import pytest

from malid.external.genetools_arrays import (
    weighted_mode,
    strings_to_character_arrays,
    strings_to_numeric_vectors,
    numeric_vectors_to_character_arrays,
    make_consensus_sequence,
)


def test_strings_to_character_arrays():
    np.testing.assert_array_equal(
        strings_to_character_arrays(["ABC", "ABD"]),
        np.array([[b"A", b"B", b"C"], [b"A", b"B", b"D"]]),
    )


def test_strings_to_character_arrays_1d():
    # still produces 2d
    np.testing.assert_array_equal(
        strings_to_character_arrays(["ABC"]), np.array([[b"A", b"B", b"C"]])
    )


@pytest.mark.xfail(raises=ValueError)
def test_strings_to_character_arrays_length_mismatch():
    strings_to_character_arrays(["ABC", "AB"], validate_equal_lengths=True)


def test_strings_to_character_arrays_length_mismatch_no_validation():
    np.testing.assert_array_equal(
        strings_to_character_arrays(["ABC", "AB"], validate_equal_lengths=False),
        np.array([[b"A", b"B", b"C"], [b"A", b"B", b""]]),
    )


def test_strings_to_numeric_vectors():
    np.testing.assert_array_equal(
        strings_to_numeric_vectors(["ABC", "ABD"]),
        np.array([[65, 66, 67], [65, 66, 68]]),
    )


def test_strings_to_numeric_vectors_1d():
    # still produces 2d
    np.testing.assert_array_equal(
        strings_to_numeric_vectors(["ABC"]), np.array([[65, 66, 67]])
    )


@pytest.mark.xfail(raises=ValueError)
def test_strings_to_numeric_vectors_length_mismatch():
    strings_to_numeric_vectors(["ABC", "AB"], validate_equal_lengths=True)


def test_strings_to_numeric_vectors_length_mismatch_no_validation():
    # NaN requires float type
    np.testing.assert_array_equal(
        strings_to_numeric_vectors(["ABC", "AB"], validate_equal_lengths=False),
        np.array([[65.0, 66.0, 67.0], [65.0, 66.0, np.nan]]),
    )


def test_numeric_vectors_to_character_arrays():
    np.testing.assert_array_equal(
        strings_to_character_arrays(["ABC"]),
        numeric_vectors_to_character_arrays(
            strings_to_numeric_vectors(["ABC"]).astype(int)
        ),
    )


# TODO: test make_consensus_vector() with int arrays matrix


def test_consensus_sequence():
    sequences = ["AB", "AC", "AA"]
    weights = [1, 1, 3]
    assert make_consensus_sequence(sequences, weights) == "AA"


def test_equal_weights():
    sequences = ["AB", "AC", "AD"]
    weights = [1, 1, 1]
    assert make_consensus_sequence(sequences, weights) == "AB"


def test_single_character():
    sequences = ["A", "A", "B"]
    weights = [1, 1, 3]
    assert make_consensus_sequence(sequences, weights) == "B"
    assert weighted_mode(sequences, weights) == "B"


def test_weighted_mode_with_numbers():
    arr = [10, 10, 5]
    weights = [1, 1, 3]
    assert weighted_mode(arr, weights) == 5


def test_single_sequence():
    sequences = ["ABC"]
    weights = [1]
    assert make_consensus_sequence(sequences, weights) == "ABC"


def test_additivity_of_repeated_character():
    sequences = ["A", "A", "B"]
    weights = [2, 3, 4]
    assert make_consensus_sequence(sequences, weights) == "A"
    assert weighted_mode(sequences, weights) == "A"


@pytest.mark.xfail(raises=ValueError)
def test_shape_mismatch():
    # note that this doesn't work the other way around, because we short-circuit if len(sequences) == 1
    sequences = ["ABC", "AB"]
    weights = [1]
    make_consensus_sequence(sequences, weights)


@pytest.mark.xfail(raises=ValueError)
def test_length_mismatch():
    sequences = ["ABC", "AB"]
    weights = [1, 2]
    make_consensus_sequence(sequences, weights)


@pytest.mark.xfail(raises=ValueError)
def test_length_mismatch_2():
    sequences = ["AB", "ABC"]
    weights = [1, 2]
    make_consensus_sequence(sequences, weights)
