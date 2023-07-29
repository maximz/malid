from typing import Any, Optional, Union, List

import numpy as np
import pandas as pd


def get_top_n_percent(df: pd.DataFrame, col: str, fraction: float) -> pd.DataFrame:
    """Get top fraction `n` of a dataframe `df` by a specific column `col`"""
    return df.sort_values(col, ascending=False).head(n=int(fraction * df.shape[0]))


def get_top_n(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    """Get top amount `n` of a dataframe `df` by a specific column `col`"""
    return df.sort_values(col, ascending=False).head(n=n)


def weighted_mode(
    arr: Union[list, np.ndarray, pd.Series],
    weights: Union[List[int], np.ndarray, pd.Series],
) -> Any:
    """
    Get weighted mode (most common value) in array.
    Faster than sklearn.utils.extmath.weighted_mode but does not support axis vectorization.
    """
    return (
        pd.DataFrame({"key": arr, "value": weights})
        .groupby("key", observed=True, sort=False)["value"]
        .sum()
        .idxmax()
    )


def strings_to_character_arrays(
    strs: Union[np.ndarray, List[str], pd.Series], validate_equal_lengths: bool = True
) -> np.ndarray:
    # create character matrix by "viewing" as a 1-character string array, then reshaping
    char_matrix = np.array(strs)
    char_matrix = (
        char_matrix.astype("bytes").view("S1").reshape((char_matrix.shape[0], -1))
    )

    if validate_equal_lengths and b"" in char_matrix:
        # Note: this won't trip up on spaces in the string, because those are " " and ASCII code 32, not "" and int code 0.
        raise ValueError("Input strings must be of equal lengths.")

    return char_matrix


def strings_to_numeric_vectors(
    strs: Union[np.ndarray, List[str], pd.Series], validate_equal_lengths: bool = True
) -> np.ndarray:
    numeric_arr = strings_to_character_arrays(
        strs, validate_equal_lengths=validate_equal_lengths
    ).view(np.uint8)

    if 0 in numeric_arr:
        # Replace blanks (0s -- from \x00) with np.nan. Cast to float first to support np.nan
        numeric_arr = numeric_arr.astype(float)
        np.place(numeric_arr, numeric_arr == 0.0, np.nan)

    return numeric_arr


def numeric_vectors_to_character_arrays(arr: np.ndarray) -> np.ndarray:
    return np.array(arr, dtype=np.uint8).view("c")


def make_consensus_vector(
    matrix: np.ndarray, frequencies: Union[np.ndarray, List[int], pd.Series]
) -> np.ndarray:
    # weighted mode at each position -> centroid
    # apply function to every column vector
    consensus_elements = np.apply_along_axis(
        lambda col: weighted_mode(col, frequencies), 0, matrix
    )
    return consensus_elements


def make_consensus_sequence(
    sequences: Union[np.ndarray, pd.Series, List[str]],
    frequencies: Union[np.ndarray, pd.Series, List[int]],
) -> str:
    """
    Get weighted mode for each character across a set of equal-length input strings.
    """
    sequences = np.array(sequences)  # defensive cast
    if sequences.shape[0] == 1:
        # If only one sequence, just return it
        # The rest of the code is legal for this case, but probably faster to short circuit
        return sequences.item(0)

    char_matrix = strings_to_character_arrays(sequences, validate_equal_lengths=True)

    # weighted mode for each character
    consensus_elements = make_consensus_vector(char_matrix, frequencies)

    # assemble consensus string from individual characters
    return "".join(consensus_elements.astype(str))


def _masked_vector_fill_for_argmin_argmax_output(
    masked_arr: np.ma.MaskedArray,
    output: Union[np.ndarray, float, int],
    axis: Optional[int] = None,
) -> Union[np.ndarray, float, int]:
    """Finalize our masked_argmin/masked_argmax output by correcting one piece:
    For rows/columns that are all maksed (all NaN), argmin/argmax will return 0. That's incorrect behavior.
    Set entries for all-NaN rows or columns to NaN.
    """
    # if axis None, we will have scalar output from argmin/argmax
    if np.isscalar(output):
        # if all are masked, output will be 0, which is incorrect
        # return nan instead
        return np.nan if masked_arr.mask.all(axis=axis) else output

    # for rows/columns that are all masked (all nan), argmin will return 0. that's incorrect behavior.
    # set entries for all-nan rows or columns to NaN
    # first, cast to float so we can later set some entries to np.nan, which is a float
    output = output.astype(float)
    # .any will return
    output[masked_arr.mask.all(axis=axis)] = np.nan

    return output


def masked_argmin(
    masked_arr: np.ma.MaskedArray, axis: Optional[int] = None
) -> Union[np.ndarray, float, int]:
    """
    argmin on masked array. return nan for row/column (depending on axis setting) of all nans
    """
    # run standard argmin
    argmin_output = masked_arr.argmin(
        fill_value=np.ma.minimum_fill_value(masked_arr), axis=axis
    )

    return _masked_vector_fill_for_argmin_argmax_output(
        masked_arr, argmin_output, axis=axis
    )


def masked_argmax(
    masked_arr: np.ma.MaskedArray, axis: Optional[int] = None
) -> Union[np.ndarray, float, int]:
    """
    argmax on masked array. return nan for row/column (depending on axis setting) of all nans
    """
    # run standard argmax
    argmax_output = masked_arr.argmax(
        fill_value=np.ma.maximum_fill_value(masked_arr), axis=axis
    )

    return _masked_vector_fill_for_argmin_argmax_output(
        masked_arr, argmax_output, axis=axis
    )


def convert_matrix_to_one_element_per_row(arr: np.ndarray) -> pd.DataFrame:
    """
    record each element of 2d matrix as one entry in dataframe, with the row and column ids stored as well as the value
    """
    # remove any index or column names
    df = pd.DataFrame(np.array(arr))

    # melt
    df = (
        df.rename_axis("row_id", axis="index")  # rename index
        .reset_index()  # index is now a column called row_id
        .melt(id_vars="row_id", var_name="col_id", value_name="value")
    )

    # cast from object to int
    df["col_id"] = df["col_id"].astype(int)

    return df


def get_trim_both_sides_mask(
    a: Union[np.ndarray, pd.DataFrame], proportiontocut: float, axis: int = 0
) -> np.ndarray:
    """returns mask that applies consistent trim-both-sides learned on one array.
    suppose you have a data array and a weights array. you want to trimboth() the data array but keep the element weights aligned.
    solution:
        trimming_mask = get_trim_both_sides_mask(data, proportiontocut=0.1)
        return data[trimming_mask], weights[trimming_mask]
    """
    # based on scipy.stats.trimboth and scipy.stats.trim_mean
    a = np.asarray(a)

    if a.size == 0:
        return slice(None)

    if axis is None:
        a = a.ravel()
        axis = 0

    nobs = a.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if lowercut >= uppercut:
        raise ValueError("Proportion too big.")

    # np.partition creates a copy of the array with its elements rearranged in such a way that the value of the element in k-th position is in the position it would be in a sorted array.
    # All elements smaller than the k-th element are moved before this element and all equal or greater are moved behind it.
    # The ordering of the elements in the two partitions is undefined.

    # tl,dr: partial sort so that anything < lowercut-th index is before it in the array, and anything >= uppercut-th index is after it in the array.
    # that way we can just chop off the two ends

    # but instead let's get the indices that partition() would keep - this is argpartition()
    # note: often used with take_along_axis() to apply the indices.
    # these are equivalent: np.take_along_axis(x, np.argpartition(x, kth=1, axis=-1), axis=-1) == np.partition(x, kth=1)

    index_array = np.argpartition(a=a, kth=(lowercut, uppercut - 1), axis=axis)

    # generate an empty slicer along each axis of the array
    ndim = (
        index_array.ndim
    )  # per original scipy: should this be np.take_along_axis(a, index_array, axis=axis).ndim or a.ndim instead?
    sl = [slice(None)] * ndim
    # along the axis that matters, apply a mask that will cut the ends off
    sl[axis] = slice(lowercut, uppercut)
    mask = tuple(sl)

    return index_array[mask]


def make_dummy_variables_in_specific_order(
    values: Union[pd.Series, List[str]],
    expected_list: List[str],
    allow_missing_entries: bool,
) -> pd.DataFrame:
    """
    Create dummy variables in a defined order. All the values are confirmed to be in the "expected order" list.
    If an entry from "expected order" list is not present, still include as a dummy variable with all 0s if allow_missing_entires is True, or throw error otherwise.
    """
    # Defensive cast to Series,
    # and remove any unused categories in case it's a Categorical
    values = pd.Series(values).astype("category").cat.remove_unused_categories()

    if values.isna().any():
        raise ValueError("Some input entries are blank")

    dummy_vars = pd.get_dummies(values)

    # If values was a Categorical series, we won't be able to add new columns below to dummy_vars.
    # So cast the columns to string.
    dummy_vars.columns = dummy_vars.columns.astype(str)

    for value in expected_list:
        # Check whether each expected value is present in this dataset.
        if value not in dummy_vars.columns:
            if not allow_missing_entries:
                raise ValueError(
                    f"Entry {value} was expected, but was not found in the dataset."
                )
            # insert new column of all 0s (never present)
            dummy_vars[value] = 0

    if set(dummy_vars.columns) != set(expected_list):
        raise ValueError(
            "There are some values included that are not in the list of expected/possible values."
        )

    # Reorder
    dummy_vars = dummy_vars[expected_list]

    return dummy_vars
