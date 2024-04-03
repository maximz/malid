from __future__ import annotations
import abc
import functools
import inspect
from typing_extensions import Self
from typing import List, Type, Union, ClassVar, Optional, Tuple
from pathlib import Path
from static_class_property import classproperty
import itertools
import more_itertools

import numpy as np
import pandas as pd

from malid.datamodels import GeneLocus, EmbedderSequenceContent


# Abstract class. Anything marked @abc.abstractmethod is required in implementations (ellipsis ... is a standin for raising NotImplementedError).


class BaseEmbedder(metaclass=abc.ABCMeta):
    """Base embedder. Must override embed()."""

    name: ClassVar[str]
    friendly_name: ClassVar[str]
    dimensionality: ClassVar[int]
    # by default, supports any gene loci
    gene_loci_supported: GeneLocus = GeneLocus.BCR | GeneLocus.TCR

    def __init__(
        self, fold_id: Optional[int] = None, gene_locus: Optional[GeneLocus] = None
    ) -> None:
        # no-op: BaseEmbedder ignores the parameters.
        # just including for consistency with BaseFineTunedEmbedder
        pass

    @classproperty
    def non_fine_tuned_version(cls: Type[Self]) -> Type[BaseEmbedder]:
        """No-op: returns self type."""
        return cls

    @classproperty
    def is_fine_tuned(cls: Type) -> bool:
        return issubclass(cls, BaseFineTunedEmbedder)

    @classproperty
    def embedder_sequence_content(cls) -> EmbedderSequenceContent:
        # Default is embed CDR1+2+3
        return EmbedderSequenceContent.CDR123

    @abc.abstractmethod
    def embed(
        self, sequences: pd.DataFrame, dtype: Optional[Union[np.dtype, str]] = None
    ) -> np.ndarray:
        """Embed sequences. Must be overridden in subclasses."""
        ...

    @abc.abstractmethod
    def calculate_cross_entropy_loss(
        self,
        sequences: Union[List[str], np.ndarray],
        batch_size: Optional[int] = None,
    ) -> float:
        """
        Calculate the average cross entropy loss for a set of sequences. Sequences do not need to be the same length.
        """
        ...

    def perplexity(self, sequences: Union[List[str], np.ndarray], **kwargs) -> float:
        """Calculate perplexity. See calculate_cross_entropy_loss() for kwargs."""
        return np.exp(self.calculate_cross_entropy_loss(sequences=sequences, **kwargs))

    @staticmethod
    def _partial_pooling(
        sequence_embeddings: List[np.ndarray],
        sequence_cut_start: Optional[np.ndarray],
        sequence_cut_end: Optional[np.ndarray],
    ):
        """Compute mean embedding of a portion of each sequence.

        sequence_embeddings: positional embedding vectors for a list of sequences (list with one entry per sequence; each entry has shape sequence_length x embedding_dimensionality)
        sequence_cut_start: optional index of first position to include in mean (array of one int per sequence)
        sequence_cut_end: optional index of last position to include in mean (array of one int per sequence)

        sequence_embeddings should not include class/start-of-sequence token or end-of-sequence/padding tokens.
        """
        # to not apply a slice index at the start or end, we will pass None instead of an int index
        # wrap the None as a list for the itertools action below. we will use zip_longest to extend this array of Nones to the necessary length
        if sequence_cut_start is None:
            sequence_cut_start = [None]
        if sequence_cut_end is None:
            sequence_cut_end = [None]

        # mean of a portion of each embedding
        return np.vstack(
            [
                e[start_ix:end_ix, :].mean(axis=0)
                for e, start_ix, end_ix in itertools.zip_longest(
                    sequence_embeddings,
                    sequence_cut_start,
                    sequence_cut_end,
                    fillvalue=None,
                )
            ]
        )

    @staticmethod
    def _get_sequences(
        df: pd.DataFrame, embedder_sequence_content: EmbedderSequenceContent
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        EmbedderSequenceContent.validate(embedder_sequence_content)
        sequence_region_columns = embedder_sequence_content.included_sequence_regions
        seqs_df = df[sequence_region_columns]
        cdr3_region_name = EmbedderSequenceContent.CDR3.included_sequence_regions[0]

        # Detect N/As (whether np.nan or empty string) and error out
        if seqs_df.mask(seqs_df == "").isna().any().any():
            raise ValueError("Sequences contain N/As or empty strings")

        if embedder_sequence_content in [
            EmbedderSequenceContent.FR1ThruFR4,
            EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3,
            EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3_middle,
            EmbedderSequenceContent.CDR123,
            EmbedderSequenceContent.CDR123_but_only_return_CDR3,
            EmbedderSequenceContent.CDR123_but_only_return_CDR3_middle,
        ]:
            if cdr3_region_name not in sequence_region_columns:
                raise ValueError(
                    f"Missing cdr3_region_name={cdr3_region_name} column in {embedder_sequence_content} included_sequence_regions={sequence_region_columns}"
                )

            cdr3_column_index = sequence_region_columns.index(cdr3_region_name)
            columns_before_cdr3 = sequence_region_columns[:cdr3_column_index]
            columns_cdr3_onwards = sequence_region_columns[cdr3_column_index:]

            all_sequence_regions_before_cdr3 = (
                seqs_df[columns_before_cdr3].to_numpy().sum(axis=1)
            )
            sequences_full = all_sequence_regions_before_cdr3 + seqs_df[
                columns_cdr3_onwards
            ].to_numpy().sum(axis=1)

            if embedder_sequence_content in [
                EmbedderSequenceContent.FR1ThruFR4,
                EmbedderSequenceContent.CDR123,
            ]:
                return sequences_full, None, None
            else:
                cdr3_start_indexes = np.array(
                    [len(s) for s in all_sequence_regions_before_cdr3]
                )
                if cdr3_column_index == len(sequence_region_columns) - 1:
                    # Special case: if CDR3 is the last column (e.g. in CDR123 sequences),
                    # then cdr3_end_indexes is None
                    cdr3_end_indexes = None
                else:
                    cdr3_end_indexes = (
                        cdr3_start_indexes
                        + seqs_df[cdr3_region_name].str.len().to_numpy()
                    )

                if embedder_sequence_content in [
                    EmbedderSequenceContent.FR1ThruFR4_but_only_return_CDR3_middle,
                    EmbedderSequenceContent.CDR123_but_only_return_CDR3_middle,
                ]:
                    # Move 3 AA in from the ends
                    # TODO: customize based on the J gene (does IgBlast have this info in human_gl.aux? see https://github.com/NBISweden/IgDiscover/issues/12#issuecomment-310373598)

                    cdr3_start_indexes += 3

                    if cdr3_end_indexes is None:
                        # Special case: cdr3_end_indexes is None if CDR3 is the last column (e.g. in CDR123 sequences)
                        cdr3_end_indexes = np.full(sequences_full.shape[0], -3)
                    else:
                        cdr3_end_indexes -= 3

                return sequences_full, cdr3_start_indexes, cdr3_end_indexes

        elif embedder_sequence_content in [
            EmbedderSequenceContent.CDR3,
        ]:
            return seqs_df[cdr3_region_name].to_numpy(), None, None

        else:
            raise NotImplementedError(
                f"Unknown EmbedderSequenceContent: {embedder_sequence_content}"
            )


class BaseFineTunedEmbedder(BaseEmbedder):
    """Base fine tuned embedder. Must override _load_fine_tuned_model() and finetune()."""

    fold_id: int
    gene_locus: GeneLocus

    # How many sequences to use for fine-tuning (training and validation).
    num_train_to_choose = 500000
    num_validation_to_choose = 20000

    def __init__(self, fold_id: int, gene_locus: GeneLocus) -> None:
        """Get fold and gene locus-specific fine-tuned embedder"""
        super().__init__(fold_id, gene_locus)

        # This is the outward-facing interface that the user calls.
        # Does some checks.
        if fold_id is None or gene_locus is None:
            raise ValueError(
                "fold_id and gene_locus must be specified for fine-tuned embedder"
            )
        GeneLocus.validate_single_value(gene_locus)
        self.fold_id = fold_id
        self.gene_locus = gene_locus

        # This is the actual implementation that must be defined in subclasses.
        self._load_fine_tuned_model(fold_id, gene_locus)

    @abc.abstractmethod
    def _load_fine_tuned_model(self: Self, fold_id: int, gene_locus: GeneLocus):
        # Must be defined in subclasses.
        ...

    @classproperty
    @abc.abstractmethod
    def non_fine_tuned_version(cls: Type[Self]) -> Type[BaseEmbedder]:
        """Given a fine-tuned embedder type, return the non-fine-tuned version."""
        ...

    @classmethod
    @abc.abstractmethod
    def finetune(
        cls,
        train_sequences: np.ndarray,
        train_sequence_position_weights: List[np.ndarray],
        validation_sequences: np.ndarray,
        validation_sequence_position_weights: List[np.ndarray],
        num_epochs: int,
        output_dir: Union[Path, str],
        emit_every_n_epochs: int,
    ) -> None:
        ...

    @staticmethod
    def _make_weights_for_sequence_positions(
        seqs: np.ndarray,
        cdr3_start_indexes: Optional[np.ndarray],
        cdr3_end_indexes: Optional[np.ndarray],
        embedder_sequence_content: EmbedderSequenceContent,
    ) -> List[np.ndarray]:
        """Make positional weight array for each sequence (which may have different lengths). Will be used for computing masked LM loss on a sequence in fine-tuning process."""
        # to not apply a slice index at the start or end, we will pass None instead of an int index
        # wrap the None as a list for the itertools action below. we will use zip_longest to extend this array of Nones to the necessary length
        if cdr3_start_indexes is None:
            cdr3_start_indexes = [None]
        if cdr3_end_indexes is None:
            cdr3_end_indexes = [None]

        def _make_single_sequence_weight(
            seq: str,
            cdr3_start_index: Optional[int],
            cdr3_end_index: Optional[int],
            cdr3_weight: float,
        ):
            weights = np.ones(len(seq))
            weights[cdr3_start_index:cdr3_end_index] = cdr3_weight
            return weights

        EmbedderSequenceContent.validate(embedder_sequence_content)
        cdr3_weight = embedder_sequence_content.cdr3_region_weight_in_loss_computation

        return [
            _make_single_sequence_weight(s, start_ix, end_ix, cdr3_weight)
            for s, start_ix, end_ix in itertools.zip_longest(
                seqs,
                cdr3_start_indexes,
                cdr3_end_indexes,
                fillvalue=None,
            )
        ]


def run_in_batches(num_sequences_per_batch=50):
    """
    Decorator to run a function over an iterable in batches of size [num_sequences_per_batch], then np.vstack the outputs.

    Implementation notes:
    - to support arguments, we have a decorator factory. see https://stackoverflow.com/a/5929165/130164
    - there are a few scenarios for how this can be used:

    ```
    class MyClass:
        # Scenario 1: decorating a function in a class
        # Not yet a "bound method" -- will become one when the class is initialized
        # Wrapper's args will be (self, arr)
        @run_in_batches(num_sequences_per_batch=50)
        def embed(self, arr):
            return arr

        # Scenario 2: decorating at runtime, meaning decorating a method (a function belonging to an already-created instance of a class)
        # Wrapper's args will be (arr) -- i.e. no self.
        def embed_inner(self, arr):
            return arr
        def embed_callable(self, arr):
            return run_in_batches(num_sequences_per_batch=1500)(self.embed_inner)(arr)


    # Scenario 3: decorating a function outside any class. Wrapper's args will be (arr) only.
    @run_in_batches(num_sequences_per_batch=50)
    def embed_outside_class(arr):
        return arr

    Recommendation: if planning to cast to a lower precision dtype, do this within each batch, rather than after batches have been concatenated.
    ```
    """
    if not isinstance(num_sequences_per_batch, int) or num_sequences_per_batch <= 0:
        raise ValueError("num_sequences_per_batch must be a positive integer")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Designed to accommodate:
            # - standalone functions or bound methods of a class instance
            # - function calls that pass the array as a positional argument or as a keyword argument

            # If the decorated function is bound to a class (i.e., has 'self'), we need to skip 'self' when processing arguments.
            # Check if the first argument is 'self' of a class instance.
            if inspect.ismethod(func) or (
                inspect.isfunction(func)
                and "self" in inspect.signature(func).parameters
            ):
                # Are we decorating a function in a class, which will become a "method" once the class is initialized?
                # Here we detected that the 'self' is present, the function is a method (or an unbound method - the second part of the if statement).
                # Therefore the decorated function args will be (self, arr), rather than (arr), so we want to use second argument to get the actual array.
                # (Note another edge case: arr may actually be passed as a kwarg, not an arg)
                skip = 1
            else:
                # Or are we decorating a method, i.e. a function belonging to an already-created instance of a class?
                # Here we detected that no 'self' is present, so this is either a static method, class method, or function completely outside a class.
                # Therefore the very first argument is the array. (Again note the edge case that the arr may be passed as a kwarg, not an arg)
                skip = 0

            # The sequences to be batched may be provided as a positional argument or as a keyword argument.
            # If keyword argument, it should be the first positional argument after 'self' (if 'self' is included):
            batchable_arg_name = list(inspect.signature(func).parameters)[skip]
            if batchable_arg_name in kwargs:
                # The sequences to be batched were provided as a keyword argument.
                arr = kwargs.pop(batchable_arg_name)
                # Remaining keyword arguments will be passed through.
                # All the positional arguments are passed through as-is.
            else:
                # The sequences to be batched were provided as a positional argument.
                arr = args[skip]
                # Keep all other positional arguments. Note that if skip=1, then original args was (self, arr, *args[2:])
                args = args[:skip] + args[skip + 1 :]

            # Generate batch outputs
            batch_outputs = [
                func(*args, **{batchable_arg_name: batch_grp}, **kwargs)
                for batch_grp in more_itertools.sliced(arr, num_sequences_per_batch)
            ]

            # Combine batches
            return np.concatenate(batch_outputs, axis=0)

        return wrapper

    return decorator
