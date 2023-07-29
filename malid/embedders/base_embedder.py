import abc
import functools
import inspect
from typing_extensions import Self

import numpy as np
import pandas as pd

from malid.datamodels import GeneLocus


# Abstract class. Anything marked @abc.abstractmethod is required in implementations.


class BaseEmbedder(metaclass=abc.ABCMeta):
    @property
    def is_fine_tuned(self) -> bool:
        return False

    def load_fine_tuned_parameters(
        self: Self, fold_id: int, gene_locus: GeneLocus
    ) -> Self:
        # Calling this is a mistake. Should only be called if is_fine_tuned is True.
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self):
        # TODO: instead of hardcoding a string for each, just use class name
        pass

    @abc.abstractmethod
    def embed(self, arr: np.ndarray) -> np.ndarray:
        pass


class BaseFineTunedEmbedder(BaseEmbedder):
    @property
    def is_fine_tuned(self) -> bool:
        return True

    @abc.abstractmethod
    def load_fine_tuned_parameters(
        self: Self, fold_id: int, gene_locus: GeneLocus
    ) -> Self:
        # Will be called because is_fine_tuned is True.
        pass


# Run in batches of N sequences at a time - can't allocate enough memory (even on CPU) for running all sequences at once.
# TODO: will batches have different token lengths (different amounts of padding?)
# TODO: look at FastaBatchedDataset pattern from ESM - where they batch by #tokens not #sequences?


def _make_batches(arr, num_sequences_per_batch: int):
    for batch_id, batch_grp in pd.Series(arr).groupby(
        np.arange(len(arr)) // num_sequences_per_batch
    ):
        yield batch_grp.values


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
    ```
    """
    #
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if (not inspect.ismethod(func)) and "self" in inspect.signature(
                func
            ).parameters:
                # Are we decorating a function in a class, which will become a "method" once the class is initialized?
                # If so, the decorated function args will be (self, arr), rather than (arr), so we want to use second argument to get the actual array.
                # args[0] is "self" and args[1] is the actual array.
                batch_outputs = [
                    func(args[0], batch_grp, *args[2:], **kwargs)
                    for batch_grp in _make_batches(
                        args[1], num_sequences_per_batch=num_sequences_per_batch
                    )
                ]

            else:
                # Or are we decorating a method, i.e. a function belonging to an already-created instance of a class?
                # Then we get (arr) as arguments -- no self.
                # And "self" will not be in inspect.signature(func).parameters, inspect.ismethod(func) will be True, but "self" will be in inspect.getfullargspec(func)

                # Or are we decorating a function that is completely outside a class, in which case we have (arr) as arguments?
                # Then inspect.ismethod(func) will be False and "self" will not be in inspect.signature(func).parameters

                # Not decorating a method. The very first argument is the array.
                batch_outputs = [
                    func(batch_grp, *args[1:], **kwargs)
                    for batch_grp in _make_batches(
                        args[0], num_sequences_per_batch=num_sequences_per_batch
                    )
                ]

            # combine batches
            return np.vstack(batch_outputs)

        return wrapper

    return decorator
