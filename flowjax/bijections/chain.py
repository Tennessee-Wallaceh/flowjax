"""Chain bijection which allows sequential application of arbitrary bijections."""
from typing import Sequence

from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import check_shapes_match, merge_cond_shapes


class Chain(AbstractBijection):
    """Chain together arbitrary bijections to form another bijection."""

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    bijections: tuple[AbstractBijection]

    def __init__(self, bijections: Sequence[AbstractBijection]):
        """Initialize the chain bijection.

        Args:
            bijections (Sequence[Bijection]): Sequence of bijections. The bijection
            shapes must match, and any none None condition shapes must match.

        """
        check_shapes_match([b.shape for b in bijections])
        self.shape = bijections[0].shape
        self.cond_shape = merge_cond_shapes([b.cond_shape for b in bijections])
        self.bijections = tuple(bijections)

    def transform(self, x, condition=None):
        for bijection in self.bijections:
            x = bijection.transform(x, condition)
        return x

    def transform_and_log_det(self, x, condition=None):
        log_abs_det_jac = 0
        for bijection in self.bijections:
            x, log_abs_det_jac_i = bijection.transform_and_log_det(x, condition)
            log_abs_det_jac += log_abs_det_jac_i.sum()
        return x, log_abs_det_jac

    def inverse(self, y, condition=None):
        for bijection in reversed(self.bijections):
            y = bijection.inverse(y, condition)
        return y

    def inverse_and_log_det(self, y, condition=None):
        log_abs_det_jac = 0
        for bijection in reversed(self.bijections):
            y, log_abs_det_jac_i = bijection.inverse_and_log_det(y, condition)
            log_abs_det_jac += log_abs_det_jac_i.sum()
        return y, log_abs_det_jac

    def __getitem__(self, i: int | slice) -> AbstractBijection:
        if isinstance(i, int):
            return self.bijections[i]
        if isinstance(i, slice):
            return Chain(self.bijections[i])
        raise TypeError(f"Indexing with type {type(i)} is not supported.")

    def __iter__(self):
        yield from self.bijections

    def __len__(self):
        return len(self.bijections)

    def merge_chains(self):
        """Returns an equivilent Chain object, in which nested chains are flattened."""
        bijections = self.bijections
        while any(isinstance(b, Chain) for b in bijections):
            bij = []
            for b in bijections:
                if isinstance(b, Chain):
                    bij.extend(b.bijections)
                else:
                    bij.append(b)
            bijections = bij
        return Chain(bijections)
