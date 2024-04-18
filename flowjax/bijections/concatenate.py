"""Module contains bijections formed by "stacking/concatenating" other bijections."""

from collections.abc import Sequence
from itertools import accumulate

import jax.numpy as jnp
from jaxtyping import Array

from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import check_shapes_match, merge_cond_shapes


class Concatenate(AbstractBijection):
    """Concatenate bijections along an existing axis, similar to ``jnp.concatenate``.

    See also :class:`Stack`.

    Args:
        bijections: Bijections, to stack into a single bijection.
        axis: Axis along which to stack. Defaults to 0.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    split_idxs: tuple[int, ...]
    bijections: Sequence[AbstractBijection]
    axis: int

    def __init__(self, bijections: Sequence[AbstractBijection], axis: int = 0):
        self.bijections = bijections
        self.axis = axis

        shapes = [b.shape for b in bijections]
        self._argcheck_shapes(shapes)
        axis = range(len(shapes[0]))[axis]  # Avoids issues when axis==-1
        self.shape = (
            shapes[0][:axis] + (sum(s[axis] for s in shapes),) + shapes[0][axis + 1 :]
        )
        self.split_idxs = tuple(accumulate([s[axis] for s in shapes[:-1]]))
        self.cond_shape = merge_cond_shapes([b.cond_shape for b in bijections])

    def transform(self, x, condition=None):
        x_parts = jnp.array_split(x, self.split_idxs, axis=self.axis)
        y_parts = [
            b.transform(x_part, condition)
            for b, x_part in zip(self.bijections, x_parts, strict=True)
        ]
        return jnp.concatenate(y_parts, axis=self.axis)

    def transform_and_log_det(self, x, condition=None):
        x_parts = jnp.array_split(x, self.split_idxs, axis=self.axis)

        ys_log_dets = [
            b.transform_and_log_det(x, condition)
            for b, x in zip(self.bijections, x_parts, strict=True)
        ]

        y_parts, log_dets = zip(*ys_log_dets, strict=True)
        return jnp.concatenate(y_parts, self.axis), sum(log_dets)

    def inverse(self, y, condition=None):
        y_parts = jnp.array_split(y, self.split_idxs, axis=self.axis)
        x_parts = [
            b.inverse(y_part, condition)
            for b, y_part in zip(self.bijections, y_parts, strict=True)
        ]
        return jnp.concatenate(x_parts, axis=self.axis)

    def inverse_and_log_det(self, y, condition=None):
        y_parts = jnp.array_split(y, self.split_idxs, axis=self.axis)

        xs_log_dets = [
            b.inverse_and_log_det(y, condition)
            for b, y in zip(self.bijections, y_parts, strict=True)
        ]

        x_parts, log_dets = zip(*xs_log_dets, strict=True)
        return jnp.concatenate(x_parts, self.axis), sum(log_dets)

    def _argcheck_shapes(self, shapes: Sequence[tuple[int, ...]]):
        axis = range(len(shapes[0]))[self.axis]  # Avoid negative index
        expected_matching = shapes[0][:axis] + shapes[0][axis + 1 :]
        for i, shape in enumerate(shapes):
            if shape[:axis] + shape[axis + 1 :] != expected_matching:
                raise ValueError(
                    f"Expected bijection shapes to match except along axis {axis}, but "
                    f"index 0 had shape {shapes[0]}, and index {i} had shape {shape}.",
                )


class Stack(AbstractBijection):
    """Stack bijections along a new axis (analagous to ``jnp.stack``).

    See also :class:`Concatenate`.

    Args:
        bijections: Bijections.
        axis: Axis along which to stack. Defaults to 0.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    bijections: Sequence[AbstractBijection]
    axis: int

    def __init__(self, bijections: list[AbstractBijection], axis: int = 0):
        self.axis = axis
        self.bijections = bijections

        shapes = [b.shape for b in bijections]
        check_shapes_match(shapes)

        self.shape = shapes[0][:axis] + (len(bijections),) + shapes[0][axis:]
        self.cond_shape = merge_cond_shapes([b.cond_shape for b in bijections])

    def transform(self, x, condition=None):
        x_parts = self._split_and_squeeze(x)
        y_parts = [
            b.transform(x, condition)
            for (b, x) in zip(self.bijections, x_parts, strict=True)
        ]
        return jnp.stack(y_parts, self.axis)

    def transform_and_log_det(self, x, condition=None):
        x_parts = self._split_and_squeeze(x)
        ys_log_det = [
            b.transform_and_log_det(x, condition)
            for b, x in zip(self.bijections, x_parts, strict=True)
        ]

        y_parts, log_dets = zip(*ys_log_det, strict=True)
        return jnp.stack(y_parts, self.axis), sum(log_dets)

    def inverse(self, y, condition=None):
        y_parts = self._split_and_squeeze(y)
        x_parts = [
            b.inverse(y, condition)
            for (b, y) in zip(self.bijections, y_parts, strict=True)
        ]
        return jnp.stack(x_parts, self.axis)

    def inverse_and_log_det(self, y, condition=None):
        y_parts = self._split_and_squeeze(y)
        xs_log_det = [
            b.inverse_and_log_det(y, condition)
            for b, y in zip(self.bijections, y_parts, strict=True)
        ]
        x_parts, log_dets = zip(*xs_log_det, strict=True)
        return jnp.stack(x_parts, self.axis), sum(log_dets)

    def _split_and_squeeze(self, array: Array):
        arrays = jnp.split(array, len(self.bijections), axis=self.axis)
        return (a.squeeze(axis=self.axis) for a in arrays)
