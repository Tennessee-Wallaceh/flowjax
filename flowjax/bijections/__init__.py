"""Bijections from ``flowjax.bijections``."""

from .affine import AdditiveCondition, Affine, Loc, Scale, TriangularAffine, UnitLULinear, LULinear
from .bijection import AbstractBijection
from .block_autoregressive_network import BlockAutoregressiveNetwork
from .concatenate import Concatenate, Stack
from .coupling import Coupling
from .exp import Exp
from .jax_transforms import Vmap
from .scan import scan
from .invert import invert
from .chain import chain
from .masked_autoregressive import MaskedAutoregressive
# from .monotonic_residual import (
#     DeepMonotonicResidual,
#     MonotonicResidual,
#     TriangularDeepMonotonicResidual,
# )
from .orthogonal import DiscreteCosine, Householder
from .planar import Planar
from .power import Power
from .rational_quadratic_spline import RationalQuadraticSpline
from .sigmoid import Sigmoid
from .softplus import SoftPlus
from .tanh import LeakyTanh, Tanh
from .utils import (
    EmbedCondition,
    Flip,
    Identity,
    Indexed,
    NumericalInverse,
    Permute,
    Reshape,
    Sandwich,
)

__all__ = [
    "AdditiveCondition",
    "Affine",
    "AbstractBijection",
    "BlockAutoregressiveNetwork",
    "chain",
    "Concatenate",
    "Coupling",
    "DiscreteCosine",
    "EmbedCondition",
    "Exp",
    "Flip",
    "Householder",
    "Identity",
    "invert",
    "LeakyTanh",
    "Loc",
    "LULinear",
    "UnitLULinear",
    "MaskedAutoregressive",
    "Indexed",
    "Permute",
    "Power",
    "Planar",
    "RationalQuadraticSpline",
    "Reshape",
    "Sandwich",
    "Scale",
    "scan",
    "Sigmoid",
    "SoftPlus",
    "Stack",
    "Tanh",
    "TriangularAffine",
    "Vmap",
    "NumericalInverse",
]
