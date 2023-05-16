from typing import Callable

import jax.numpy as jnp
from jax.scipy.special import erfc, ndtri

from flowjax.utils import Array
from flowjax.bijections import Bijection
from flowjax.bijections import domain_transformations
"""
The extreme transformations, used to define extreme value bijections real -> real
"""
MIN_ERF_INV = 5e-7
# this is the maximum value which we want to transform, this is a very low probability from a normal base
# any larger value would introduce numerical instability
MAX_EXTREME = 12.5 
def _erfcinv(x):
    x = jnp.clip(x, a_min=MIN_ERF_INV) # Should be in (0, 1]
    return -ndtri(0.5 * x) / jnp.sqrt(2)

def _extreme_transform(z, tail_param):
    """
    Forward transformation (0, inf) -> (0, inf)
    tail_param > 0
    """
    z = jnp.clip(z, a_max=MAX_EXTREME)
    g = erfc(z / jnp.sqrt(2))
    x = (jnp.power(g, -tail_param) - 1) / tail_param
    return x

def _extreme_transform_and_lad(z, tail_param):
    """
    Forward transformation (0, inf) -> (0, inf)
    also provides log abs det jacobian
    tail_param > 0
    """
    z = jnp.clip(z, a_max=MAX_EXTREME)
    g = erfc(z / jnp.sqrt(2))
    x = (jnp.power(g, -tail_param) - 1) / tail_param
    
    lad = jnp.log(g) * (-tail_param -1)
    lad -= 0.5 * jnp.square(z) 
    lad += jnp.log(jnp.sqrt(2) / jnp.sqrt(jnp.pi))

    return x, lad

def _extreme_inverse(x, tail_param):
    """
    Inverse transformation (0, inf) -> (0, inf)
    """
    g = jnp.power(1 + tail_param * x, -1 / tail_param)
    return jnp.sqrt(2) * _erfcinv(g) 

def _extreme_inverse_and_lad(x, tail_param):
    inner = 1 + tail_param * x
    g = jnp.power(inner, -1 / tail_param)
    transformed = jnp.sqrt(2) * _erfcinv(g)

    lad = (-1 - 1 / tail_param) * jnp.log(inner)
    lad += jnp.log(0.5 * jnp.sqrt(2) * jnp.sqrt(jnp.pi))
    lad += jnp.square(_erfcinv(g))
    return transformed, lad

"""
The tail transformations, used to define extreme value bijections [-1, 1] -> real
"""
def _tail_transform(u, tail_param):
    """
    Forward transformation (0, 1) -> (0, inf)
    Generating pareto like tails
    """
    transformed = jnp.power(1 - u, -tail_param) - 1
    transformed /= tail_param
    return transformed

def _tail_transform_and_lad(u, tail_param):
    """
    Forward transformation (0, 1) -> (0, inf)
    Generating pareto like tails
    """
    transformed = jnp.power(1 - u, -tail_param) - 1
    transformed /= tail_param
    lad = (-tail_param - 1) * jnp.log(1 - u)
    return transformed, lad

def _tail_inverse(x, tail_param):
    neg_tail_inv = -1 / tail_param
    u = 1 - jnp.power(1 + tail_param * x, neg_tail_inv)
    return u

def _tail_inverse_and_lad(x, tail_param):
    neg_tail_inv = -1 / tail_param
    u = 1 - jnp.power(1 + tail_param * x, neg_tail_inv)
    lad = (neg_tail_inv - 1) * jnp.log(1 + tail_param * x)
    return u, lad

"""
The power transformations, used to define chi2-like bijections real->real
"""
def _shift_power_transform(z, tail_param):
    return jnp.power(1 + z / tail_param, tail_param) - 1

def _shift_power_transform_and_lad(z, tail_param):
    transformed = jnp.power(1 + z / tail_param, tail_param) - 1
    lad = (tail_param - 1) * jnp.log(1 + z / tail_param)
    return transformed, lad

def _shift_power_inverse(x, tail_param):
    return tail_param * (jnp.power(1 + x, 1 / tail_param) - 1)
           
def _shift_power_inverse_and_lad(x, tail_param):
    transformed = tail_param * (jnp.power(1 + x, 1 / tail_param) - 1)
    lad = ((1 / tail_param) - 1) * jnp.log(1 + x)
    return transformed, lad

"""
The sinh-arcsinh transformations
"""
def _sinh_arcsinh_transform(z, tail_param):
    return jnp.sinh(tail_param * jnp.arcsinh(z))

def _sinh_arcsinh_transform_and_lad(z, tail_param):
    x = jnp.sinh(tail_param * jnp.arcsinh(z))
    lad = jnp.log(jnp.cosh((jnp.arcsinh(x)) / tail_param)) 
    lad -= jnp.log(tail_param)
    lad -= 0.5 * jnp.log(1 + x ** 2)
    return x, -lad

def _sinh_arcsinh_inverse(x, tail_param):
    return jnp.sinh(jnp.arcsinh(x) / tail_param)
           
def _sinh_arcsinh_inverse_and_lad(x, tail_param):
    z = jnp.sinh(jnp.arcsinh(x) / tail_param)
    lad = jnp.log(jnp.cosh((jnp.arcsinh(x)) / tail_param)) 
    lad -= jnp.log(tail_param)
    lad -= 0.5 * jnp.log(1 + x ** 2)
    return z, lad

"""
The extreme value bijection classes which tie together the above 
scalar transformations.
"""
class ExtremeValueActivation(Bijection):
    """
    ExtremeValueActivation (D. Prangle, T. Hickling)
    This transform is Reals -> Reals.
    """
    MIN_TAIL_PARAM = 1e-3
    pos_tail_unc: Array
    neg_tail_unc: Array
    parameter_transformation: Callable
    def __init__(self, pos_tail_init, neg_tail_init, parameter_transformation=domain_transformations.Softplus):
        self.shape = ()
        self.cond_shape = None
        self.parameter_transformation = parameter_transformation(self.MIN_TAIL_PARAM)
        self.pos_tail_unc = self.parameter_transformation.inverse(pos_tail_init)
        self.neg_tail_unc = self.parameter_transformation.inverse(neg_tail_init)

    @property
    def pos_tail(self):
        return self.parameter_transformation.transform(self.pos_tail_unc)
    
    @property
    def neg_tail(self):
        return self.parameter_transformation.transform(self.neg_tail_unc)

    def transform(self, z, condition):
        sign = jnp.sign(z)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        x = sign * _extreme_transform(jnp.abs(z), tail_param)
        return x

    def transform_and_log_abs_det_jacobian(self, z, condition):
        sign = jnp.sign(z)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        x, lad = _extreme_transform_and_lad(jnp.abs(z), tail_param)
        return  sign * x, lad

    def inverse(self, x, condition):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        z = sign * _extreme_inverse(jnp.abs(x), tail_param)
        return z

    def inverse_and_log_abs_det_jacobian(self, x, condition):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        z, lad = _extreme_inverse_and_lad(jnp.abs(x), tail_param)
        return sign * z, lad


class SinhArcsinh(Bijection):
    """
    SinhArcsinh (T. Hickling)
    This transform is Reals -> Reals.
    """
    MIN_TAIL_PARAM = 0.99
    tail_unc: Array
    parameter_transformation: Callable
    def __init__(self, tail_init, parameter_transformation=domain_transformations.Softplus):
        self.shape = ()
        self.cond_shape = None
        self.parameter_transformation = parameter_transformation(self.MIN_TAIL_PARAM)
        self.tail_unc = self.parameter_transformation.inverse(tail_init)

    @property
    def tail_param(self):
        return self.parameter_transformation.transform(self.tail_unc)

    def transform(self, z, condition):
        return _sinh_arcsinh_transform(z, self.tail_param)

    def transform_and_log_abs_det_jacobian(self, z, condition):
        return _sinh_arcsinh_transform_and_lad(z, self.tail_param)

    def inverse(self, x, condition):
        return _sinh_arcsinh_inverse(x, self.tail_param)

    def inverse_and_log_abs_det_jacobian(self, x, condition):
        return _sinh_arcsinh_inverse_and_lad(x, self.tail_param)


class TailTransformation(Bijection):
    """
    TailTransformation (D. Prangle, T. Hickling)

    The transform is scalar, from (-1, 1) -> Reals.
    """
    MIN_TAIL_PARAM = 1e-8
    pos_tail_unc: Array
    neg_tail_unc: Array
    target_domain: tuple[float, float]
    parameter_transformation: Callable
    def __init__(self, pos_tail_init, neg_tail_init, target_domain=(-1, 1)):
        self.parameter_transformation = domain_transformations.Softplus(min=self.MIN_TAIL_PARAM)
        self.pos_tail_unc = self.parameter_transformation.inverse(pos_tail_init)
        self.neg_tail_unc = self.parameter_transformation.inverse(neg_tail_init)
        self.shape = ()
        self.cond_shape = None
        self.target_domain = target_domain

    @property
    def pos_tail(self):
        return self.parameter_transformation.transform(self.pos_tail_unc)
    
    @property
    def neg_tail(self):
        return self.parameter_transformation.transform(self.neg_tail_unc)

    def transform(self, u, condition=None):
        sign = jnp.sign(u)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        x = sign * _tail_transform(jnp.abs(u), tail_param)
        return x

    def transform_and_log_abs_det_jacobian(self, u, condition=None):
        sign = jnp.sign(u)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        x, lad =  _tail_transform_and_lad(jnp.abs(u), tail_param)
        return sign * x, lad

    def inverse(self, x, condition=None):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        u = _tail_inverse(jnp.abs(x), tail_param)
        u = (u - self.target_domain[0]) / (self.target_domain[1] - self.target_domain[0])
        return sign * u

    def inverse_and_log_abs_det_jacobian(self, x, condition=None):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        u, lad = _tail_inverse_and_lad(jnp.abs(x), tail_param)
        u = (u - self.target_domain[0]) / (self.target_domain[1] - self.target_domain[0])
        return sign * u, lad - jnp.log(self.target_domain[1] - self.target_domain[0])


class Switch(Bijection):
    pos_tail_unc: Array
    neg_tail_unc: Array
    parameter_transformation: Callable
    def __init__(self, pos_tail_init, neg_tail_init):
        self.shape = ()
        self.cond_shape = None

        self.parameter_transformation = domain_transformations.Softplus(min=-1)
        self.pos_tail_unc = self.parameter_transformation.inverse(pos_tail_init)
        self.neg_tail_unc = self.parameter_transformation.inverse(neg_tail_init)

    """
    Tail parameters are in [-1, inf]
    """
    @property
    def pos_tail(self):
        return self.parameter_transformation.transform(self.pos_tail_unc)
    
    @property
    def neg_tail(self):
        return self.parameter_transformation.transform(self.neg_tail_unc)

    """
    The heavy tail parameter are in [0, inf]
    """
    @property
    def heavy_pos_tail(self):
        return jnp.abs(self.pos_tail)
    
    @property
    def heavy_neg_tail(self):
        return jnp.abs(self.neg_tail)
    
    """
    The light tail parameters are mapped [-1, 0] -> [1, 2]
    """
    @property
    def light_pos_tail(self):
        return jnp.abs(self.pos_tail + 2)
    
    @property
    def light_neg_tail(self):
        return jnp.abs(self.neg_tail + 2)

    def transform(self, z, condition=None):
        sign = jnp.sign(z)
        abs_z = jnp.abs(z)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        light_tail_param = jnp.where(sign > 0, self.light_pos_tail, self.light_neg_tail) # always -ve
        heavy_tail_param = jnp.where(sign > 0, self.heavy_pos_tail, self.heavy_neg_tail) # always -ve # always +ve

        # light transform, param in [-1, 0]
        light_x = sign * _shift_power_transform(abs_z, light_tail_param)

        # heavy transform, param in (0, )
        heavy_x = sign * _extreme_transform(abs_z, heavy_tail_param)

        return jnp.where(tail_param > 0, heavy_x, light_x)

    def transform_and_log_abs_det_jacobian(self, z, condition=None):
        sign = jnp.sign(z)
        abs_z = jnp.abs(z)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        light_tail_param = jnp.where(sign > 0, self.light_pos_tail, self.light_neg_tail) # always -ve
        heavy_tail_param = jnp.where(sign > 0, self.heavy_pos_tail, self.heavy_neg_tail) # always -ve # always +ve

        # light transform, param in [-1, 0]
        light_x, light_lad = _shift_power_transform_and_lad(abs_z, light_tail_param)

        # heavy transform, param in (0, )
        heavy_x, heavy_lad =  _extreme_transform_and_lad(abs_z, heavy_tail_param)

        # select correct transformation
        x = sign * jnp.where(tail_param > 0, heavy_x, light_x)
        lad = jnp.where(tail_param > 0, heavy_lad, light_lad)
        return x, lad

    def inverse(self, x, condition=None):
        sign = jnp.sign(x)
        abs_x = jnp.abs(x)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        light_tail_param = jnp.where(sign > 0, self.light_pos_tail, self.light_neg_tail) # always -ve
        heavy_tail_param = jnp.where(sign > 0, self.heavy_pos_tail, self.heavy_neg_tail) # always -ve # always +ve

        # light transform, param in [-1, 0]
        light_z = sign * _shift_power_inverse(abs_x, light_tail_param)

        # heavy transform, param in (0, )
        heavy_z = sign * _extreme_inverse(abs_x, heavy_tail_param)

        return jnp.where(tail_param > 0, heavy_z, light_z)

    def inverse_and_log_abs_det_jacobian(self, x, condition=None):
        sign = jnp.sign(x)
        abs_x = jnp.abs(x)
        tail_param = jnp.where(sign > 0, self.pos_tail, self.neg_tail)
        light_tail_param = jnp.where(sign > 0, self.light_pos_tail, self.light_neg_tail) # always -ve
        heavy_tail_param = jnp.where(sign > 0, self.heavy_pos_tail, self.heavy_neg_tail) # always -ve # always +ve
        
        # light transform, param in [-1, 0]
        light_z, light_lad =  _shift_power_inverse_and_lad(abs_x, light_tail_param)

        # heavy transform, param in (0, )
        # always use abs even though when negative we don't end up apply the transformation
        heavy_z, heavy_lad = _extreme_inverse_and_lad(abs_x, heavy_tail_param)

        # select correct transformation
        z = sign * jnp.where(tail_param > 0, heavy_z, light_z)
        lad = jnp.where(tail_param > 0, heavy_lad, light_lad)

        return z, lad


class Kuma(Bijection):
    MIN_PARAM = 0.01
    a_unc: Array
    b_unc: Array
    min: float
    max: float
    parameter_transformation: Callable
    def __init__(self, a_init, b_init, parameter_transformation=domain_transformations.Softplus):
        self.shape = ()
        self.cond_shape = None
        self.parameter_transformation = parameter_transformation(self.MIN_PARAM)
        self.a_unc = self.parameter_transformation.inverse(a_init)
        self.b_unc = self.parameter_transformation.inverse(b_init)
        self.min = -1.
        self.max = 1.

    @property
    def a(self):
        return self.parameter_transformation.transform(self.a_unc)
    
    @property
    def b(self):
        return self.parameter_transformation.transform(self.b_unc)

    def transform(self, z, condition):
        inner = jnp.power(1 - z, 1 /  self.b)
        x = jnp.power(1 - inner, 1 / self.a)
        return self.min + x * (self.max - self.min)

    def transform_and_log_abs_det_jacobian(self, z, condition):
        inner = jnp.power(1 - z, 1 /  self.b)
        x = jnp.power(1 - inner, 1 / self.a)
        lad = (self.b - 1) * jnp.log(1 - jnp.power(x, self.a))
        lad += jnp.log(self.a * self.b)
        lad += (self.a - 1) * jnp.log(x)
        lad -= jnp.log(self.max - self.min)
        x = self.min + x * (self.max - self.min)
        return x, -lad
    
    def inverse(self, x, condition):
        x = (x - self.min) / (self.max - self.min)
        u = 1 - jnp.power(1 - jnp.power(x, self.a), self.b)
        return u

    def inverse_and_log_abs_det_jacobian(self, x, condition):
        x = jnp.clip(
            (x - self.min) / (self.max - self.min),
            a_min=1e-4,
            a_max=1 - 1e-4,
        )
        u = 1 - jnp.power(1 - jnp.power(x, self.a), self.b)
        lad = (self.b - 1) * jnp.log(1 - jnp.power(x, self.a))
        lad += jnp.log(self.a * self.b)
        lad += (self.a - 1) * jnp.log(x)
        lad -= jnp.log(self.max - self.min)
        return u, lad