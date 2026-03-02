# coding=utf-8
"""
Lorentz model operations for hyperbolic geometry, following the MERU implementation.

All functions input/output only the SPACE components. The time component is
computed on-the-fly from the hyperboloid constraint:
    x_time = sqrt(1 / curv + ||x_space||^2)

This avoids storing the redundant time dimension and simplifies the interface.

Reference: MERU (Meta Platforms, Inc.)
"""
from __future__ import annotations

import math
import torch
from torch import Tensor


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """
    Pairwise Lorentzian inner product.
    Args:
        x: (B1, D) space components
        y: (B2, D) space components
        curv: positive scalar (negative curvature = -curv)
    Returns:
        (B1, B2) Lorentzian inner products
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y ** 2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Pairwise geodesic distance on the hyperboloid.
    Args:
        x: (B1, D) space components
        y: (B2, D) space components
    Returns:
        (B1, B2) geodesic distances
    """
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv ** 0.5


def dist_to_origin(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Geodesic distance from each point to the hyperboloid origin.
    Args:
        x: (B, D) space components
    Returns:
        (B,) distances to origin
    """
    # Inner product with origin: origin has x_space=0, x_time=1/sqrt(curv)
    # <x, origin>_L = -x_time * (1/sqrt(curv))
    x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1))
    # c * <x, origin>_L = curv * x_time / sqrt(curv) = sqrt(curv) * x_time
    rc_x_time = torch.sqrt(curv) * x_time
    _distance = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    return _distance / curv ** 0.5


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Exponential map at the hyperboloid origin: tangent space -> hyperboloid.
    Args:
        x: (B, D) Euclidean vectors (tangent vectors at origin)
        curv: positive curvature scalar
    Returns:
        (B, D) space components on the hyperboloid
    """
    rc_xnorm = curv ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2 ** 15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Logarithmic map at the hyperboloid origin: hyperboloid -> tangent space.
    Args:
        x: (B, D) space components on the hyperboloid
    Returns:
        (B, D) Euclidean vectors in the tangent space at origin
    """
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x ** 2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    rc_xnorm = curv ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """
    Half-aperture angle of the entailment cone at x.
    Args:
        x: (B, D) space components
        curv: positive curvature scalar
        min_radius: minimum neighborhood radius around origin
    Returns:
        (B,) half-aperture angles in (0, pi/2)
    """
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv ** 0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))
    return _half_aperture


def oxy_angle(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Exterior angle at x in the hyperbolic triangle Oxy (O = origin).
    Derived from the hyperbolic law of cosines.
    Args:
        x: (B, D) space components
        y: (B, D) space components (same batch size)
    Returns:
        (B,) angles in (0, pi)
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y ** 2, dim=-1))

    # Lorentzian inner product * curvature (diagonal only)
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl ** 2 - 1, min=eps))
    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
    return _angle