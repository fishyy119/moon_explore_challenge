"""
Reeds Shepp 曲线的几种基本模式
"""

import math
from functools import wraps
from typing import Callable, Dict, List, NamedTuple, Tuple

import numpy as np


class PathFunctionResult(NamedTuple):
    success: bool
    distances: List[float]
    directions: List[str]


PathFunction = Callable[[float, float, float], PathFunctionResult]
RawPathFunction = Callable[[float, float, float], Tuple[bool, List[float], List[str]]]
path_funcs: Dict[str, PathFunction] = {}


# 在原始实现基础上的一层封装
def wrap_as_result(fn: RawPathFunction) -> PathFunction:
    @wraps(fn)
    def wrapper(x: float, y: float, phi: float) -> PathFunctionResult:
        success, params, path = fn(x, y, phi)
        return PathFunctionResult(success, params, path)

    return wrapper


# 用于注册各个基本函数到`path_funcs`
def register(name: str):
    def decorator(fn: RawPathFunction):
        if name in path_funcs:
            raise ValueError(f"[register] Duplicate registration: '{name}' already exists.")
        path_funcs[name] = wrap_as_result(fn)
        return path_funcs[name]

    return decorator


def polar(x, y):
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return r, theta


def mod2pi(x: float) -> float:
    # Be consistent with fmod in cplusplus here.
    v = np.mod(x, np.copysign(2.0 * math.pi, x))
    if v < -math.pi:
        v += 2.0 * math.pi
    else:
        if v > math.pi:
            v -= 2.0 * math.pi
    return v


@register("CSC1")
def left_straight_left(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ["L", "S", "L"]

    return False, [], []


@register("CSC2")
def left_straight_right(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1**2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)

        if (t >= 0.0) and (v >= 0.0):
            return True, [t, u, v], ["L", "S", "R"]

    return False, [], []


@register("CCC1")
def left_x_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi / 2)
        u = mod2pi(math.pi - 2 * A)
        v = mod2pi(phi - t - u)
        return True, [t, -u, v], ["L", "R", "L"]

    return False, [], []


@register("CCC2")
def left_x_right_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi / 2)
        u = mod2pi(math.pi - 2 * A)
        v = mod2pi(-phi + t + u)
        return True, [t, -u, -v], ["L", "R", "L"]

    return False, [], []


@register("CCC3")
def left_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        u = math.acos(1 - u1**2 * 0.125)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(-A + theta + math.pi / 2)
        v = mod2pi(t - u - phi)
        return True, [t, u, -v], ["L", "R", "L"]

    return False, [], []


@register("CCCC1")
def left_right_x_left_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    # Solutions refering to (2 < u1 <= 4) are considered sub-optimal in paper
    # Solutions do not exist for u1 > 4
    if u1 <= 2:
        A = math.acos((u1 + 2) * 0.25)
        t = mod2pi(theta + A + math.pi / 2)
        u = mod2pi(A)
        v = mod2pi(phi - t + 2 * u)
        if (t >= 0) and (u >= 0) and (v >= 0):
            return True, [t, u, -u, -v], ["L", "R", "L", "R"]

    return False, [], []


@register("CCCC2")
def left_x_right_left_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    u2 = (20 - u1**2) / 16

    if 0 <= u2 <= 1:
        u = math.acos(u2)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(theta + A + math.pi / 2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -u, -u, v], ["L", "R", "L", "R"]

    return False, [], []


@register("CSCC1")
def left_straight_right90_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(math.sqrt(u1**2 - 4), 2)
        t = mod2pi(theta - A + math.pi / 2)
        v = mod2pi(t - phi - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi / 2, -v], ["L", "S", "R", "L"]

    return False, [], []


@register("CSCC2")
def left_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi / 2, -v], ["L", "S", "L", "R"]

    return False, [], []


@register("CCSC1")
def left_x_right90_straight_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi / 2)
        v = mod2pi(t - phi + math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi / 2, -u, -v], ["L", "R", "S", "L"]

    return False, [], []


@register("CCSC2")
def left_x_right90_straight_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta + math.pi / 2)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi / 2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi / 2, -u, -v], ["L", "R", "S", "R"]

    return False, [], []


@register("CCSCC")
def left_x_right90_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 4.0:
        u = math.sqrt(u1**2 - 4) - 4
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi / 2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi / 2, -u, -math.pi / 2, v], ["L", "R", "S", "L", "R"]

    return False, [], []
