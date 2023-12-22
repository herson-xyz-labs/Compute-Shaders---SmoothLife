#!/usr/bin/env python3

"""SmoothLife -- Conway's GoL in continuous space
Re-written in Python using the speedups of Numpy
"""

import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

def logistic_threshold(x, x0, alpha):
    return 1.0 / (1.0 + np.exp(-4.0 / alpha * (x - x0)))

def logistic_interval(x, a, b, alpha):
    return logistic_threshold(x, a, alpha) * (1.0 - logistic_threshold(x, b, alpha))

def lerp(a, b, t):
    return (1.0 - t) * a + t * b

class BasicRules:
    B1 = 0.278
    B2 = 0.365
    D1 = 0.267
    D2 = 0.445
    N = 0.028
    M = 0.147

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError("Unexpected attribute %s" % k)

    def clear(self):
        pass

    def s(self, n, m, field):
        aliveness = logistic_threshold(m, 0.5, self.M)
        threshold1 = lerp(self.B1, self.D1, aliveness)
        threshold2 = lerp(self.B2, self.D2, aliveness)
        new_aliveness = logistic_interval(n, threshold1, threshold2, self.N)

        return np.clip(new_aliveness, 0, 1)

def antialiased_circle(size, radius, roll=True, logres=None):
    y, x = size
    yy, xx = np.mgrid[:y, :x]
    radiuses = np.sqrt((xx - x / 2) ** 2 + (yy - y / 2) ** 2)
    if logres is None:
        logres = math.log(min(*size), 2)
    with np.errstate(over="ignore"):
        logistic = 1 / (1 + np.exp(logres * (radiuses - radius)))
    if roll:
        logistic = np.roll(logistic, y // 2, axis=0)
        logistic = np.roll(logistic, x // 2, axis=1)
    return logistic

class Multipliers:
    INNER_RADIUS = 7.0
    OUTER_RADIUS = INNER_RADIUS * 3.0

    def __init__(self, size, inner_radius=INNER_RADIUS, outer_radius=OUTER_RADIUS):
        inner = antialiased_circle(size, inner_radius)
        outer = antialiased_circle(size, outer_radius)
        annulus = outer - inner

        inner /= np.sum(inner)
        annulus /= np.sum(annulus)

        self.M = np.fft.fft2(inner)
        self.N = np.fft.fft2(annulus)

class SmoothLife:
    def __init__(self, height, width):
        self.width = width
        self.height = height

        self.multipliers = Multipliers((height, width))

        self.rules = BasicRules()

        self.clear()

    def clear(self):
        self.field = np.zeros((self.height, self.width))
        self.rules.clear()

    def step(self):
        field_ = np.fft.fft2(self.field)
        M_buffer_ = field_ * self.multipliers.M
        N_buffer_ = field_ * self.multipliers.N
        M_buffer = np.real(np.fft.ifft2(M_buffer_))
        N_buffer = np.real(np.fft.ifft2(N_buffer_))

        self.field = self.rules.s(N_buffer, M_buffer, self.field)
        return self.field

    def add_speckles(self, count=None, intensity=1):
        if count is None:
            count = int(
                self.width * self.height / ((self.multipliers.OUTER_RADIUS * 2) ** 2)
            )
        for i in range(count):
            radius = int(self.multipliers.OUTER_RADIUS)
            r = np.random.randint(0, self.height - radius)
            c = np.random.randint(0, self.width - radius)
            self.field[r : r + radius, c : c + radius] = intensity

def show_animation():
    w = 1 << 9
    h = 1 << 9
    sl = SmoothLife(h, w)
    sl.add_speckles()
    sl.step()

    fig = plt.figure()
    im = plt.imshow(
        sl.field, animated=True, cmap=plt.get_cmap("viridis"), aspect="equal"
    )

    def animate(*args):
        im.set_array(sl.step())
        return (im,)

    ani = animation.FuncAnimation(fig, animate, interval=60, blit=True)
    plt.show()

if __name__ == "__main__":
    show_animation()
