"""Generation of 2d points for Euclidean approval profiles.

Ported from Martin Lackner's perpetual voting codebase
(experiments/experiments.py: generate_2d_points). Logic is unchanged.
"""

from __future__ import annotations

import random


# generate a list of 2d coordinates subject to
# various distributions
def generate_2d_points(pointids, mode, sigma):

    numpoints = len(pointids)
    points = [0] * numpoints

    # normal distribution, 1/3 of points centered on (-0.5,-0.5),
    #                      2/3 of points on (0.5,0.5)
    #                      all within [-1,1]x[-1,1]
    if mode == "eucl1":

        def within_bounds(point):
            return point[0] <= 1 and point[0] >= -1 and point[1] <= 1 and point[1] >= -1

        for i in range(int(numpoints // 3)):
            while True:
                points[i] = (random.gauss(-0.5, sigma), random.gauss(-0.5, sigma))
                if within_bounds(points[i]):
                    break
        for i in range(numpoints // 3, numpoints):
            while True:
                points[i] = (random.gauss(0.5, sigma), random.gauss(0.5, sigma))
                if within_bounds(points[i]):
                    break
    # normal distribution, 1/3 of points centered on (-0.5,-0.5),
    #                      2/3 of points on (0.5,0.5)
    elif mode == "eucl2":
        for i in range(int(numpoints // 3)):
            points[i] = (random.gauss(-0.5, sigma), random.gauss(-0.5, sigma))
        for i in range(int(numpoints // 3), numpoints):
            points[i] = (random.gauss(0.5, sigma), random.gauss(0.5, sigma))
    # normal distribution, 1/5 of points centered on (-0.5,-0.5),
    #                      4/5 of points on (0.5,0.5)
    elif mode == "eucl4":
        for i in range(numpoints // 5):
            points[i] = (random.gauss(-0.5, sigma), random.gauss(-0.5, sigma))
        for i in range(numpoints // 5, numpoints):
            points[i] = (random.gauss(0.5, sigma), random.gauss(0.5, sigma))
    # normal distribution, 3/5 of points centered on (-0.25,0),
    #                      2/5 of points on (0.25,0)
    elif mode == "eucl6":
        for i in range(2 * numpoints // 5):
            points[i] = (random.gauss(-0.25, sigma), random.gauss(0, sigma))
        for i in range(2 * numpoints // 5, numpoints):
            points[i] = (random.gauss(0.25, sigma), random.gauss(0, sigma))
    # normal distribution
    elif mode == "normal":
        for i in range(numpoints):
            points[i] = (random.gauss(0.0, sigma), random.gauss(0.0, sigma))
    # normal distribution, each 1/4 of points centered on (+-0.5,+-0.5)
    elif mode == "eucl5":
        for i in range(numpoints // 4):
            points[i] = (random.gauss(-0.5, sigma), random.gauss(-0.5, sigma))
        for i in range(numpoints // 4, 2 * numpoints // 4):
            points[i] = (random.gauss(0.5, sigma), random.gauss(0.5, sigma))
        for i in range(2 * numpoints // 4, 3 * numpoints // 4):
            points[i] = (random.gauss(-0.5, sigma), random.gauss(0.5, sigma))
        for i in range(3 * numpoints // 4, numpoints):
            points[i] = (random.gauss(0.5, sigma), random.gauss(-0.5, sigma))
    # normal distribution, 1/5 of points centered on (-0.5,-0.5),
    #                      1/5 of points centered on (-0.5,0.5),
    #                      1/5 of points centered on (0.5,-0.5),
    #                      2/5 of points on (0.5,0.5)
    elif mode == "eucl3":
        for i in range(numpoints // 5):
            points[i] = (random.gauss(-0.5, sigma), random.gauss(-0.5, sigma))
        for i in range(numpoints // 5, 2 * numpoints // 5):
            points[i] = (random.gauss(-0.5, sigma), random.gauss(0.5, sigma))
        for i in range(2 * numpoints // 5, 3 * numpoints // 5):
            points[i] = (random.gauss(0.5, sigma), random.gauss(-0.5, sigma))
        for i in range(3 * numpoints // 5, numpoints):
            points[i] = (random.gauss(0.5, sigma), random.gauss(0.5, sigma))
    elif mode == "eucl2plus":
        for i in range(numpoints // 6):
            points[i] = (random.gauss(-0.5, sigma), random.gauss(-0.5, sigma))
        for i in range(numpoints // 6, numpoints):
            points[i] = (random.gauss(0.5, sigma), random.gauss(0.5, sigma))
    elif mode == "uniform_square":
        for i in range(numpoints):
            points[i] = (random.uniform(-1, 1), random.uniform(-1, 1))
    else:
        raise ValueError("mode " + str(mode) + " not known")

    pointsdict = {}
    random.shuffle(points)
    for i in range(numpoints):
        pointsdict[pointids[i]] = points[i]

    return pointsdict
