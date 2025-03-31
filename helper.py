import math
from itertools import combinations

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def closest_pair(points):
    min_dist = float('inf')
    closest = None
    for a, b in combinations(points, 2):
        d = distance(a, b)
        if d < min_dist:
            min_dist = d
            closest = (a, b)
    return closest

"""Given two points p1, p2 intersection returns 2 points"""
def get_inters(p1, p2, r0, r1):
    x0, y0 = p1
    x1, y1 = p2

    d = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    if d > r0 + r1:
        return None

    if d < abs(r0 - r1):
        return None

    if d == 0 and r0 == r1:
        return None

    else:
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        h = math.sqrt(r0**2 - a**2)

        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d

        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        return [(x3, y3), (x4, y4)]


def path_loss_dist(rssi, tx):
    return 10 ** ((tx - rssi)/10 * 3) # assuming n = 3, because its a mix between open - office spaces 