import numpy as np

def get_dist(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def euc_min_dist(cur_point, points):
    dists = []
    for point in points:
        dists.append(get_dist(cur_point, point))
    return np.argmin(dists)

def gen_data(n_samples : int):
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5

    x = np.random.uniform(x_min, x_max, n_samples)
    y = np.random.uniform(y_min, y_max, n_samples)

    data = np.column_stack((x, y))
    return data
