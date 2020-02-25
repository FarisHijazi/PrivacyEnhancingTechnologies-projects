import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# make random colors
r = lambda: np.random.randint(0,255) # get random color
rcolor = lambda: ('#%02X%02X%02X'%(r(),r(),r()))


# (may take up to a minute)
def scatter_matrix(data):
    from itertools import combinations

    # 28
    combos = list(combinations(data.columns, 2))

    print('combinations:', len(combos))

    plt.figure(figsize=(24, 24))

    # we have 28, can be 4 by 7
    n_rows, n_cols = 7, 4

    # plotting all combinations of features together
    for i, feats in enumerate(combos):
        df = data.loc[:, feats]

        plt.subplot(n_rows, n_cols, 1 + i)
        plt.scatter(df.values[:, 0], df.values[:, 1])
        plt.title(f'"{feats[0]}" vs "{feats[1]}"')

    plt.show()


def kmeans_matrix(data, k, centroids, labels, chunk_size=3, columns=None):
    """
    :param chunk_size: how many features to show in each plot
    """
    from itertools import combinations

    if isinstance(data, pd.DataFrame):
        columns = data.columns
        data = data.values

    print('plotting kmeans_matrix...')
    fig = plt.figure(figsize=(24, 24))

    combos = list(enumerate(combinations(np.arange(data.shape[1]), chunk_size)))

    n_rows, n_cols = int(np.round(np.sqrt(len(combos)))) + 1, int(np.round(np.sqrt(len(combos))))

    axes = []
    # plotting all combinations of features together
    for combo_i, feats in combos:
        df = data[:, feats]
        projection = '3d' if df.shape[1] == 3 else None
        ax = plt.subplot(n_rows, n_cols, 1 + combo_i, projection=projection)
        axes.append(ax)

        for k_ in range(k):
            try:
                points = np.array([df[j] for j in range(len(df)) if labels[j] == k_])
                ax.scatter(*points.T, s=7, c=rcolor())
            except Exception as e:
                try:
                    print(f'ERROR: {e}\n  Too few features to plot: points.shape={points.shape}')
                except:
                    pass

                continue
        else:
            ax.scatter(*centroids[:, feats].T, marker='*', s=200, c='#050505')

            if columns is not None:
                ax.set_title(' vs '.join([f'"{columns[f]}"' for f in feats]))
            else:
                ax.set_title(' vs '.join([f'"{f}"' for f in feats]))

    plt.show()
    return fig, axes


def connect_points(x, y, p1, p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1, x2], [y1, y2], 'o-')


def kmeans_matrix_compare(data, k, centroids1, labels1, centroids2, labels2, chunk_size=2, columns=None):
    """
    draws lines where the points went
    """
    from itertools import combinations

    if isinstance(data, pd.DataFrame):
        columns = data.columns
        data = data.values

    if centroids1.shape != centroids2.shape:
        print('ERROR: k values is not the same, different centroid list lengths')
        return

    print('plotting comparison kmeans_matrix...')
    fig = plt.figure(figsize=(24, 24))

    combos = list(enumerate(combinations(np.arange(data.shape[1]), chunk_size)))

    n_rows, n_cols = int(np.round(np.sqrt(len(combos)))) + 1, int(np.round(np.sqrt(len(combos))))

    axes = []
    # plotting all combinations of features together
    for combo_i, feats in combos:
        df = data[:, feats]
        c1 = centroids1[:, feats]
        c2 = centroids2[:, feats]

        projection = '3d' if df.shape[1] == 3 else None
        ax = plt.subplot(n_rows, n_cols, 1 + combo_i, projection=projection)
        axes.append(ax)

        for k_ in range(k):
            try:
                points = np.array([df[j] for j in range(len(df)) if labels1[j] == k_])
                ax.scatter(*points.T, s=7, c=rcolor())
            except Exception as e:
                try:
                    print(f'ERROR: {e}\n  Too few features to plot: points.shape={points.shape}')
                except:
                    pass

                continue
        else:
            ax.scatter(*centroids1[:, feats].T, marker='*', s=200, c='#050505')
            ax.scatter(*centroids2[:, feats].T, marker='o', s=50, c='#050505')

            if columns is not None:
                ax.set_title(' vs '.join([f'"{columns[f]}"' for f in feats]))
            else:
                ax.set_title(' vs '.join([f'"{f}"' for f in feats]))

        #######
        # connecting the 2 centroids
        #######

        centroid_perm = get_ordered_list_of_closest_points_between2lists(c1, c2)

        for line_pair in centroid_perm:
            plt.plot(*line_pair.T, 'o-')

    plt.show()
    return fig, axes


def get_ordered_list_of_closest_points_between2lists(list1, list2):
    import itertools
    def perm_cost(perm):
        return np.linalg.norm(perm[:, 0] - perm[:, 1])

    perms = np.array([list(zip(x, list2)) for x in itertools.permutations(list1, len(list2))])

    index = np.argsort(list(map(perm_cost, perms)))[0]
    perm = perms[index]
    return perm


def add_noise(X, eps_k_t, seed=None):
    np.random.seed(seed)
    size = None
    if hasattr(X, '__len__'):
        size = len(X)

    noise = np.random.laplace(loc=0.0, scale=1.0 / eps_k_t, size=size)
    return X + noise

