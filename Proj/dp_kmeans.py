import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from copy import deepcopy
import time
from utils import add_noise, kmeans_matrix


def kmeans(X, n_clusters=3, eps=None, STOP_THRESHOLD=0.00001, MAX_LOOPS=0,
           seed=None, plot=True, verbose=False, losses=[], plot_every=None):
    """

    :param X: the array of pixel coordinates (those of interest) | or the img itself
    :param n_clusters: n_clusters, optional
    :param eps: epsilon (privacy budget), optional
    :param STOP_THRESHOLD:
    :param MAX_LOOPS:
    :param seed:
    :param plot:
    :param verbose:
    :param losses:
    :param plot_every:
    :return: returning a list of clusters as: [centroids, labels]
      small_cluster_size: the minimum amount of values per cluster to not be discarded
    """

    columns = None
    if isinstance(X, pd.DataFrame):
        columns = X.columns
        X = X.values

    def S_k(k):
        if verbose:
            print(f'S_k({k}) ', end='')

        start_time = time.time()

        global warned
        warned = False  # used to make sure that printing only happens once

        def S_i(point):
            global warned
            si_start_time = time.time()

            # ai represents the average similarity between sample i and other samples in the same cluster.
            # The smaller ai is, more sample i should be clustered.

            ai = np.mean(dist(point, X[labels == k], axis=1), axis=0)

            # bi represents the minimum value of the average distance from i to all samples from other clusters.
            # That is to say, bi = min{bi1,bi2,…,bik}.

            # going over other clusters and getting the average distance to them

            # shape: (n_clusters-1,)
            means = np.zeros(n_clusters) + np.inf
            for k_other in np.delete(np.arange(n_clusters), k):
                points_other = X[labels == k_other]
                if len(points_other):
                    means[k_other] = np.mean(dist(point, points_other, axis=1), axis=0)

            bi = np.min(means, axis=0)

            s_i = (bi - ai) / max(ai, bi)

            # if verbose and (np.abs(s_i) >= 1.0 or np.isnan(s_i) and not warned):
            if verbose and (not warned):
                warned = True
                print(
                    '\n    clusters: ', np.array([X[labels == k_].shape[0] for k_ in range(n_clusters)]),
                    '\n    means=', means,
                    '\n    s_i={:.3f}\t ai={:.3f}\tbi={:.3f}'.format(s_i, ai, bi),
                )

            return s_i


        s_i_list = [S_i(point) for point in X[labels == k]]
        s_k = np.mean(s_i_list, axis=0)

        #         print('  {:.3f}(sec), sk={}'.format(time.time()-start_time, s_k))

        # S_k() takes 27.511 seconds
        return s_k


    #########
    # SETUP #
    #########

    # init labels
    # shape: (len(X))
    labels = np.array([k for k in range(n_clusters)] * int(np.ceil(len(X) / n_clusters)), dtype=int)[:len(X)]
    np.random.seed(seed)
    np.random.shuffle(labels)  # random shuffle

    # meaning: (cluster_idx, ...cluster_center)
    # each row corresponds to the cluster centers
    centroids = np.zeros((n_clusters, X.shape[1]))  # shape: (n_clusters, *features)
    for k in range(n_clusters):  # init centroids
        centroids[k] = np.mean([X[j] for j in range(len(X)) if labels[j] == k], axis=0)

    centroids_old = np.zeros(centroids.shape)
    # X, labels, sums, nums, centroids

    ########
    # LOOP #
    ########
    error = np.sum(dist(centroids, centroids_old, axis=1), axis=0)
    t = 1  # iteration counter
    while np.abs(error) > STOP_THRESHOLD:
        # Storing the old centroid values
        centroids_old = deepcopy(centroids)

        for i in range(len(X)):  # update labels
            # if len(labels == labels[i]) != 0:
            #     print('this is the only point in the cluster, not gonna change it')
            #     continue
            labels[i] = np.argmin(dist(X[i], centroids, axis=1))

        # for each centroid: re-calc centroid
        for k in range(n_clusters):
            cluster_points = np.array([X[j] for j in range(len(X)) if labels[j] == k])  #
            num = len(cluster_points)
            sum = np.sum(cluster_points, axis=0)

            if eps is not None:
                s_k = S_k(k)

                if np.isnan(s_k):
                    print(f'\nNaN ENCOUNTERED! S_k({k})=NaN restarting...')
                    seed = incr(seed)
                    return kmeans(X, n_clusters, eps, STOP_THRESHOLD, MAX_LOOPS,
                                  seed, plot, verbose, losses, plot_every)

                eps_k_t_ = eps_k_t(eps, t, s_k)

                seed = incr(seed)
                noise = get_noise(eps_k_t_, seed=seed)

                # adding different noises will cause it to diverge

                sum += noise
                num += noise  # updating centroids

            centroids[k] = sum / num

        error = np.sum(dist(centroids, centroids_old, axis=1), axis=0)
        losses.append(error)
        t += 1

        if np.isnan(error):
            print('NaN encountered, restarting...')
            seed = incr(seed)
            return kmeans(X, n_clusters, eps, STOP_THRESHOLD, MAX_LOOPS,
                          seed, plot, verbose, losses, plot_every)

        ## plot every loop
        if plot_every is not None and plot_every % t == 0:
            kmeans_matrix(X, n_clusters, centroids, labels)

        if verbose:
            print(f'{t} error={error}')

        if 0 < MAX_LOOPS < t:
            print(f'Stop condition reached: MAX_LOOPS exceeded: {MAX_LOOPS}, stopping iterations')
            break

    # end while

    if plot:
        plt.figure(figsize=(5, 5))
        plt.plot(losses)
        plt.title('losses')
        plt.show()

        print('plotting kmeans clusters matrix...')
        kmeans_matrix(X, n_clusters, centroids, labels, chunk_size=3, columns=columns)

    return centroids, labels


def get_noise(eps_k_t_, seed=None, size=None):
    np.random.seed(seed)
    size = None

    noise = np.random.laplace(loc=0.0, scale=1.0 / eps_k_t_, size=size)
    return noise


def eps_k_t(eps, t, s_k):
    return eps / (2 ** t) * (1 + s_k) / (1 + np.min(s_k))


def dist(a, b, axis=1):
    return np.linalg.norm(np.subtract(a, b), axis=axis)


def incr(seed):
    return seed + 1 if seed is not None else seed
