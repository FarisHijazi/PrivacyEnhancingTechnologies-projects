import pandas as pd
from typing import List
from k_anon import get_attr_ranges, parse_record_min_max
import numpy as np

from math import log2, fabs


def c_dm(df: pd.DataFrame, k, ecs: List[pd.DataFrame] = None):
    return sum(np.array(list(map(len, ecs))) ** 2)


def c_avg(df: pd.DataFrame, k, ecs: List[pd.DataFrame] = None):
    return (len(df) / len(ecs)) / k


def iloss(df: pd.DataFrame, k, ecs: List[pd.DataFrame] = None):
    T, n = df.shape
    attr_ranges = get_attr_ranges(df)  # compute the attribute ranges

    sum_ = 0
    # iterating row-wise
    for j in range(df.shape[1]):  # for each attribute j
        attr_range = attr_ranges[j]
        for i in range(df.shape[0]):  # for each row i
            low, high = parse_record_min_max(df.iloc[i, j])
            sum_ += fabs((high - low) / (attr_range))

    return 1 / (T * n) * sum_


## ===== L Diversity ======

import pandas as pd
from typing import List


def is_l_diverse_distinct(ecs: List[pd.DataFrame], l: int, sens_attr: str = None):
    for ec in ecs:
        if ec[sens_attr].nunique() < l:
            return False

    return True


def is_l_diverse_entropy(ecs: List[pd.DataFrame], l: int, sens_attr: str = None):
    for ec in ecs:
        if entropy(ec[sens_attr]) < log2(l):
            return False
    return True


def entropy(ec_col: pd.Series) -> float:
    """
    :param ec_col: should be a single Series column
    :return:
    """

    freq_map = ec_col.value_counts()

    sum_ = 0
    for x_i in ec_col:
        p_ = freq_map[x_i] / len(ec_col)
        sum_ += p_ * log2(p_)
    return -sum_
