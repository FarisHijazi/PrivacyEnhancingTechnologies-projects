from typing import Tuple, Union, List

import pandas as pd
from collections import Counter
import numpy as np
from utils import safe_parse
import re
import time


# this assigns unique IDs to equivalency classes

# @author: Faris Hijazi
# COE449 K anonymization implementation
#
# NOTE: in the implementation, the split I have taken is not following the mondrian convention,
#       it goes as follows:     lhs=(<splitval), rhs=(>=splitval)
#

ec_id_counter = 0


def choose_dim(QIs, df, dim_counter_seed=None):
    QIs = sorted(QIs, key=lambda qi: df[qi].nunique())
    dim_counter_seed = (dim_counter_seed + 1)
    if dim_counter_seed < len(QIs):
        return QIs[dim_counter_seed]


def generalize_attr(df: pd.DataFrame, attr) -> pd.DataFrame:
    """
    merge values to a bunch of ranges
    :param df:
    :param attr: name of attribute
    :return: dataframe
    """
    # given a partition and an attribute, will set them all to the same value
    attr_domain = df.loc[:, attr].apply(safe_parse)

    is_all_the_same = attr_domain.nunique() == 1
    if len(attr_domain) <= 1:
        return attr_domain

    attr_domain = attr_domain.values  # just sorting

    low, high = get_attr_min_max(attr_domain)

    if not is_all_the_same:
        df.loc[:, attr] = str('["{}" - "{}"]'.format(low, high))
    else:
        df.loc[:, attr] = df.loc[:, attr].apply(str)

    # adding the EC ID
    global ec_id_counter
    ec_id_counter += 1
    df['ec'] = ec_id_counter

    # print("setting values of {} to {}".format(attr, anonymized_value))
    return df


def get_mondrian_split(df, attr, k) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    :returns the split of the dataframe as a 2 dataframe tuple if possible (else: None)
    :param attr: a which attr to split on

    Example: to split [0,1,2,3] in half, get_split() => [0, 1] [2, 3]
    """
    attrs = list(df[attr])
    n = len(attrs)

    # TODO: this doesn't try hard enough to check for both cases of n/2 and (n/2 + 1)
    split_index = (n // 2) if (n % 2 == 0) \
        else ((n + 1) // 2)  # where to cut the sorted index list

    # using frequency set
    fs = Counter(attrs)  # frequency set
    meta_index = split_index  # just an initial value
    counter_ = 0
    for (val, freq) in fs.items():
        counter_ += freq
        if counter_ >= split_index:
            meta_index = counter_
            break

    argsort = np.argsort(attrs)  # indexes of sorted list

    lhs_index, rhs_index = argsort[0: meta_index + 1], argsort[meta_index + 1:]  # the index within
    lhs = df.iloc[lhs_index, :]
    rhs = df.iloc[rhs_index, :]

    if len(lhs) < k:
        lhs = None
    if len(rhs) < k:
        rhs = None

    return (lhs, rhs)


####

def get_ecs(df: pd.DataFrame):
    """
    @returns the list of the equivalency classes
    """
    return [group for name, group in df.groupby('ec')]


def parse_record_min_max(val):
    """
    parses a generalized range in a record and
    :param val:
    :return: returns the min and max or (in the case that it isn't a range): returns the same
        value as min and max
    ```
    l, h = parse_record_min_max(val)
    ```
    """
    low_ = high_ = val
    m = re.match('^\["(.+?)" - "(.+?)"\]$', str(val), re.DOTALL)
    if m and len(m.groups()) == 2:
        low_, high_ = m.group(1), m.group(2)
    return safe_parse(low_), safe_parse(high_)


def get_attr_min_max(attr_domain):
    """
    :param attr_domain:
    :return: low,high of a column
    """

    def convert_to_parsed_max(a):
        l, h = parse_record_min_max(a)
        return h if (l != h) else safe_parse(a)

    def convert_to_parsed_min(a):
        l, h = parse_record_min_max(a)
        return l if (l != h) else safe_parse(a)

    # parsing the old value if it's already anonymized
    # it's unfortunate that I had to add this part, if it's excluded, what happens is that the anonymized attributes will get anonymized again
    # example: there would be a single entry containing: ["["21" - "38"]" - "39"] instead of ["21" - "39"]
    low = min(map(convert_to_parsed_min, attr_domain))
    high = max(map(convert_to_parsed_max, attr_domain))
    return low, high


def combine_partitions(partitions):
    res_df = pd.DataFrame(columns=partitions[0].columns)  # create empty dataframe
    global ec_id_counter
    for partition in partitions:
        ec_id_counter += 1
        partition['ec'] = ec_id_counter
        res_df = pd.concat([res_df, partition], sort=False)
    return res_df


def get_attr_ranges(df):
    return [np.subtract(*reversed(get_attr_min_max(df[col]))) for col in df]


###

def mondrian_attr_once(df, dim, k) -> Tuple[bool, pd.DataFrame]:
    """
    anonymize for a single attribute/dimension
    :param df:
    :param dim:
    :return: tuple (isDone, dataframe) anonymized partitions, if isDone is true, there are no more valid cuts
    """

    done = True
    lhs, rhs = get_mondrian_split(df, dim, k)
    if lhs is not None and rhs is not None:
        done_l, lhs = mondrian_attr_once(lhs, dim, k)
        done_r, rhs = mondrian_attr_once(rhs, dim, k)

        done = done_l and done_r
        combined = combine_partitions([generalize_attr(lhs, dim), generalize_attr(rhs, dim)])
    return (done, combined)


def mondrian_attr(df, dim, k) -> Tuple[bool, pd.DataFrame]:
    """
    anonymize for a single attribute/dimension
    :param df:
    :param dim:
    :return: tuple (isDone, dataframe) anonymized partitions, if isDone is true, there are no more valid cuts
    """

    # for dim in QIs:
    if len(df) >= k * 2:  # if can't split ( too small )

        lhs, rhs = get_mondrian_split(df, dim, k)
        if lhs is not None and rhs is not None:
            done_l, lhs = mondrian_attr_once(lhs, dim, k)
            done_r, rhs = mondrian_attr_once(rhs, dim, k)

            done = done_l and done_r
            combined = combine_partitions([generalize_attr(lhs, dim), generalize_attr(rhs, dim)])
            return (done, combined)
        # if no valid split, try another dimension (if possible) otherwise just return
        # if no new dimensions, just return

    return (True, df)


def mondrian(df: pd.DataFrame, k=3, QIs=None, as_partitions=False, i=0) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    :param df:  a pandas dataframe object
    :param k:   k value for the algorithm
    :param QIs: (Quasi identifiers) pandas DataFrame index (either column names or numbered indexes)
    :param as_partitions: if True, will return the list of equivalency sets instead of the combined table
            (you can then merge them with `combine_partitions()`)

    :returns a k-anonymized copy of the dataframe
    """

    """
    for each partition,
    choose a dimension
    if can split, split
        generalize once
        then put it back in the stack
    """

    df['ec'] = 0  # just appending a column of EC (equivalency class ID), instead of adding it every bit

    print("mondrian k={}-anonymization, dimensions: {}".format(k, df.shape))

    ready_partitions = []
    stack = [df]
    dim_counter_seed = 0
    start_time = time.time()

    while True:
        i += 1

        if len(stack) == 0:
            break

        partition = stack.pop()

        dim = choose_dim(QIs, partition, dim_counter_seed=dim_counter_seed)

        # if too small
        if len(partition) < k * 2 or dim is None:
            # print('{}: partition exhausted'.format(i))
            ready_partitions.append(partition)
            dim_counter_seed = 0
            continue

        lhs, rhs = get_mondrian_split(partition, dim, k)
        if lhs is None or rhs is None:  # can't split on this attr
            stack.append(partition)  # put it back
            dim_counter_seed += 1
            continue

        lhs = generalize_attr(lhs, dim)
        rhs = generalize_attr(rhs, dim)

        stack += [lhs, rhs]

    print('{:.3}seconds'.format(time.time() - start_time))
    if as_partitions:
        return ready_partitions
    else:
        return combine_partitions(ready_partitions)


if __name__ == "__main__":
    print(
        "Faris Hijazi s201578750"
        "\nCOE 449: Privacy Enhancing Technologies - Assignment 1: K-anonynmization"
    )
    print("testing")

    dataset = pd.read_csv('./dataset/ipums.csv')
    print(dataset)

    QIs = [
        "Age",
        "Gender",
        "Marital status",
        "Race status",
        "Birth",
        "Language",
        "Occupation"
    ]

    df_anon = mondrian(dataset, k=3, QIs=QIs)

    print('=================')
    print("Anonymized table:")
    print('=================')
    print('size: {}, EC count: {}'.format(len(df_anon), df_anon['ec'].nunique()))
    print("ranges", get_attr_ranges(df_anon))
