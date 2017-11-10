# File for my utility functions
from bisect import bisect_left
import pandas as pd

# Controllare output
def hist_get_bin(df, point):
    """
    Function that get in "df" the data associated with the (Multi)Index
    closest to point. For example, in a 2D histogram with bin centers
    as bin labels, it can be used to get the containing bin of "point".

    df: Dataframe or groupby object with a ordered (Multi)Index.

    point: tuple of size (df.index.nlevels,), the point whose bin we want.

    return: the content of the bin (can be a DataFrame if df is a GroupBy
    object).
    """
    bin_coordinates = list()

    # Cannot use same methods on dataframes and groupby objects
    # We use df.first to get a dataframe with the same index as the groupby object
    # Otherwise we just get the index
    index = df.first().index if isinstance(df, pd.core.groupby.GroupBy) \
                             else df.index

    # Search the closest for every multiindex component (coordinate)
    for n in range(index.nlevels):
       nth_axis_values = index.get_level_values(n) # Bin centers values
       user_nth_coordinate = point[n]

       closest_bin_nth_coordinate = take_closest(nth_axis_values,
                                                 user_nth_coordinate)

       bin_coordinates.append(closest_bin_nth_coordinate)

    # Cannot use same methods on dataframes and groupby objects
    result = df.get_group(tuple(bin_coordinates)) \
             if isinstance(df, pd.core.groupby.GroupBy) \
             else df.loc[tuple(bin_coordinates)]

    return result


# From https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-values
def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.

    myList: ordered array-like

    myNumber: number which we search the closest element to in list

    return: closest list element
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before
