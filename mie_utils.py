# File for my utility functions
from bisect import bisect_left
import pandas as pd
import os


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

def get_from_csv(csv_file, *args):
    """
    Get parameters from a csv file. The syntax of the csv allows comments using
    '#' as prefix and tab as separator.

    Example .csv:
    parameter_name	value
    init_scan	1570674.0
    xmin	0.0
    xmax	0.475

    Example use:
    >>> a0, b1, c2 = get_from_csv("conf/file.csv","a0","b1","c2")

    csv_file: file name of the csv.

    *args: list of strings with the parameters' names.

    return: tuple containing the retrieved parameters, in the same order as the
            function argument (see example above for correct use).
    """
    if os.path.isfile(csv_file):
        # read up the parameters (like init_scan)
        parameters_table = pd.read_csv(csv_file,
                           delimiter=":", index_col=0, comment="#",
                           skipinitialspace=True)
    else: #
        raise FileNotFoundError("[ERROR]: parameter file'"+ csv_file +"''  not found. "
                                "Create it with save_as_hdf.py")
    values = list()
    for p in args:
        values.append(parameters_table.loc[p].values[0])
    if len(values) == 1:
        # Extract the value from the values array if it's only one variable
        # So for example in assigments like
        # >>> x = get_from_csv("f.csv", "x")
        # x will be a scalar like 10, and not [10]
        return values[0]
    else:
        # Python already unzips if one assigns to multiple vars:
        # >>> x,y = get_from_csv("f.csv", "x", "y")
        # x,y already scalars in this case
        return values

def save_in_csv(csv_file, **kwargs):
    """
    Save parameters in a csv file, if the file does not exist create it,
    otherwise append to it. If the parameters to write already exist in the file
    it will overwrite them

    Example .csv:
    parameter_name	value
    init_scan	1570674.0
    xmin	0.0
    xmax	0.475

    Example use:
    >>> save_in_csv("conf/file.csv",a0=1,b1=-1,c2=1e10)

    csv_file: file name of the csv.

    *kwargs: keyword (var_name=value) arguments, 'var_name' will be the name of
             the parameter in file and 'value' will be its value (see example
             above).

    return: void
    """
    if os.path.isfile(csv_file):
        # read up the already existing parameters (like init_scan)
        parameters_table = pd.read_csv(csv_file,
                           sep=":", index_col=0, comment="#",
                           skipinitialspace=True)
    else:
        parameters_table = pd.DataFrame({"parameter_name": [],
                                        "value": []})
        parameters_table.set_index("parameter_name",inplace=True)

    for var_name,value in kwargs.items():
        if var_name in parameters_table.index:
            print("[LOG]: Overwriting parameter already in csv: '{}' = {} -> {}"
                        .format(var_name,parameters_table.loc[var_name].value,value))

        parameters_table.loc[var_name] = value

    parameters_table.to_csv(csv_file,sep=':')
