#Author: Thodoris Petropoulos

import pandas as pd

def fill_missing_dates(group, date_col, freq = 'D', per_group_imputation = False):
    """This function can be used together with apply to fill in missing dates per group will fill missing dates per time series group. values per time series group follow the full script to see how you
    Input:
        - group <grouped pandas DataFrame> (No need to specify anything. It will work with .apply method)
        - date_col <string> (Column that represents time)
        - freq <string> Frequency of data imputation ("D" for daily, "W" for weekly, "M" for monthly)
        - per_group_imputation <BOOLEAN> (If True, then the min and max value of dates will be specified by the individual series.
                                          If False, then the min and max value of dates will be specified by the whole dataset).
    """
    if per_group_imputation == True:
        date_range = pd.date_range(group.index.min(), group.index.max(), freq=freq)
    else: 
        date_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq=freq)
    group = group.reindex(date_range)
    return group

##########
#Usage
##########s

#To use the function above: 
dataframe = dataframe.set_index(date_col).groupby('series_id').apply(fill_missing_dates).rename_axis(('series_id',date_col)).drop('series_id', 1).reset_index()

#Impute missing values of target feature with 0
dataframe['target'].fillna(0,inplace=True)

#Forward fill on the categorical features if needed (depending on dataset)
dm_imputed.update(dm_imputed.groupby('series_id')[categorical_features].ffill())

#Backward fill on the categorical features if needed (depending on dataset)
dm_imputed.update(dm_imputed.groupby('series_id')[categorical_features].bfill())
