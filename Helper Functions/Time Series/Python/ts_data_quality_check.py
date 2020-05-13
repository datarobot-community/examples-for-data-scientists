#Authors:  Mark Philip, Justin Swansburg

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm


###################################
# Time Series Data Quality Checks
###################################


class DataQualityCheck:
    """
    A class used to capture summary stats and data quality checks prior to uploading time series data to DataRobot
    Attributes:
    -----------
    df : DataFrame
        time series data, including a date column and target variable at a minimum
    settings : dict
        definitions of date_col, target_col, series_id and time series parameters
    stats : dict
        summary statistics generated from `calc_summary_stats`
    duplicate_dates : int
        duplicate dates in the time series date_col
    series_timesteps : series
        steps between time units for each series_id
    series_max_gap : series
        maximum time gap per series
    series_lenth : series
        length of each series_id
    series_pct : series
        percent of series with complete time steps
    irregular : boolean
        True if df contains irregular time series data
    series_negative_target_pct : float
        Percent of target values that are negative
    Methods:
    --------
    calc_summary_stats(settings, df)
        generates a dictionary of summary statistics
    calc_time_steps(settings, df)
        calculate time steps per series_id
    hierarchical_check(settings, df)
        check if time series data passes heirarchical check
    zero_inflated_check(settings, df)
        check if target value contains zeros
    negative_values_check(settings, df)
        check if target value contains negative values
    time_steps_gap_check(settings, df)
        check if any series has missing time steps
    irregular_check(settings, df)
        check is time series data irregular
    """

    def __init__(self, df, ts_settings):
        self.df = df
        self.settings = ts_settings
        self.stats = None
        self.duplicate_dates = None
        self.series_time_steps = None
        self.series_length = None
        self.series_pct = None
        self.irregular = None
        self.series_negative_target_pct = None
        self.project_time_unit = None
        self.project_time_step = None
        self.calc_summary_stats()
        self.calc_time_steps()
        self.run_all_checks()

    def calc_summary_stats(self):
        """
        Analyze time series data to perform checks and gather summary statistics prior to modeling.
        """

        date_col = self.settings['date_col']
        series_id = self.settings['series_id']
        target = self.settings['target']
        df = self.df

        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(by=[date_col, series_id], ascending=True, inplace=True)

        # Create dictionary of helpful statistics
        stats = dict()

        stats['rows'] = df.shape[0]
        stats['columns'] = df.shape[1]
        stats['min_' + str(target)] = df[target].min()
        stats['max_' + str(target)] = df[target].max()
        stats['series'] = len(df[series_id].unique())
        stats['start_date'] = df[date_col].min()
        stats['end_date'] = df[date_col].max()
        stats['timespan'] = stats['end_date'] - stats['start_date']
        stats['median_timestep'] = df.groupby([series_id])[date_col].diff().median()
        stats['min_timestep'] = df.groupby([series_id])[date_col].diff().min()
        stats['max_timestep'] = df.groupby([series_id])[date_col].diff().max()

        # create data for histogram of series lengths
        stats['series_length'] = (
            df.groupby([series_id])[date_col].apply(lambda x: x.max() - x.min())
            / stats['median_timestep']
        )

        # calculate max gap per series
        stats['series_max_gap'] = (
            df.groupby([series_id])[date_col].apply(lambda x: x.diff().max())
            / stats['median_timestep']
        )

        self.stats = stats

    def calc_percent_missing(self, missing_value=np.nan):
        """
        Calculate percentage of rows where target is np.nan
        """
        target = self.settings['target']
        df = self.df

        if np.isnan(missing_value):
            percent_missing = sum(np.isnan(df[target])) / len(df)
        else:
            percent_missing = sum(df[target] == missing_value) / len(df)

        self.stats['percent_missing'] = percent_missing
        print('{:0.2f}% of the rows are missing a target value'.format(percent_missing * 100))

    def get_zero_inflated_series(self, cutoff=0.99):
        """
        Identify series where the target is 0.0 in more than x% of the rows
        Returns:
        --------
        List of series
        """
        assert 0 < cutoff <= 1.0, 'cutoff must be between 0 and 1'

        series_id = self.settings['series_id']
        target = self.settings['target']
        df = self.df

        df = df.groupby([series_id])[target].apply(lambda x: (x.dropna() == 0).mean())
        series = df[df >= cutoff].index.values

        pct = len(series) / self.stats['series']

        print(
            '{:0.2f}% series have zeros in more than {:0.2f}% or more of the rows'.format(
                pct * 100, cutoff * 100
            )
        )

    def calc_time_steps(self):
        """
        Calculate timesteps per series
        """
        date_col = self.settings['date_col']
        series_id = self.settings['series_id']
        df = self.df

        if self.stats is None:
            print('calc_summary_stats must be run first!')

        # create data for histogram of timestep
        series_timesteps = df.groupby([series_id])[date_col].diff() / self.stats['median_timestep']
        self.series_time_steps = series_timesteps

    def hierarchical_check(self):
        """
        Calculate percentage of series that appear on each timestep
        """
        date_col = self.settings['date_col']
        series_id = self.settings['series_id']
        df = self.df

        if self.stats is None:
            print('calc_summary_stats must be run first!')

        # Test if series passes the hierarchical check
        series_pct = df.groupby([date_col])[series_id].apply(
            lambda x: x.count() / self.stats['series']
        )
        if np.where(series_pct > 0.95, 1, 0).mean() > 0.95:
            self.stats['passes_hierarchical_check'] = True
            print(
                'Data passes hierarchical check! DataRobot hierarchical blueprints will run if you enable cross series features.'
            )
        else:
            print('Data fails hierarchical check! No hierarchical blueprints will run.')
            self.stats['passes_hierarchical_check'] = False

        self.series_pct = series_pct

    def zero_inflated_check(self):
        """
        Check if minimum target value is 0.0
        """
        target = self.settings['target']
        df = self.df

        if min(df[target]) == 0:
            self.stats['passes_zero_inflated_check'] = False
            print('The minimum target value is zero. Zero-Inflated blueprints will run.')
        else:
            self.stats['passes_zero_inflated_check'] = True
            print('Minimum target value is <> 0. Zero-inflated blueprints will not run.')

    def negative_values_check(self):
        """
        Check if any series contain negative values. If yes, identify and call out which series by id.
        """
        series_id = self.settings['series_id']
        target = self.settings['target']
        df = self.df

        df['target_sign'] = np.sign(df[target])

        try:
            # Get percent of series that have at least one negative value
            any_series_negative = (
                df.groupby([series_id])['target_sign'].value_counts().unstack()[-1]
            )
            series_negative_target_pct = np.sign(any_series_negative).sum() / len(
                df[series_id].unique()
            )
            df.drop('target_sign', axis=1, inplace=True)
            self.stats['passes_negative_values_check'] = False

            print(
                '{0:.2f}% of series have at least one negative {1} value.'.format(
                    (round(series_negative_target_pct * 100), 2), target
                )
            )

            # Identify which series have negative values
            # print('{} contain negative values. Consider creating a seperate project for these series.'.format(any_series_negative[any_series_negative == 1].index.values))
        except:
            series_negative_target_pct = 0
            self.stats['passes_negative_values_check'] = True
            print('No negative values are contained in {}.'.format(target))

        self.series_negative_target_pct = series_negative_target_pct

    def new_series_check(self):
        """
        Check if any series start after the the minimum datetime
        """
        min_dates = self.df.groupby(self.settings['series_id'])[self.settings['date_col']].min()
        new_series = min_dates > self.stats['start_date'] + dt.timedelta(days=30)

        if new_series.sum() == 0:
            self.stats['series_introduced_over_time'] = False
            print('No new series were introduced after the start of the training data')
        else:
            self.stats['series_introduced_over_time'] = True
            print(
                'Warning: You may encounter new series at prediction time. \n {0:.2f}% of the series appeared after the start of the training data'.format(
                    round(new_series.mean() * 100, 0)
                )
            )

    def old_series_check(self):
        """
        Check if any series end before the maximum datetime
        """
        max_dates = self.df.groupby(self.settings['series_id'])[self.settings['date_col']].max()
        old_series = max_dates < self.stats['end_date'] - dt.timedelta(days=30)

        if old_series.sum() == 0:
            self.stats['series_removed_over_time'] = False
            print('No series were removed before the end of the training data')
        else:
            self.stats['series_removed_over_time'] = True
            print(
                'Warning: You may encounter fewer series at prediction time. \n {0:.2f}% of the series were removed before the end of the training data'.format(
                    round(old_series.mean() * 100, 0)
                )
            )

    def leading_or_trailing_zeros_check(self, threshold=5, drop=True):
        """
        Check for contain consecutive zeros at the beginning or end of each series
        """

        date_col = self.settings['date_col']
        series_id = self.settings['series_id']
        target = self.settings['target']
        df = self.df

        new_df = remove_leading_and_trailing_zeros(
            df,
            series_id,
            date_col,
            target,
            leading_threshold=threshold,
            trailing_threshold=threshold,
            drop=drop,
        )

        if new_df.shape[0] < df.shape[0]:
            print(f'Warning: Leading and trailing zeros detected within series')
        else:
            print(f'No leading or trailing zeros detected within series')

    def duplicate_dates_check(self):
        """
        Check for duplicate datetimes within each series
        """

        duplicate_dates = self.df.groupby([self.settings['series_id'], self.settings['date_col']])[
            self.settings['date_col']
        ].count()
        duplicate_dates = duplicate_dates[duplicate_dates > 1]
        if len(duplicate_dates) == 0:
            print(f'No duplicate timestamps detected within any series')
            self.stats['passes_duplicate_timestamp_check'] = True
        else:
            print('Warning: Data contains duplicate timestamps within series!')
            self.stats['passes_duplicate_timestamp_check'] = False

    def time_steps_gap_check(self):
        """
        Check for missing timesteps within each series
        """
        date_col = self.settings['date_col']
        series_id = self.settings['series_id']
        df = self.df
        gap_size = self.stats['median_timestep']

        if self.stats is None:
            print('calc_summary_stats must be run first!')

        # check is series has any missing time steps
        self.stats['pct_series_w_gaps'] = (
            df.groupby([series_id])[date_col].apply(lambda x: x.diff().max()) > gap_size
        ).mean()

        print(
            '{0:.2f}% of series have at least one missing time step.'.format(
                round(self.stats['pct_series_w_gaps'] * 100), 2
            )
        )

    def _get_spacing(self, df, project_time_unit):
        """
        Helper function for self.irregular_check()
        Returns:
        --------
        List of series
        """
        project_time_unit = self.project_time_unit
        ts_settings = self.settings
        date_col = ts_settings['date_col']
        series_id = ts_settings['series_id']

        df['indicator'] = 1
        df = fill_missing_dates(df=df, ts_settings=ts_settings)

        if project_time_unit == 'minute':
            df['minute'] = df[date_col].dt.minute
        elif project_time_unit == 'hour':
            df['hour'] = df[date_col].dt.hour
        elif project_time_unit == 'day':
            df['day'] = df[date_col].dt.dayofweek
        elif project_time_unit == 'week':
            df['week'] = df[date_col].dt.week
        elif project_time_unit == 'month':
            df['month'] = df[date_col].dt.month

        sums = df.groupby([series_id, project_time_unit])['indicator'].sum()
        counts = df.groupby([series_id, project_time_unit])['indicator'].agg(
            lambda x: x.fillna(0).count()
        )

        pcts = sums / counts

        irregular = pcts.reset_index(drop=True) < 0.8
        irregular = irregular[irregular]

        return irregular

    def irregular_check(self, plot=False):
        """
        Check for irregular spacing within each series
        """

        date_col = self.settings['date_col']
        df = self.df.copy()

        # first cast date column to a pandas datetime type
        df[date_col] = pd.to_datetime(df[date_col])

        project_time_unit, project_time_step = get_timestep(self.df, self.settings)

        self.project_time_unit = project_time_unit
        self.project_time_step = project_time_step

        print('Project Timestep: ', project_time_step, ' ', project_time_unit)

        if project_time_unit == 'minute':
            df['minute'] = df[date_col].dt.minute
        elif project_time_unit == 'hour':
            df['hour'] = df[date_col].dt.hour
        elif project_time_unit == 'day':
            df['day'] = df[date_col].dt.dayofweek
        elif project_time_unit == 'week':
            df['week'] = df[date_col].dt.week
        elif project_time_unit == 'month':
            df['month'] = df[date_col].dt.month

        # Plot histogram of timesteps
        time_unit_counts = df[project_time_unit].value_counts()

        if plot:
            time_unit_percent = time_unit_counts / sum(time_unit_counts.values)

            fig = px.bar(
                time_unit_percent,
                x=time_unit_percent.index,
                y=time_unit_percent.values,
                title=f'Percentage of records per {project_time_unit}',
            )
            fig.update_xaxes(title=project_time_unit)
            fig.update_yaxes(title='Percentage')
            fig.show()

        # Detect uncommon time steps
        # If time bin has less than 30% of most common bin then it is an uncommon time bin
        uncommon_time_bins = list(
            time_unit_counts[(time_unit_counts / time_unit_counts.max()) < 0.3].index
        )
        common_time_bins = list(
            time_unit_counts[(time_unit_counts / time_unit_counts.max()) >= 0.3].index
        )

        if len(uncommon_time_bins) > 0:
            print(f'Uncommon {project_time_unit}s:', uncommon_time_bins)
        else:
            print('There are no uncommon time steps')

        # Detect irregular series
        df = df.loc[df[project_time_unit].isin(common_time_bins), :]
        irregular_series = self._get_spacing(df, project_time_unit)

        if len(irregular_series) > 0:
            print(
                'Series are irregularly spaced. Projects will only be able to run in row-based mode!'
            )
            self.stats['passes_irregular_check'] = False
        else:
            self.stats['passes_irregular_check'] = True
            print(
                'Timesteps are regularly spaced. You will be able to run projects in either time-based or row-based mode'
            )

    def detect_periodicity(self, alpha=0.05):
        """
        Calculate project-level periodicity
        """

        timestep = self.project_time_unit
        df = self.df
        target = self.settings['target']
        date_col = self.settings['date_col']
        metric = self.settings['metric']

        metrics = {
            'LogLoss': sm.families.Binomial(),
            'RMSE': sm.families.Gaussian(),
            'Poisson Deviance': sm.families.Poisson(),
            'Gamma Deviance': sm.families.Gamma(),
        }

        periodicity = {
            'moh': 'hourly',
            'hod': 'daily',
            'dow': 'weekly',
            'dom': 'monthly',
            'month': 'yearly',
        }

        try:
            loss = metrics[metric]
        except KeyError:
            loss = metrics['RMSE']

        # Instantiate a glm with the default link function.
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.loc[np.isfinite(df[target]), :].copy()

        df['moh'] = df[date_col].dt.minute
        df['hod'] = df[date_col].dt.hour
        df['dow'] = df[date_col].dt.dayofweek
        df['dom'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month

        if timestep == 'minute':
            inputs = ['moh', 'hod', 'dow', 'dom', 'month']
        elif timestep == 'hour':
            inputs = ['hod', 'dow', 'dom', 'month']
        elif timestep == 'day':
            inputs = ['dow', 'dom', 'month']
        elif timestep == 'week':
            inputs = ['month']
        else:
            raise ValueError('timestep has to be either minute, hour, day, week, or month')

        output = []
        for i in inputs:
            x = pd.DataFrame(df[i])
            y = df[target]

            x = pd.get_dummies(x.astype('str'), drop_first=True)
            x['const'] = 1

            clf = sm.GLM(endog=y, exog=x, family=loss)
            model = clf.fit()

            if any(model.pvalues[:-1] <= alpha):
                output.append(periodicity[i])
                # print(f'Detected periodicity: {periodicity[i]}')
                # return periodicity[i]

        if len(output) > 0:
            print(f'Detected periodicity: {output}')
        else:
            print('No periodicity detected')

    def run_all_checks(self):
        """
        Runner function to run all data checks in one call
        """
        print('Running all data quality checks...\n')

        series = self.stats['series']
        start_date = self.stats['start_date']
        end_date = self.stats['end_date']
        rows = self.stats['rows']
        cols = self.stats['columns']

        print(f'There are {rows} rows and {cols} columns')
        print(f'There are {series} series')
        print(f'The data spans from  {start_date} to {end_date}')

        self.hierarchical_check()
        self.zero_inflated_check()
        self.new_series_check()
        self.old_series_check()
        self.duplicate_dates_check()
        self.leading_or_trailing_zeros_check()
        self.time_steps_gap_check()
        self.calc_percent_missing()
        self.get_zero_inflated_series()
        self.irregular_check()
        self.detect_periodicity()


def get_timestep(df, ts_settings):
    """
    Calculate the project-level timestep
    Returns:
    --------
    project_time_unit: minute, hour, day, week, or month
    project_time_step: int
    Examples:
    --------
    '1 days'
    '4 days'
    '1 week'
    '2 months'
    """
    date_col = ts_settings['date_col']
    series_id = ts_settings['series_id']
    df = df.copy()

    # Cast date column to a pandas datetime type and sort df
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(by=[date_col, series_id], ascending=True, inplace=True)

    # Calculate median timestep
    deltas = df.groupby([series_id])[date_col].diff().reset_index(drop=True)
    median_timestep = deltas.apply(lambda x: x.total_seconds()).median()

    # Logic to detect project time step and time unit
    if (60 <= median_timestep < 3600) & (median_timestep % 60 == 0):
        project_time_unit = 'minute'
        project_time_step = int(median_timestep / 60)
        df['minute'] = df[date_col].dt.minute
    elif (3600 <= median_timestep < 86400) & (median_timestep % 3600 == 0):
        project_time_unit = 'hour'
        project_time_step = int(median_timestep / 3600)
        df['hour'] = df[date_col].dt.hour
    elif (86400 <= median_timestep < 604800) & (median_timestep % 86400 == 0):
        project_time_unit = 'day'
        project_time_step = int(median_timestep / 86400)
        df['day'] = df[date_col].dt.strftime('%A')
    elif (604800 <= median_timestep < 2.628e6) & (median_timestep % 604800 == 0):
        project_time_unit = 'week'
        project_time_step = int(median_timestep / 604800)
        df['week'] = df[date_col].dt.week
    elif (median_timestep >= 2.628e6) & (median_timestep % 2.628e6 == 0):
        project_time_unit = 'month'
        project_time_step = int(median_timestep / 2.628e6)
        df['month'] = df[date_col].dt.month
    else:
        raise ValueError(f'{median_timestep} seconds is not a supported timestep')

    # print('Project Timestep: 1', project_time_unit)

    return project_time_unit, project_time_step


def _reindex_dates(group, freq):
    """
    Helper function for fill_missing_dates()
    """
    date_range = pd.date_range(group.index.min(), group.index.max(), freq=freq)
    group = group.reindex(date_range)
    return group


def fill_missing_dates(df, ts_settings, freq=None):
    """
    Insert rows with np.nan targets for series with missing timesteps between the series start and end dates
    df: pandas df
    ts_settings: dictionary of parameters for time series project
    freq: project time unit and timestep
    Returns:
    --------
    pandas df with inserted rows
    """
    date_col = ts_settings['date_col']
    series_id = ts_settings['series_id']
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(by=[series_id, date_col], ascending=True, inplace=True)

    if freq is None:
        mapper = {'minute': 'min', 'hour': 'H', 'day': 'D', 'week': 'W', 'month': 'M'}
        project_time_unit, project_time_step = get_timestep(df, ts_settings)
        freq = str(project_time_step) + mapper[project_time_unit]

    df = (
        df.set_index(date_col)
        .groupby(series_id)
        .apply(_reindex_dates, freq)
        .rename_axis((series_id, date_col))
        .drop(series_id, axis=1)
        .reset_index()
    )

    return df.reset_index(drop=True)


def _remove_leading_zeros(df, date_col, target, threshold=5, drop=False):
    df[date_col] = pd.to_datetime(df[date_col])
    df_non_zero = df[(df[target] != 0) & (~pd.isnull(df[target]))]
    min_date = df_non_zero[date_col].min()
    df_begin = df[df[date_col] < min_date]
    if df_begin[target].dropna().shape[0] >= threshold or pd.isnull(min_date):
        if drop:
            if pd.isnull(min_date):
                return pd.DataFrame(columns=df.columns, dtype=float)
            return df[df[date_col] >= min_date]
        else:
            df[target] = df.apply(
                lambda row: np.nan
                if pd.isnull(min_date) or row[date_col] < min_date
                else row[target],
                axis=1,
            )
            return df
    else:
        return df


def _remove_trailing_zeros(df, date_col, target, threshold=5, drop=False):
    df[date_col] = pd.to_datetime(df[date_col])
    df_non_zero = df[(df[target] != 0) & (~pd.isnull(df[target]))]
    max_date = df_non_zero[date_col].max()
    df_end = df[df[date_col] > max_date]
    if df_end[target].dropna().shape[0] >= threshold or pd.isnull(max_date):
        if drop:
            if pd.isnull(max_date):
                return pd.DataFrame(columns=df.columns, dtype=float)
            return df[df[date_col] <= max_date]
        else:
            df[target] = df.apply(
                lambda row: np.nan
                if pd.isnull(max_date) or row[date_col] > max_date
                else row[target],
                axis=1,
            )
            return df
    else:
        return df


def remove_leading_and_trailing_zeros(
    df, series_id, date_col, target, leading_threshold=5, trailing_threshold=5, drop=False
):
    """
    Remove excess zeros at the beginning or end of series
    df: pandas df
    leading_threshold: minimum number of consecutive zeros at the beginning of a series before rows are dropped
    trailing_threshold: minimum number of consecutive zeros at the end of series before rows are dropped
    drop: specifies whether to drop the zeros or set them to np.nan
    Returns:
    --------
    pandas df
    """

    df = (
        df.groupby(series_id)
        .apply(_remove_leading_zeros, date_col, target, leading_threshold, drop)
        .reset_index(drop=True)
    )
    df = (
        df.groupby(series_id)
        .apply(_remove_trailing_zeros, date_col, target, trailing_threshold, drop)
        .reset_index(drop=True)
    )

    return df.reset_index(drop=True)


#####################
# Data Visualization
#####################


def _cut_series_by_rank(df, ts_settings, n=1, top=True):
    df_agg = df.groupby(ts_settings['series_id']).mean()
    selected_series_names = (
        df_agg.sort_values(by=ts_settings['target'], ascending=top).tail(n).index.values
    )

    return selected_series_names


def _cut_series_by_quantile(df, ts_settings, quantile=0.95, top=True):
    series_id = ts_settings['series_id']
    target = ts_settings['target']

    df_agg = df.groupby(series_id).mean()

    if top:
        selected_series_names = df_agg[
            df_agg[target] >= df_agg[target].quantile(quantile)
        ].index.values
    else:
        selected_series_names = df_agg[
            df_agg[target] <= df_agg[target].quantile(quantile)
        ].index.values

    return selected_series_names


def plot_series_average(df, settings):
    date_col = settings['date_col']
    target = settings['target']

    # Average of all series over time
    df_agg = df.groupby(date_col).mean()
    df_agg['Date'] = pd.to_datetime(df_agg.index.values)

    fig = px.line(df_agg, x='Date', y=target)
    fig.update_layout(title_text='Average of all Series')
    fig.show()


def plot_individual_series(df, ts_settings, n=None, top=True):
    """
    Plot individual series on the same chart
    n: (int) number of series to plot
    top: (boolean) whether to select the top n largest or smallest series ranked by average target value
    """
    date_col = ts_settings['date_col']
    series_id = ts_settings['series_id']
    target = ts_settings['target']

    if n is None:
        n = len(df[series_id].unique())

    series = _cut_series_by_rank(df, ts_settings, n=n, top=top)
    df_subset = df[df[series_id].isin(series)]

    fig = px.line(df_subset, x=date_col, y=target, color=df_subset[series_id])
    fig.update_layout(title_text='Top Series By Target Over Time')
    fig.show()
