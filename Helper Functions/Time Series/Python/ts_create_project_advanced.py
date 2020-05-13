#Authors: Justin Swansburg, Mark Philip

import datetime as dt
import time

import datarobot as dr
import numpy as np
import pandas as pd

from ts_data_quality import get_timestep #ts_data_quality is a helper function that exists within this Repo


###################
# Project Creation
###################


def create_dr_project(df, project_name, ts_settings, **advanced_options):
    """
    Kickoff single DataRobot project
    df: pandas df
    project_name: name of project
    ts_settings: dictionary of parameters for time series project
    Returns:
    --------
    DataRobot project object
    """

    print(f'Building Next Project \n...\n')

    #######################
    # Get Advanced Options
    #######################
    opts = {
        'weights': None,
        'response_cap': None,
        'blueprint_threshold': None,
        'seed': None,
        'smart_downsampled': False,
        'majority_downsampling_rate': None,
        'offset': None,
        'exposure': None,
        'accuracy_optimized_mb': None,
        'scaleout_modeling_mode': None,
        'events_count': None,
        'monotonic_increasing_featurelist_id': None,
        'monotonic_decreasing_featurelist_id': None,
        'only_include_monotonic_blueprints': None,
    }

    for opt in advanced_options.items():
        opts[opt[0]] = opt[1]

    opts = dr.AdvancedOptions(
        weights=opts['weights'],
        seed=opts['seed'],
        monotonic_increasing_featurelist_id=opts['monotonic_increasing_featurelist_id'],
        monotonic_decreasing_featurelist_id=opts['monotonic_decreasing_featurelist_id'],
        only_include_monotonic_blueprints=opts['only_include_monotonic_blueprints'],
        accuracy_optimized_mb=opts['accuracy_optimized_mb'],
        smart_downsampled=opts['smart_downsampled'],
    )

    ############################
    # Get Datetime Specification
    ############################
    settings = {
        'max_date': None,
        'known_in_advance': None,
        'num_backtests': None,
        'validation_duration': None,
        'holdout_duration': None,
        'holdout_start_date': None,
        'disable_holdout': False,
        'number_of_backtests': None,
        'backtests': None,
        'use_cross_series_features': None,
        'aggregation_type': None,
        'cross_series_group_by_columns': None,
        'calendar_id': None,
        'use_time_series': False,
        'series_id': None,
        'metric': None,
        'target': None,
        'mode': dr.AUTOPILOT_MODE.FULL_AUTO,  # MANUAL #QUICK
        'date_col': None,
        'fd_start': None,
        'fd_end': None,
        'fdw_start': None,
        'fdw_end': None,
    }

    for s in ts_settings.items():
        settings[s[0]] = s[1]

    df[settings['date_col']] = pd.to_datetime(df[settings['date_col']])

    if settings['max_date'] is None:
        settings['max_date'] = df[settings['date_col']].max()
    else:
        settings['max_date'] = pd.to_datetime(settings['max_date'])

    if ts_settings['known_in_advance']:
        settings['known_in_advance'] = [
            dr.FeatureSettings(feat_name, known_in_advance=True)
            for feat_name in settings['known_in_advance']
        ]

    # Update validation and holdout duration, start, and end date
    project_time_unit, project_time_step = get_timestep(df, settings)

    validation_durations = {'minute': 0, 'hour': 0, 'day': 0, 'month': 0}
    holdout_durations = {'minute': 0, 'hour': 0, 'day': 0, 'month': 0}

    if project_time_unit == 'minute':
        validation_durations['minute'] = settings['validation_duration']
        holdout_durations['minute'] = settings['holdout_duration']

    elif project_time_unit == 'hour':
        validation_durations['hour'] = settings['validation_duration']
        holdout_durations['hour'] = settings['holdout_duration']

    elif project_time_unit == 'day':
        validation_durations['day'] = settings['validation_duration']
        holdout_durations['day'] = settings['holdout_duration']

    elif project_time_unit == 'week':
        validation_durations['day'] = settings['validation_duration'] * 7
        holdout_durations['day'] = settings['holdout_duration'] * 7

    elif project_time_unit == 'month':
        validation_durations['day'] = settings['validation_duration'] * 31
        holdout_durations['day'] = settings['holdout_duration'] * 31

    else:
        raise ValueError(f'{project_time_unit} is not a supported timestep')

    if settings['disable_holdout']:
        settings['holdout_duration'] = None
        settings['holdout_start_date'] = None
    else:
        settings['holdout_start_date'] = settings['max_date'] - dt.timedelta(
            minutes=holdout_durations['minute'],
            hours=holdout_durations['hour'],
            days=holdout_durations['day'],
        )

        settings['holdout_duration'] = dr.partitioning_methods.construct_duration_string(
            minutes=holdout_durations['minute'],
            hours=holdout_durations['hour'],
            days=holdout_durations['day'],
        )

    ###############################
    # Create Datetime Specification
    ###############################
    time_partition = dr.DatetimePartitioningSpecification(
        feature_settings=settings['known_in_advance'],
        # gap_duration = dr.partitioning_methods.construct_duration_string(years=0, months=0, days=0),
        validation_duration=dr.partitioning_methods.construct_duration_string(
            minutes=validation_durations['minute'],
            hours=validation_durations['hour'],
            days=validation_durations['day'],
        ),
        datetime_partition_column=settings['date_col'],
        use_time_series=settings['use_time_series'],
        disable_holdout=settings['disable_holdout'],  # set this if disable_holdout is set to False
        holdout_start_date=settings['holdout_start_date'],
        holdout_duration=settings[
            'holdout_duration'
        ],  # set this if disable_holdout is set to False
        multiseries_id_columns=[settings['series_id']],
        forecast_window_start=int(settings['fd_start']),
        forecast_window_end=int(settings['fd_end']),
        feature_derivation_window_start=int(settings['fdw_start']),
        feature_derivation_window_end=int(settings['fdw_end']),
        number_of_backtests=settings['num_backtests'],
        calendar_id=settings['calendar_id'],
        use_cross_series_features=settings['use_cross_series_features'],
        aggregation_type=settings['aggregation_type'],
        cross_series_group_by_columns=settings['cross_series_group_by_columns'],
    )

    ################
    # Create Project
    ################
    project = dr.Project.create(
        project_name=project_name, sourcedata=df, max_wait=14400, read_timeout=14400
    )

    print(f'Project {project_name} Created...')

    #################
    # Start Autopilot
    #################
    project.set_target(
        target=settings['target'],
        metric=settings['metric'],
        mode=settings['mode'],
        advanced_options=opts,
        worker_count=-1,
        partitioning_method=time_partition,
        max_wait=14400,
    )

    return project


def create_dr_projects(
    df, ts_settings, prefix='TS', split_col=None, fdws=None, fds=None, **advanced_options
):
    """
    Kickoff multiple DataRobot projects
    df: pandas df
    ts_settings: dictionary of parameters for time series project
    prefix: str to concatenate to start of project name
    split_col: column in df that identifies cluster labels
    fdws: list of tuples containing feature derivation window start and end values
    fds: list of tuples containing forecast distance start and end values
    Returns:
    --------
    List of projects
    Example:
    --------
    split_col = 'Cluster'
    fdws=[(-14,0),(-28,0),(-62,0)]
    fds = [(1,7),(8,14)]
    """

    if fdws is None:
        fdws = [(ts_settings['fdw_start'], ts_settings['fdw_end'])]

    if fds is None:
        fds = [(ts_settings['fd_start'], ts_settings['fd_end'])]

    clusters = range(1) if split_col is None else df[split_col].unique()

    assert isinstance(fdws, list), 'fdws must be a list object'
    assert isinstance(fds, list), 'fds must be a list object'
    if split_col:
        assert len(df[split_col].unique()) > 1, 'There must be at least 2 clusters'

    n_projects = len(clusters) * len(fdws) * len(fds)
    print(f'Kicking off {n_projects} projects\n')

    projects = []
    for c in clusters:
        for fdw in fdws:
            for fd in fds:
                ts_settings['fd_start'], ts_settings['fd_end'] = fd[0], fd[1]
                ts_settings['fdw_start'], ts_settings['fdw_end'] = fdw[0], fdw[1]
                cluster_suffix = 'all_series' if split_col is None else 'Cluster-' + c.astype('str')

                # Name project
                project_name = '{prefix}_FD:{start}-{end}_FDW:{fdw}_{cluster}'.format(
                    prefix=prefix,
                    fdw=ts_settings['fdw_start'],
                    start=ts_settings['fd_start'],
                    end=ts_settings['fd_end'],
                    cluster=cluster_suffix,
                )

                if split_col is not None:
                    data = df.loc[df[split_col] == c, :].copy()
                    data.drop(columns=split_col, axis=1, inplace=True)
                else:
                    data = df.copy()

                # Create project
                project = create_dr_project(
                    data, project_name, ts_settings, advanced_options=advanced_options
                )
                projects.append(project)

    return projects


def wait_for_jobs_to_process(projects):
    """
    Check if any DataRobot jobs are still processing
    """
    all_jobs = np.sum([len(p.get_all_jobs()) for p in projects])
    while all_jobs > 0:
        print(f'There are {all_jobs} jobs still processing')
        time.sleep(60)
        all_jobs = np.sum([len(p.get_all_jobs()) for p in projects])

    print('All jobs have finished processing...')