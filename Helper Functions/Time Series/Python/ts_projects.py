#Author: Justin Swansburg, Mark Philip

#Make sure you are connected to DataRobot Client.

#The functions below will help you evaluate a DataRobot TS project.

import datarobot as dr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ts_metrics import *


######################
# Project Evaluation
######################


def get_top_models_from_project(
    project, n_models=1, data_subset='allBacktests', include_blenders=True, metric=None
):
    """
    project: project object
        DataRobot project
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    Returns:
    --------
    List of model objects from a DataRobot project
    """
    assert data_subset in [
        'backtest_1',
        'allBacktests',
        'holdout',
    ], 'data_subset must be either backtest_1, allBacktests, or holdout'
    if n_models is not None:
        assert isinstance(n_models, int), 'n_models must be an int'
    if n_models is not None:
        assert n_models >= 1, 'n_models must be greater than or equal to 1'
    assert isinstance(include_blenders, bool), 'include_blenders must be a boolean'

    mapper = {
        'backtest_1': 'backtestingScores',
        'allBacktests': 'backtesting',
        'holdout': 'holdout',
    }

    if metric is None:
        metric = project.metric

    if data_subset == 'holdout':
        project.unlock_holdout()

    models = [
        m
        for m in project.get_datetime_models()
        if m.backtests[0]['status'] != 'BACKTEST_BOUNDARIES_EXCEEDED'
    ]  # if m.holdout_status != 'HOLDOUT_BOUNDARIES_EXCEEDED']

    if data_subset == 'backtest_1':
        # models = sorted(models, key=lambda m: np.mean([i for i in m.metrics[metric][mapper[data_subset]][0] if i]), reverse=False)
        models = sorted(
            models, key=lambda m: m.metrics[metric][mapper[data_subset]][0], reverse=False
        )
    elif data_subset == 'allBacktests':
        models = sorted(
            models,
            key=lambda m: m.metrics[metric][mapper[data_subset]]
            if m.metrics[metric][mapper[data_subset]] is not None
            else np.nan,
            reverse=False,
        )
    else:
        models = sorted(models, key=lambda m: m.metrics[metric][mapper[data_subset]], reverse=False)

    if not include_blenders:
        models = [m for m in models if m.model_category != 'blend']

    if n_models is None:
        n_models = len(models)

    models = models[0:n_models]

    assert len(models) > 0, 'You have not run any models for this project'

    return models


def get_top_models_from_projects(
    projects, n_models=1, data_subset='allBacktests', include_blenders=True, metric=None
):
    """
    Pull top models from leaderboard across multiple DataRobot projects
    projects: list
        DataRobot project object(s)
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    Returns:
    --------
    List of model objects from DataRobot project(s)
    """
    assert isinstance(projects, list), 'Projects must be a list object'

    models_all = []
    for p in projects:
        models = get_top_models_from_project(p, n_models, data_subset, include_blenders, metric)
        models_all.extend(models)
    return models_all


def compute_backtests(
    projects, n_models=5, data_subset='backtest_1', include_blenders=True, metric=None
):
    """
    Compute all backtests for top models across multiple DataRobot projects
    projects: list
        DataRobot project object(s)
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    """
    assert isinstance(projects, list), 'Projects must be a list object'

    for p in projects:
        models = get_top_models_from_project(
            p,
            n_models=n_models,
            data_subset=data_subset,
            include_blenders=include_blenders,
            metric=metric,
        )

        for m in models:
            try:
                m.score_backtests()  # request backtests for top models
                print(f'Computing backtests for model {m.id} in Project {p.project_name}')
            except dr.errors.ClientError:
                pass
        print(
            f'All available backtests have been submitted for scoring for project {p.project_name}'
        )


def get_or_request_backtest_scores(
    projects, n_models=5, data_subset='allBacktests', include_blenders=True, metric=None
):
    """
    Get or request backtest and holdout scores from top models across multiple DataRobot projects
    projects: list
        DataRobot project object(s)
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    Returns:
    --------
    pandas df
    """
    assert isinstance(projects, list), 'Projects must be a list object'

    scores = pd.DataFrame()
    for p in projects:

        models = get_top_models_from_project(
            p,
            n_models=n_models,
            data_subset=data_subset,
            include_blenders=include_blenders,
            metric=metric,
        )

        if metric is None:
            metric = p.metric

        backtest_scores = pd.DataFrame(
            [
                {
                    'Project_Name': p.project_name,
                    'Project_ID': p.id,
                    'Model_ID': m.id,
                    'Model_Type': m.model_type,
                    'Featurelist': m.featurelist_name,
                    f'Backtest_1_{metric}': m.metrics[metric]['backtestingScores'][0],
                    'Backtest_1_MASE': m.metrics['MASE']['backtestingScores'][0],
                    'Backtest_1_Theils_U': m.metrics["Theil's U"]['backtestingScores'][0],
                    'Backtest_1_SMAPE': m.metrics['SMAPE']['backtestingScores'][0],
                    'Backtest_1_R_Squared': m.metrics['R Squared']['backtestingScores'][0],
                    f'All_Backtests_{metric}': m.metrics[metric]['backtestingScores'],
                    'All_Backtests_MASE': m.metrics['MASE']['backtestingScores'],
                    'All_Backtests_Theils_U': m.metrics["Theil's U"]['backtestingScores'],
                    'All_Backtests_SMAPE': m.metrics['SMAPE']['backtestingScores'],
                    'All_Backtests_R_Squared': m.metrics['R Squared']['backtestingScores'],
                    f'Holdout_{metric}': m.metrics[metric]['holdout'],
                    'Holdout_MASE': m.metrics['MASE']['holdout'],
                    'Holdout_Theils_U': m.metrics["Theil's U"]['holdout'],
                    'Holdout_SMAPE': m.metrics['SMAPE']['holdout'],
                    'Holdout_R_Squared': m.metrics['R Squared']['holdout'],
                }
                for m in models
            ]
        ).sort_values(by=[f'Backtest_1_{metric}'])

        scores = scores.append(backtest_scores).reset_index(
            drop=True
        )  # append top model from each project

    print(f'Scores for all {len(projects)} projects have been computed')

    return scores


def get_or_request_training_predictions_from_model(model, data_subset='allBacktests'):
    project = dr.Project.get(model.project_id)

    if data_subset == 'holdout':
        project.unlock_holdout()

    try:
        predict_job = model.request_training_predictions(data_subset)
        training_predictions = predict_job.get_result_when_complete(max_wait=10000)

    except dr.errors.ClientError:
        prediction_id = [
            p.prediction_id
            for p in dr.TrainingPredictions.list(project.id)
            if p.model_id == model.id and p.data_subset == data_subset
        ][0]
        training_predictions = dr.TrainingPredictions.get(project.id, prediction_id)

    return training_predictions.get_all_as_dataframe(serializer='csv')


def get_or_request_training_predictions_from_projects(
    projects, n_models=1, data_subset='allBacktests', include_blenders=True, metric=None
):
    """
    Get row-level backtest or holdout predictions from top models across multiple DataRobot projects
    projects: list
        DataRobot project object(s)
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    Returns:
    --------
    pandas Series
    """
    assert isinstance(projects, list), 'Projects must be a list object'

    preds = pd.DataFrame()
    for p in projects:
        models = get_top_models_from_project(p, n_models, data_subset, include_blenders, metric)

        for m in models:
            tmp = get_or_request_training_predictions_from_model(m, data_subset)
            tmp['Project_Name'] = p.project_name
            tmp['Project_ID'] = p.id
            tmp['Model_ID'] = m.id
            tmp['Model_Type'] = m.model_type
        preds = preds.append(tmp).reset_index(drop=True)

    return preds


def get_preds_and_actuals(
    df,
    projects,
    ts_settings,
    n_models=1,
    data_subset='allBacktests',
    include_blenders=True,
    metric=None,
):
    """
    Get row-level predictions and merge onto actuals
    df: pandas df
    projects: list
        DataRobot project object(s)
    ts_settings: dict
        Parameters for time series project
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    Returns:
    --------
    pandas df
    """
    assert isinstance(projects, list), 'Projects must be a list object'

    preds = get_or_request_training_predictions_from_projects(
        projects,
        n_models=1,
        data_subset=data_subset,
        include_blenders=include_blenders,
        metric=metric,
    )
    preds['timestamp'] = pd.to_datetime(preds['timestamp'].apply(lambda x: x[:-8]))
    df = df.merge(
        preds,
        how='left',
        left_on=[ts_settings['date_col'], ts_settings['series_id']],
        right_on=['timestamp', 'series_id'],
        validate='one_to_many',
    )
    df = df.loc[~np.isnan(df['prediction']), :].reset_index(drop=True)
    return df


def get_cluster_acc(
    df,
    projects,
    ts_settings,
    data_subset='allBacktests',
    include_blenders=True,
    metric=None,
    acc_calc=rmse,
):
    """
    Get cluster-level and overall accuracy across multiple DataRobot projects
    df: pandas df
    projects: list
        DataRobot project object(s)
    ts_settings: dict
        Parameters for time series project
    data_subset: str
        Valid values are either holdout or allBacktests
    include_backtests: boolean (optional)
        Controls whether blender models are considered
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    acc_calc: function
        Function to calculate row-level prediction accuracy. Choose from mae, rmse, mape, smape, gamma, poission, and tweedie
    Returns:
    --------
    pandas df
    """
    assert isinstance(projects, list), 'Projects must be a list object'
    assert data_subset in [
        'allBacktests',
        'holdout',
    ], 'data_subset must be either allBacktests or holdout'

    print('Getting cluster accuracy...')

    df = get_preds_and_actuals(
        df,
        projects,
        ts_settings,
        n_models=1,
        data_subset=data_subset,
        include_blenders=include_blenders,
        metric=metric,
    )
    df = get_project_info(df)

    groups = (
        df.groupby(['Cluster'])
        .apply(lambda x: acc_calc(x[ts_settings['target']], x['prediction']))
        .reset_index()
    )
    groups.columns = ['Cluster', f'Cluster_{acc_calc.__name__.upper()}']
    groups[f'Total_{acc_calc.__name__.upper()}'] = acc_calc(
        act=df[ts_settings['target']], pred=df['prediction']
    )

    return groups


def plot_cluster_acc(cluster_acc, ts_settings, data_subset='allBacktests', acc_calc=rmse):
    """
    Plots cluster-level and overall accuracy across multiple DataRobot projects
    cluster_acc: pandas df
        Output from get_cluster_acc()
    ts_settings: dict
        Pparameters for time series project
    data_subset: str
        Choose either holdout or allBacktests
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    Returns:
    --------
    Plotly barplot
    """
    cluster_acc['Label'] = '=' + cluster_acc['Cluster']

    fig = px.bar(cluster_acc, x='Label', y=f'Cluster_{acc_calc.__name__.upper()}').for_each_trace(
        lambda t: t.update(name=t.name.replace('=', ''))
    )

    fig.add_trace(
        go.Scatter(
            x=cluster_acc['Label'],
            y=cluster_acc[f'Total_{acc_calc.__name__.upper()}'],
            mode='lines',
            marker=dict(color='black'),
            name=f'Overall {acc_calc.__name__.upper()}',
        )
    )

    fig.update_yaxes(title=acc_calc.__name__.upper())
    fig.update_xaxes(tickangle=45)
    fig.update_layout(title_text=f'Cluster Accuracy - {data_subset}')
    fig.show()


def get_series_acc(
    df,
    projects,
    ts_settings,
    data_subset='allBacktests',
    include_blenders=True,
    metric=None,
    acc_calc=rmse,
):
    """
    Get series-level and overall accuracy across multiple DataRobot projects
    df: pandas df
    projects: list
        DataRobot project object(s)
    ts_settings: dict
        Parameters for time series project
    data_subset: str
        Valid values are either holdout or allBacktests
    include_backtests: boolean (optional)
        Controls whether blender models are considered
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    acc_calc: function
        Function to calculate row-level prediction accuracy. Choose from mae, rmse, mape, smape, gamma, poission, and tweedie
    Returns:
    --------
    pandas df
    """
    assert isinstance(projects, list), 'Projects must be a list object'
    assert data_subset in [
        'allBacktests',
        'holdout',
    ], 'data_subset must be either allBacktests or holdout'

    series_id = ts_settings['series_id']
    target = ts_settings['target']

    print('Getting series accuracy...')

    df = get_preds_and_actuals(
        df,
        projects,
        ts_settings,
        n_models=1,
        data_subset=data_subset,
        include_blenders=include_blenders,
        metric=metric,
    )
    df = get_project_info(df)

    groups = (
        df.groupby([series_id]).apply(lambda x: acc_calc(x[target], x['prediction'])).reset_index()
    )
    groups.columns = [series_id, f'Series_{acc_calc.__name__.upper()}']
    right = df[[series_id, 'Cluster']].drop_duplicates().reset_index(drop=True)
    groups = groups.merge(right, how='left', on=series_id)
    groups[f'Total_{acc_calc.__name__.upper()}'] = acc_calc(act=df[target], pred=df['prediction'])

    return groups


def plot_series_acc(series_acc, ts_settings, data_subset='allBacktests', acc_calc=rmse, n=50):
    """
    Plots series-level and overall accuracy across multiple DataRobot projects
    cluster_acc: pandas df
        Output from get_series_acc()
    ts_settings: dict
        Parameters for time series project
    data_subset: str
        Choose from either holdout or allBacktests
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    Returns:
    --------
    Plotly barplot
    """
    n_series = len(series_acc[ts_settings['series_id']].unique())
    n = min(n_series, n)

    series_acc.sort_values(by=f'Series_{acc_calc.__name__.upper()}', ascending=False, inplace=True)

    series_acc = series_acc[0:n]

    fig = px.bar(
        series_acc,
        x=ts_settings['series_id'],
        y=f'Series_{acc_calc.__name__.upper()}',
        color='Cluster',
    ).for_each_trace(lambda t: t.update(name=t.name.replace('Project_Name=', '')))

    fig.add_trace(
        go.Scatter(
            x=series_acc[ts_settings['series_id']],
            y=series_acc[f'Total_{acc_calc.__name__.upper()}'],
            mode='lines',
            marker=dict(color='black'),
            name=f'Overall {acc_calc.__name__.upper()}',
        )
    )

    fig.update_yaxes(title=acc_calc.__name__.upper())
    fig.update_xaxes(tickangle=45)
    fig.update_layout(title_text=f'Series Accuracy - {data_subset}')
    fig.show()


def get_project_info(df):
    """
    Parse project name to get FD, FDW, and Cluster information
    Returns:
    --------
    pandas df
    """
    df = df.copy()
    try:
        df['Cluster'] = df['Project_Name'].apply(lambda x: x.split('_Cluster-')[1])
    except:
        df['Cluster'] = 'all_series'

    df['FD'] = df['Project_Name'].apply(lambda x: x.split('_FD:')[1].split('_FDW:')[0])
    df['FDW'] = df['Project_Name'].apply(lambda x: x.split('_FDW:')[1].split('_Cluster-')[0])

    return df


def filter_best_fdw_scores(scores, col_error='All_Backtests_RMSE'):
    """
    Subset df to projects with the best error metric for each FD and Cluster pair
    scores: pandas df
        Output from get_or_request_backtest_scores()
    col_error: str
        Column name from scores df
    Returns:
    --------
    pandas df
    """
    df = get_project_info(scores)
    df['_tmp'] = df[col_error].apply(lambda x: np.nanmean(np.array(x, dtype=np.float32)))
    idx = df.groupby(['Cluster', 'FD']).apply(lambda x: x['_tmp'].idxmin()).values
    return scores.iloc[idx, :]


def filter_best_fdw_projects(scores, projects, col_error='All_Backtests_RMSE'):
    """
    Subset list to projects with the best error metric for each FD and Cluster pair
    scores: pandas df
        Output from get_or_request_backtest_scores()
    projects: list
        DataRobot projects object(s)
    col_error: str
        Column name from scores df
    Returns:
    --------
    list
    """
    df = filter_best_fdw_scores(scores, col_error)
    return [p for p in projects if p.project_name in df['Project_Name'].unique()]


def plot_fd_accuracy(df, projects, ts_settings, data_subset='allBacktests', metric='RMSE'):
    assert isinstance(projects, list), 'Projects must be a list object'
    assert data_subset in [
        'allBacktests',
        'holdout',
    ], 'data_subset must be either allBacktests or holdout'

    mapper = {
        'MAE': mae,
        'RMSE': rmse,
        'Gamma Deviance': gamma_loss,
        'Tweedie Deviance': tweedie_loss,
        'Poisson Deviance': poisson_loss,
    }

    df = get_preds_and_actuals(
        df, projects, ts_settings, n_models=1, data_subset=data_subset, metric=metric
    )
    df = (
        df.groupby(['Project_Name', 'forecast_distance'])
        .apply(lambda x: mapper[metric](x[ts_settings['target']], x['prediction']))
        .reset_index()
    )

    df.columns = ['Project_Name', 'forecast_distance', mapper[metric].__name__.upper()]
    fig = px.line(
        df, x='forecast_distance', y=mapper[metric].__name__.upper(), color='Project_Name'
    ).for_each_trace(lambda t: t.update(name=t.name.replace('Project_Name=', '')))

    fig.update_layout(title_text='Forecasting Accuracy per Forecast Distance')
    fig.update_yaxes(title=mapper[metric].__name__.upper())
    fig.update_xaxes(title='Forecast Distance')
    fig.show()


def plot_fd_accuracy_by_cluster(
    df, scores, projects, ts_settings, data_subset='holdout', metric='RMSE', split_col='Cluster'
):
    scores = get_project_info(scores)

    for c in scores[split_col].unique():
        project_names = list(
            scores.loc[scores[split_col] == c, 'Project_Name'].reset_index(drop=True)
        )
        projects_by_cluster = [p for p in projects if p.project_name in project_names]
        plot_fd_accuracy(df, projects_by_cluster, ts_settings, data_subset, metric)


###########################
# Performance Improvements
###########################


def get_reduced_features_featurelist(project, model, threshold=0.99):
    """
    Helper function for train_reduced_features_models()
    project: DataRobot project object
    model: DataRobot model object
    threshold: np.float
    Returns:
    --------
    DataRobot featurelist
    """
    print(
        f'Collecting Feature Impact for M{model.model_number} in project {project.project_name}...'
    )

    impact = pd.DataFrame.from_records(model.get_or_request_feature_impact())
    impact['impactUnnormalized'] = np.where(
        impact['impactUnnormalized'] < 0, 0, impact['impactUnnormalized']
    )
    impact['cumulative_impact'] = (
        impact['impactUnnormalized'].cumsum() / impact['impactUnnormalized'].sum()
    )

    to_keep = np.where(impact['cumulative_impact'] <= threshold)[0]
    if len(to_keep) < 1:
        print('Applying this threshold would result in a featurelist with no features')
        return None

    idx = np.max(to_keep)

    selected_features = impact.loc[0:idx, 'featureName'].to_list()
    feature_list = project.create_modeling_featurelist(
        f'Top {len(selected_features)} features M{model.model_number}', selected_features
    )

    return feature_list


def train_reduced_features_models(
    projects,
    n_models=1,
    threshold=0.99,
    data_subset='allBacktests',
    include_blenders=True,
    metric=None,
):
    """
    Retrain top models with reduced feature featurelists
    projects: list
        DataRobot project objects(s)
    n_models: int
        Number of models to retrain with reduced feature featurelists
    threshold: np.float
        Controls the number of features to keep in the reduced feature list. Percentage of cumulative feature impact
    data_subset: str
        Choose from either holdout or allBacktests
    """
    assert isinstance(projects, list), 'Projects must be a list object'

    for p in projects:
        models = get_top_models_from_project(p, n_models, data_subset, include_blenders, metric)

        for m in models:
            try:
                feature_list = get_reduced_features_featurelist(p, m, threshold)
                if feature_list is None:
                    continue
                try:
                    m.retrain(featurelist_id=feature_list.id)
                    print(f'Training {m.model_type} on Featurelist {feature_list.name}')
                except dr.errors.ClientError as e:
                    print(e)
            except dr.errors.ClientError as e:
                print(e)