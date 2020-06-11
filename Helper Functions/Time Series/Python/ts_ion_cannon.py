#Author: Lukas Innig

#Make sure you are connected to DataRobot and have a completed TS project.

import datarobot as dr
from datarobot.errors import ClientError

class TimeSeriesIonCannon(dr.Project):
    """ This class takes as input a DataRobot Object and initiates a brute force search to increase accuracy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        all_models = self.get_models()
        self.supported_metrics = all_models[0].metrics.keys()
        self.training_duration = [m for m in all_models if m.training_duration][0].training_duration
    sort_order = {'MASE': False,
     'FVE Poisson': True,
     "Theil's U": False,
     'RMSE': False,
     'FVE Gamma': True,
     'R Squared': True,
     'Gamma Deviance': False,
     'FVE Tweedie': True,
     'MAE': False,
     'SMAPE': True,
     'MAPE': True,
     'Gini Norm': True,
     'Tweedie Deviance': False,
     'Poisson Deviance': False,
     'RMSLE': False}
    
    @classmethod
    def aim(cls, *args, **kwargs):
        return super().get(*args, **kwargs)
    
    def get_models_sorted(self, partition='validation', metric='RMSE', model_type_filter = ['']):
        if partition not in ['backtesting', 'holdout', 'validation']:
            raise ValueError(f"Partition {partition} not in ['backtesting', 'holdout', 'validation']")
        if partition == 'holdout' and not self.holdout_unlocked:
            print("Holdout not unlocked!")
            return []
        if metric not in self.supported_metrics:
            raise ValueError(f'Metric {metric} not supported')
        reverse = self.sort_order.get(metric)
        return sorted([m for m in self.get_datetime_models() 
                       if metric in m.metrics 
                       and m.metrics[metric][partition] 
                       and any([f in m.model_type for f in model_type_filter])], 
                      key=lambda m: m.metrics[metric][partition], reverse=reverse)
    
    def calculate_backtests(self, models):
        def score_backtests(m):
            try: 
                return m.score_backtests()
            except ClientError as e:
                return None
        jobs = [score_backtests(m) for m in models]
        [job.wait_for_completion() for job in jobs if job]
    
    def identify_best_featurelist(self):
        best_models = self.get_models_sorted('backtesting')
        if not best_models:
            print('calculate some backtests')
        featurelists = [m.featurelist_id for m in best_models[:20] if 'Blender' not in m.model_type]
        reduced_fl = [fl for fl in featurelists if 'Reduced' in fl.name]
        other_fl = [fl for fl in featurelists if 'Reduced' not in fl.name]
        return reduced_fl + other_fl[:1]
    
    def run_all_blueprints(self, featurelist, training_duration=None, 
                           model_type_filter=['Mean', 'Eureqa', 'Keras', 'VARMAX']):
        if not training_duration:
            training_duration = self.training_duration
        def train_blueprint(bp, fl):
            try:
                return self.train_datetime(bp.id, fl.id, training_duration=training_duration)
            except ClientError as e:
                print(e)
                return None
        bps = [bp for bp in self.get_blueprints() if all([f not in bp.model_type for f in model_type_filter])]
        jobs = [train_blueprint(bp, featurelist) for bp in bps]
        [job.wait_for_completion() for job in jobs if job]
    def run_blenders(self):
        def blend(model_ids, blender_method):
            try:
                return self.blend(model_ids, blender_method)
            except ClientError as e:
                print(e)
                return None
        best_models = self.get_models_sorted('backtesting')
        best_models = [m for m in best_models if 'Blender' not in m.model_type]
        jobs = []
        for n in [3, 5, 7]:
            for blender_method in [dr.enums.BLENDER_METHOD.FORECAST_DISTANCE_AVG, 
                                   dr.enums.BLENDER_METHOD.AVERAGE,
                                   dr.enums.BLENDER_METHOD.FORECAST_DISTANCE_ENET]:
                jobs.append(blend([m.id for m in best_models[:n]], blender_method=blender_method))
        blender_models = [j.get_result_when_complete() for j in jobs if j]
        blender_models = [dr.DatetimeModel.get(self.id, bm.id) for bm in blender_models]
        return blender_models
    def shoot(self):
        self.calculate_backtests(self.get_models_sorted('validation')[:20])
        fls = self.identify_best_featurelist()
        for fl in fls:
            self.run_all_blueprints(fl)
        self.calculate_backtests(self.get_models_sorted('validation')[:20])
        self.run_blenders()
        self.calculate_backtests(self.get_models_sorted('validation')[:20])


##USAGE##
#cannon = TimeSeriesIonCannon.aim('YOUR_PROJECT_ID')
#cannon.shoot()