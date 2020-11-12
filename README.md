# API Samples for Data Scientists

This repository contains Python notebooks and R Markdown guides for achieving specific tasks using the API.

Start learning with the [API Training](https://github.com/datarobot-community/tutorials-for-data-scientists/tree/master/DRU/API_Training) module.

## Usage

For each respective guide, follow the instructions in its own `.ipynb` or `.Rmd` file. 

**Please pay attention to the different DataRobot API Endpoints**

The API endpoint you specify for accessing DataRobot is dependent on the deployment environment, as follows:

- AI Platform Trial—https://app2.datarobot.com/api/v2
- US Managed AI Cloud—https://app.datarobot.com/api/v2
- EU Managed AI Cloud—https://app.eu.datarobot.com/api/v2
- On-Premise—https://{datarobot.example.com}/api/v2 
       (replacing {datarobot.example.com} with your specific deployment endpoint
       
The DataRobot API Endpoint is used to connect your IDE to DataRobot.

## Important Links

- To learn to use DataRobot, visit [DataRobot University](https://university.datarobot.com/)
- For General articles on DataRobot and news, visit [DataRobot Community](https://community.datarobot.com/)
- End to end DataRobot API [Tutorials for Data Scientists](https://github.com/datarobot-community/tutorials-for-data-scientists)

## Contents

### Advanced Tuning

- *Advanced Tuning:* how to do advanced tuning. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Advanced%20Tuning%20and%20Partitioning/Python/Advanced%20Tuning.ipynb) [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Advanced%20Tuning%20and%20Partitioning/R/Advanced_Tuning.Rmd)

- *Datetime Partitioning:* how to do datetime partitioning. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Advanced%20Tuning%20and%20Partitioning/Python/Datetime%20Partitioning.ipynb) [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Advanced%20Tuning%20and%20Partitioning/R/Datetime_Partitioning.Rmd)

### Compliance Documentation

- *Getting Compliance Documentation:* how to get Compliance Documentation documents.  [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Compliance%20Docs/Python/Getting%20Compliance%20Documentation.ipynb)  [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Compliance%20Docs/R/Getting_Compliance_Documentation.Rmd)

### Feature Lists Manipulation

- *Advanced Feature Selection:* how to do advanced feature selection using all of the models created during a run of DataRobot autopilot. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Feature%20Lists%20Manipulation/Python/Advanced%20Feature%20Selection.ipynb)  [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Feature%20Lists%20Manipulation/R/Advanced_Feature_Selection.Rmd)

- *Feature Lists Manipulation:* how to create and manipulate custom feature lists and use it for training.  [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Feature%20Lists%20Manipulation/Python/Feature%20Lists%20Manipulation.ipynb)  [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Feature%20Lists%20Manipulation/R/Feature_Lists_Manipulation.Rmd)

- *Transforming Feature Type:* how to transform feature types.  [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Feature%20Lists%20Manipulation/Python/Transforming%20Feature%20Types.ipynb)  [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Feature%20Lists%20Manipulation/R/Transforming_Feature_Types.Rmd)

### Helper Functions

- *Modeling/Python:* A function that helps you search for specific blueprints within a DataRobot's project's repository and then initiates all of these models. [Python](https://github.com/datarobot-community/examples-for-data-scientists/tree/master/Helper%20Functions/Modeling/Python)

- *Time Series/Python:* a set of custom functions for AutoTS projects (advanced settings, data quality, filling dates, preprocessing, brute force, cloning, accuracy metrics, modeling, project lists). [Python](https://github.com/datarobot-community/examples-for-data-scientists/tree/master/Helper%20Functions/Time%20Series/Python)

### Initiating Projects

- *Starting a Binary Classification Project:* how to initiate a DataRobot project for a Binary Classification target. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Initiating%20Projects/Python/Starting%20a%20Binary%20Classification%20Project.ipynb)  [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Initiating%20Projects/R/Starting_a_Binary_Classification_Project.Rmd)

- *Starting a Multiclass Project:* how to initiate a DataRobot project for a Multiclass Classification target.  [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Initiating%20Projects/Python/Starting%20a%20Multiclass%20Classification%20Project.ipynb)  [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Initiating%20Projects/R/Starting_a_Multiclass_Classification_Project.Rmd)

- *Starting a Project with Selected Blueprints:* how to initiate a DataRobot project manually where the user has the option to choose which models/blueprints to initiate.   [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Initiating%20Projects/Python/Starting%20a%20Project%20with%20Selected%20Blueprints.ipynb)  [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Initiating%20Projects/R/Starting_a_Project_with_Selected_Blueprints.Rmd)

- *Starting a Regression Project:* how to initiate a DataRobot project for a numerical target. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Initiating%20Projects/Python/Starting%20a%20Regression%20Project.ipynb)  [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Initiating%20Projects/R/Starting_a_Regression_Project.Rmd)

- *Starting a Time Series Project:* how to initiate a DataRobot project for a Time Series problem. This notebook also covers calendars and feature settings for time series projects.  [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Initiating%20Projects/Python/Starting%20a%20Time%20Series%20Project.ipynb)  [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Initiating%20Projects/R/Starting_a_time_Series_Project.Rmd)

### Making Predictions

- *Getting Predictions and Prediction Explanations:* how to get predictions and prediction explanations out of a trained model. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Making%20Predictions/Python/Getting%20Predictions%20and%20Prediction%20Explanations.ipynb) [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Making%20Predictions/R/Getting%20Predictions%20and%20Prediction%20Explanations.Rmd)

- *Scoring Big Datasets - Batch Prediction API:* how to use DataRobot's batch prediction script to get predictions out of a DataRobot deployed model. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Making%20Predictions/Python/Scoring%20Big%20Datasets--Batch%20Prediction%20API.ipynb) 

- *Prediction Explanation Clustering:*  creating clusters of prediction explanations to better understand patterns in your data. [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Making%20Predictions/R/PredictionExplanationClustering.Rmd)

### Model Evaluation

- *Getting Confusion Chart:* how to get the Confusion Matrix Chart. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/Python/Getting%20Confusion%20Chart.ipynb) [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/R/Getting_Confusion_Chart.Rmd)

- *Getting Feature Impact:* how to get the Feature Impact scores. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/Python/Getting%20Feature%20Impact.ipynb) [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/R/Getting_Feature_Impact.Rmd)

- *Getting Lift Chart:* how to get the lift chart. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/Python/Getting%20Lift%20Chart.ipynb) [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/R/Getting_Lift_Chart.Rmd)

- *Getting Partial Dependence:* how to get partial dependence.[Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/Python/Getting%20Partial%20Dependence%20Plot.ipynb) [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/R/Getting_Partial_Dependence_Plot.rmd)

- *Getting ROC Curve:* how to get the ROC Curve data. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/Python/Getting%20ROC%20Curve.ipynb)  [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/R/Getting_ROC_Curve.Rmd)

- *Getting SHAP Values:* how to get SHAP values.  [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/Python/Getting%20SHAP%20Values.ipynb)

- *Getting Word Cloud:* how to pull the word cloud data. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/Python/Getting%20Word%20Cloud.ipynb) [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/R/Getting_Word_Cloud.Rmd)

- *Plotting Prediction Intervals:* how to plot prediction intervals for time series projects (single and multi series).  [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Evaluation/Python/Plotting%20Prediction%20Intervals%20for%20Time%20Series%20Projects.ipynb)

### Model Management

- *Model Management and Monitoring:* how to manage models through the API. This includes deployment, replacement, deletion, and monitoring capabilities. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Management/Python/Model%20Management%20and%20Monitoring.ipynb) [R](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Management/R/Model_Management_and_Monitoring.Rmd)

- *Sharing Projects:* how to share projects with colleagues. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Management/Python/Sharing%20Projects.ipynb)

- *Uploading Actuals to a DataRobot Deployment:* how to upload actuals into the DataRobot platform in order to calculate accuracy metrics [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Model%20Management/Python/Uploading%20Actuals%20to%20a%20DataRobot%20Deployment.ipynb)

### AI Catalog

- *AI Catalog API Demo:* how to create and share datasets in AI Catalog and use them to create projects and run predictions. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/AI%20Catalog/AI_Catalog_API.ipynb)

### Paxata
- *Paxata Functions: A collection of functions to interact with DataRobot Paxata. [Python](https://github.com/datarobot-community/examples-for-data-scientists/blob/master/Paxata/paxata_functions.py)


## Development and Contributing

If you'd like to report an issue or bug, suggest improvements, or contribute code to this project, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).


# Code of Conduct

This project has adopted the Contributor Covenant for its Code of Conduct. 
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) to read it in full.

# License

Licensed under the Apache License 2.0. 
See [LICENSE](LICENSE) to read it in full.


