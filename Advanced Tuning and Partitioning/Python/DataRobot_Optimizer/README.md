# Controlling DataRobot Optimizer Through Python

TODO: In some cases, one wants to find best combination of values of some features that give the best predicted value.  For instance, a bank wants to reduce loan risk by trying to figure out the loan size and other factors for a given borrowers that results in the lowest risk.  For this purpose, the bank builds a model to predict default of a loan, and then use the optimizer to figure out the loan size and other factors. 
This notebook shows how to interact to DataRobot Optimizer using as example Lending Club dataset "Lending Club Sample 30.csv". In this example, we are trying to find the best combination of values for revol_util, inq_last_6mths, loan_amnt, and dti that minizes the probability of a loan going bad.
 

## Usage
1. Change key_dict
      a. Get the url from the application (see figure below)
      b. Put the values in key_dict
2. Put the name of the dataset in ts_settings["filename"]
3. Read file into a dataframe
4. result_df = perform_optimization(data_df)


## Repository Contents

*get_optimization* : make a post request to perform optimization.  It returns optimized values for the constraint features, and the predicted target

*get_constraints*  : access the constraint features and their range from the optimizer

*create_constrain_from_df* : In case one wants to decide on the fly which features to constrain, one has to change the list in "cfeatures" in ts_settings and provide a file to estimate these features min and the max

*set_optimizer* : Prepare the elements required by the optimizer


## Development and Contributing

If you'd like to report an issue or bug, suggest improvements, or contribute code to this project, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).


# Code of Conduct

This project has adopted the Contributor Covenant for its Code of Conduct. 
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) to read it in full.

# License

Licensed under the Apache License 2.0. 
See [LICENSE](LICENSE) to read it in full.


