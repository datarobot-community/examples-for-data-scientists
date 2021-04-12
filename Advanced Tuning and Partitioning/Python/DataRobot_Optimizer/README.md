# Controlling DataRobot Optimizer Through Python

In some cases, one wants to find best combination of values of some features that give the best predicted value.  For instance, a bank wants to reduce loan risk by trying to figure out the loan size and other factors for a given borrowers that results in the lowest risk.  For this purpose, the bank builds a model to predict default of a loan, and then use the optimizer to figure out the loan size and other factors. 
This notebook shows how to interact to DataRobot Optimizer using as example Lending Club dataset "Lending Club Sample 30.csv". In this example, we are trying to find the best combination of values for revol_util, inq_last_6mths, loan_amnt, and dti that minizes the probability of a loan going bad.
