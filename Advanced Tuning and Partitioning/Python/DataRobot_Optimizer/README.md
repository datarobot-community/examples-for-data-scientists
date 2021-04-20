# Controlling DataRobot Optimizer through Python

This code accompanies a community article: https://community.datarobot.com/t5/resources/interacting-with-the-optimizer-app-through-the-api/ta-p/11426.
In some cases, you want to find the combination of values for some features that give the best predicted value. For instance, consider a bank that wants to reduce loan risk by determining which loan size and other factors for given borrowers results in the lowest risk. For this purpose, the bank will build a model that predicts load default, and then use the Optimizer App to figure out the loan size and other factors.
This notebook shows how to interact with the DataRobot Optimizer App using an example Lending Club dataset, "Lending Club Sample 30.csv". In this example, we are trying to find the best combination of values for `revol_util`, `inq_last_6mths`, `loan_amnt`, and `dti` that minimizes the probability of a loan going bad.
