# Soccer prediction demo with Feature Discovery through API

## Background
The goal of this project is to predict the number of goals that will be scored by the home and away teams, respectively, in future matches. We leverage DataRobot's Automated Feature Discovery tool (AFD) to engineer many features which we test in models built using Automated Machine Learning (AutoML). 

Using the predicted number of goals scored for teams, and assuming each prediction is independently distributed as Poisson (a fairly reasonable simplifying assumption), we can also generate the probabilities of all possible score lines (e.g., a 3-2 win for the home team). This project was created such that the outputs could be used to simulate the goal outcomes of all remaining MLS games this season, to estimate probabilities of various playoff outcomes.

The data used in this project come from American Soccer Analysis's API, which feeds their publicly available data application (here)[https://app.americansocceranalysis.com]. Major League Soccer data were used in the examples here, but you can also get historical data from the National Women's Soccer League (NWSL), the United Soccer Leagues (USL), and the North American Soccer League (NASL, now defunct).

## Contents
This repository contains:
- An R notebook for gathering publicly available soccer data online and create primary and secondary datasets for a DataRobot AutoML project
- A snapshot of the datasets produced by the notebook above, as of 2021-07-13
- A Python notebook for setting up AutoML projects to predict goals scored for home and away teams in future matches

There is more information contained in each notebook.
