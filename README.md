NBA Daily Fantasy Analysis
==============================

This project was part of my coursework during Metis Immersive Data Science Bootcamp

**Project Status - Active**

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Scratch Work
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
            ├── kmeans_utilities.py    
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   └── model_utilities.py



**Project Objective**
--------------

Predict the daily fantasy point output of NBA players to use in Daily Fantasy Contests

**Methods Used**
* Exploratory Data Analysis
* Data Visualization
* Machine Learning
* Predictive Modeling

**Technologies Used**
* Python 3.7
* Jupyter Notebooks
* pandas
* numpy
* sklearn

**Project Overview**
--------------
NBA Daily Fantasy contests have risen in popularity over the past few years, as a minimal commitment alternative to season long fantasy leagues. A participants objective is to create a synthetic fantasy team that will score the most "fantasy points" for any given set of games. Each individual player has a "salary" and "position", allowing for both salary and positional constraints for a full team.

Individual NBA players accrue "fantasy points" by accruing actual statistics in games (points, rebound, assist, blocks, etc..) at predefined multiple (i.e 1 assist = 1.5 fantasy points, 1 block = 3 fantasy points). Salaries are positively correlated to total fantasy points and adjusted nightly depending on previous performance and opposing team (proprietary formula from Daily Fantasy providor (Fanduel, Draft Kings)

My objective is to predict a players daily fantasy points in order to take advantage of player mispricings (i.e choose to play players who will score more than their salary suggests, or vice verse)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
--------------

**Please see the "NBA Daily Fantasy Value" notebook for project visualizations and findings**

**All source code found under "src" folder**
