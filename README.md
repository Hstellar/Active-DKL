# Epistemic_uncertainty

Active_learning and DKL (Deep Kernel learning)folder contains following structure for code base:<br> <br>
File description:
features.py contains function for feature engineering <br>
run_exp.py contains code for training model, validation and testing <br>
model.py contains code for defining model architecture <br>
 Run .ipynb files to reproduce the code. Code will be open sourced once acceptance deadline is passed.<br>
To install dependencies run requirements.txt file to create environment. <br>
Python 3.8.10, pytorch 2.0.0+cu118, recent version gpytorch is used. <br> <br>

MC_dropout folder contains following files:<br>
features.py contains function for feature engineering <br>
mc_dropout_air_quality.ipynb contains training and inference code for air quality dataset <>br
mc_dropout.ipynb contains training and inference code for Shift dataset

Data used is from [Grand shift challenge](https://shifts.grand-challenge.org/). Data can be downloaded from [here](https://zenodo.org/record/7684813). Other dataset used was Air Quality from [UCI machine learning repository](https://archive.ics.uci.edu/dataset/360/air+quality)
