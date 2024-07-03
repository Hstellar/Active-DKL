## Using Actively-learned Deep kernels to estimate epistemic uncertainty

#### Approach
At the query stage in Figure 1(a), candidates are chosen from the pool of unlabeled samples based on pairwise distance and the variance reduction method. The chosen samples are checked in the lab, and the results are added to the training data. This makes it easier to draw conclusions about the unlabeled samples in the pool. A variance reduction approach is used, i.e., selecting the most uncertain sample by computing the similarity score using the pairwise distance between the training batch and the unlabled batch. In this workflow, the base model is Deep kernels shown in Figure 1(b), which combine a feature extractor and a gaussian process to select the most useful samples from a pool of unlabeled samples. In the active learning workflow, at the query stage in Figure 2, candidates are chosen from the pool of unlabeled samples based on pairwise distance and the variance reduction method. For active learning, 20% of the data were used as test data and then 80% of the remaining data were used as training data. The initial training set is 20% which is labeled data. Pairwise distance was used batch-wise to select query points from remaining 60% of data (train hold or unlabeled samples)(Mamun et al., 2022):
<img width="936" alt="image" src="https://github.com/Hstellar/Active-DKL/assets/22677436/1342c6e7-6ed0-4473-a4fa-e06ff5eece22">

#### Results
We compare error metrics such as Mean absolute error(MAE), Root Mean Square error(RMSE), Symmetric Mean absolute percentage error(sMAPE) and Correlation metric(R2 score) for Monte carlo dropout denoted as MC-dropout, Deep kernel learning trained without active learning strategy denoted as DKL and with Active learning which is denoted as Active-DKL in table 1. 
<img width="401" alt="image" src="https://github.com/Hstellar/Active-DKL/assets/22677436/2e613b95-4039-48b1-821e-706719f72786">

From table 1, Active learning with Deep kernel performs better for Shift dataset compared to Air Quality where Monte carlo dropout gives comparable performance. Though we haven’t made enough comparison between more number of dataset but Active-DKL successfully captures distribution shift in Shift Dataset.

### Code base structure:<br> 
File description for `Active_learning` and `DKL` folder:<br>
`features.py` contains function for feature engineering <br>
`run_exp.py` contains code for training model, validation and testing <br>
`model.py` contains code for defining model architecture <br>
 Run `.ipynb` files to reproduce the code.<br>
To install dependencies run `pip install -r requirements.txt` file to create environment. <br>
Python 3.8.10, pytorch 2.0.0+cu118, recent version gpytorch is used. <br> <br>

MC_dropout folder contains following files:<br>
`features.py` contains function for feature engineering <br>
`mc_dropout_air_quality.ipynb` contains training and inference code for air quality dataset <>br
`mc_dropout.ipynb` contains training and inference code for Shift dataset

Data used is from [Grand shift challenge](https://shifts.grand-challenge.org/). Data can be downloaded from [here](https://zenodo.org/record/7684813). Other dataset used was Air Quality from [UCI machine learning repository](https://archive.ics.uci.edu/dataset/360/air+quality)

#### Citations

```
@article{mamun2022uncertainty,
  title={Uncertainty quantification for Bayesian active learning in rupture life prediction of ferritic steels},
  author={Mamun, Osman and Taufique, MFN and Wenzlick, Madison and Hawk, Jeffrey and Devanathan, Ram},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={2083},
  year={2022},
  publisher={Nature Publishing Group UK London}}
```
```
@inproceedings{wilson2016deep,
  title={Deep kernel learning},
  author={Wilson, Andrew Gordon and Hu, Zhiting and Salakhutdinov, Ruslan and Xing, Eric P},
  booktitle={Artificial intelligence and statistics},
  pages={370--378},
  year={2016},
  organization={PMLR}
}
```



