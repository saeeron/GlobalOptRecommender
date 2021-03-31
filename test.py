import numpy as np 
from numpy import linalg 
from scipy import optimize, stats
import matplotlib.pyplot as plt
from functools import partial
import seaborn as sn 
import pandas as pd
from pandas import DataFrame as DF
from matplotlib import colors, cm
import zipfile
import io


from utils import subset_sparse_mat, test_train


ml_tmp = zipfile.ZipFile('ml-100k.zip', mode='r')
ml_df = pd.read_csv(io.BytesIO(ml_tmp.read('ml-100k/u.data')),sep = '\t', names= ['user id' , 'item id' , 'rating' , 'timestamp'])
ml_df['timestamp'] = pd.to_datetime(ml_df['timestamp'], unit='s', origin='unix')
print(ml_df.head(10))


# pivoting the datafarme ans extracting the values
ml_df = ml_df[['user id','item id', 'rating']].pivot(index = "user id", columns = "item id", values = "rating")

sparsity_ = (ml_df.values.size - np.isnan(ml_df.values).sum()) / ml_df.values.size
print("Utility matrix is {:0.02f}% sparse".format(sparsity_ * 100))

rank_of_A  = 2
print("total number of parameters : {:d}".format(int(rank_of_A * ml_df.shape[0] + rank_of_A * ml_df.shape[1])))
print("total number of observations : {:d}".format(int((ml_df.values.size - np.isnan(ml_df.values).sum()))))

# subsetting the data
ml_df = subset_sparse_mat(ml_df, 0.2)

AA, AAtest = test_train(ml_df.values)


def objF(X_ , A_ , rank_of_A, reg_lam ):
   
  A__ = A_.copy()
  X_l = np.reshape(X_.copy().flat[:rank_of_A * A__.shape[0]], (-1,rank_of_A))
  X_r = np.reshape(X_.copy().flat[rank_of_A * A__.shape[0]:], (rank_of_A,-1))
  XX = X_l @ X_r 
  I_rmn = np.where(~np.isnan(A__.flat))[0]  # finding where is notnan in the incomplete matrix
  residue = ((XX.flat[I_rmn] - A__.flat[I_rmn])**2).sum()
  regul   = reg_lam * np.linalg.norm(X_)
  return residue + regul

# we need to freeze A_ to work on X_ 

objF_ = partial(objF, A_ = AA / 5, rank_of_A = rank_of_A, reg_lam  = 10)

# optimization constraints 

lw = np.zeros(rank_of_A * AA.shape[0] + rank_of_A * AA.shape[1])
up = 1 * np.ones(rank_of_A * AA.shape[0] + rank_of_A * AA.shape[1])

# running optimization 
if __name__ == "__main__":
	
	ret = optimize.dual_annealing(objF_, bounds=list(zip(lw, up)), maxiter=1000, seed = 1414) 


def ret_factors(x, A__, rank_of_A):

	X_l = np.reshape(x.copy().flat[:rank_of_A * A__.shape[0]], (-1,rank_of_A))
	X_r = np.reshape(x.copy().flat[rank_of_A * A__.shape[0]:], (rank_of_A,-1))
	return X_l @ X_r * 5 # multiplying by 5 to scale it from 1 to 5 stars

A_pred = ret_factors(ret.x, AA, rank_of_A)   


I_train = np.where(~np.isnan(AA.flat[:]))[0]
print("RMSE on train data : {:0.02f}".format(np.sqrt(((AA.flat[I_train] - A_pred.flat[I_train])**2).mean())))


I_test = np.where(~np.isnan(AAtest.flat[:]))[0]
print("RMSE on test data : {:0.02f}".format(np.sqrt(((AAtest.flat[I_test] - A_pred.flat[I_test])**2).mean())))

