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

ml_tmp = zipfile.ZipFile('ml-100k.zip', mode='r')
ml_df = pd.read_csv(io.BytesIO(ml_tmp.read('ml-100k/u.data')),sep = '\t', names= ['user id' , 'item id' , 'rating' , 'timestamp'])
ml_df['timestamp'] = pd.to_datetime(ml_df['timestamp'], unit='s', origin='unix')
print(ml_df.head(10))


# pivoting the datafarme ans extracting the values
A = ml_df[['user id','item id', 'rating']].pivot(index = "user id", columns = "item id", values = "rating").values

sparsity_ = (A.size - np.isnan(A).sum()) / A.size
print("Utility matrix is {:0.02f} dense".format(sparsity_ * 100))

rank_of_A  = 2
print("total number of parameters : {:d}".format(int(rank_of_A * A.shape[0] + rank_of_A * A.shape[1])))
print("total number of observations : {:d}".format(int((A.size - np.isnan(A).sum()))))


def objF(X_ , A_ , rank_of_A):
  """ X_ is a vector fucntion that builds two X_l and X_r marices """
  A__ = A_.copy()
  X_l = np.reshape(X_.copy().flat[:rank_of_A * A__.shape[0]], (-1,rank_of_A))
  X_r = np.reshape(X_.copy().flat[rank_of_A * A__.shape[0]:], (rank_of_A,-1))
  XX = X_l @ X_r 
  I_rmn = np.where(~np.isnan(A__.flat))[0]  # finding where is notnan in the incomplete matrix
  return np.sqrt(((XX.flat[I_rmn] - A__.flat[I_rmn])**2).mean())

# we need to freeze A_ to work on X_ 

objF_ = partial(objF, A_ = A / 5, rank_of_A = rank_of_A)

# optimization constraints 

lw = np.zeros(rank_of_A * A.shape[0] + rank_of_A * A.shape[1])
up = np.ones(rank_of_A * A.shape[0] + rank_of_A * A.shape[1])

# running optimization 
if __name__ == "__main__":
	#ret = optimize.differential_evolution(objF_, bounds=list(zip(lw, up)), workers= 4, updating="deferred", maxiter=1000, disp = True) 
	ret = optimize.dual_annealing(objF_, bounds=list(zip(lw, up)), maxiter=100, seed = 1414) 