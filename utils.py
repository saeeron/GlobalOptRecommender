import numpy as np 
import pandas as pd 


def subset_sparse_mat(df, fract):
	""" df is a pivoted data-frame and fract is the fraction [0,1] to be extracted 
	it returns  """
	A = df.values 
	ii = np.random.choice(np.arange(0, A.shape[0]), size = int(fract * A.shape[0]),replace=False)
	jj = np.random.choice(np.arange(0, A.shape[1]), size = int(fract * A.shape[1]),replace=False)
	ind_ = df.index[ii] 
	col_ = df.columns[jj] 
	AA = A[ii][:,jj].copy()
	i_rem = np.where(np.isnan(AA).all(axis=1))
	j_rem = np.where(np.isnan(AA).all(axis=0))
	AA = np.delete(AA, i_rem, 0)
	AA = np.delete(AA, j_rem, 1)
	ind_ = np.delete(ind_, i_rem)
	col_ = np.delete(col_, j_rem)

	return pd.DataFrame(AA, columns = col_, index = ind_)

	
def test_train(A, test_frac = 0.15):
	
	I_obs = np.where(~np.isnan(A.flat[:]))[0]
	I_obs_test = np.random.choice(I_obs, size = int(test_frac * I_obs.size))
	I_obs_train = np.setdiff1d(I_obs, I_obs_test)
	A_test = A.copy()
	A_train = A.copy()
	A_test.flat[I_obs_train] = np.nan
	A_train.flat[I_obs_test] = np.nan 

	return A_train, A_test

