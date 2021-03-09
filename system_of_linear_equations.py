import numpy as np 

def gaussian_elimination(A, b):
	"""
	Solve a system of linear equations by Gaussian elemination 
	given in the form of Ax = b where `A` matrix and `b` vecotr form 
	an augmented matrix.

	Args:
	    A (ndarray) : 2D array containing the coefficients of each 
	                  variable in each linear equation
	    b (ndarray) : 1D array containing the constant terms

	Returns:
	    ndarray : 1D array containing the solution to each variable

	>>> A = np.array([[2, 7, -1, 3, 1], \
	                  [2, 3, 4, 1, 7], \
	                  [6, 2, -3, 2, -1], \
	                  [2, 1, 2, -1, 2], \
	                  [3, 4, 1, -2, 1]], float)
    >>> b = np.array([5, 7, 2, 3, 4], float)
    >>> np.around(gaussian_elimination(A, b), 6)
    array([0.444444, 0.555556, 0.666667, 0.222222, 0.222222])
	"""
	# Copy the arrays because we don't want to change the original arrays
	A = np.copy(A)
	b = np.copy(b)
	n = len(b)
	x = np.zeros(n, float)
	# Elimination step 
	for k in range(n-1):
		for row in range(k+1, n):
			factor = A[k, k]/A[row, k]
			# Constant terms also have to go through the elimination process
			b[row] = b[k] - factor*b[row] 
			for col in range(k, n):
				A[row, col] = A[k, col] - factor*A[row, col]
	# Back Substitution step
	x[n-1] = b[n-1]/A[n-1, n-1]
	for row in range(n-1, -1, -1):
		x[row] = (b[row] - sum([A[row, col]*x[col] for col in range(row+1, n)]))/A[row, row]
	return x
