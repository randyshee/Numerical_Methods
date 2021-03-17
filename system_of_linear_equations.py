import numpy as np 

def gaussian_elimination(A, b):
	"""
	Solve a system of linear equations by Gaussian elemination 
	given the form of Ax = b where `A` matrix and `b` vecotr form 
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
	>>> A = np.array([[0, 7, -1, 3, 1], \
	                  [2, 3, 4, 1, 7], \
	                  [6, 2, 0, 2, -1], \
	                  [2, 1, 2, 0, 2], \
	                  [3, 4, 1, -2, 1]], float)
	>>> b = np.array([5, 7, 2, 3, 4], float)
	>>> np.around(gaussian_elimination(A, b), 6)
	array([0.021705, 0.792248, 1.051163, 0.15814 , 0.031008])
	"""
	# Copy the arrays because we don't want to change the original arrays
	A = np.copy(A)
	b = np.copy(b)
	n = len(b)
	x = np.zeros(n, float)
	# Elimination step 
	for k in range(n-1):
		# Swap two rows if the diagonal term is zero
		if A[k, k] == 0:
			for col in range(n):
				A[k, col], A[k+1, col] = A[k+1, col], A[k, col]
			b[k], b[k+1] = b[k+1], b[k]
		for row in range(k+1, n):
			# This will avoid division by 0
			if A[row, k] == 0:
				continue
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

def jacobi_method(A, b, initial_guess, max_iter=100, decimals=6):
	"""
	Solve a system of linear equations by simple iteration method 
	after rearrangement known as the Jacobi's method given the 
	form of Ax = b

	Args:
	    A (ndarray) : 2D array containing the coefficients of each 
	                  variable in each linear equation
	    b (ndarray) : 1D array containing the constant terms
	    initial_guess (ndarray) : 1D array containing the guess of solution
	    max_iter (int) : the maximum iteration
	    decimals (int) : The desired decimal accuracy

	Returns:
	    ndarray : 1D array containing the solution to each variable

	>>> A = np.array([[4, 1, 2, -1], \
	                  [3, 6, -1, 2], \
	                  [2, -1, 5, -3], \
	                  [4, 1, -3, -8]], float)
	>>> b = np.array([2, -1, 3, 2], float)
	>>> initial_guess = np.full(len(b), 1., float)
	>>> jacobi_method(A, b, initial_guess)[1]
	array([ 0.36501, -0.23379,  0.28507, -0.20362])
	"""
	n = len(b)
	x = initial_guess
	xnew = np.empty(n, float)
	threshold = 10**(-decimals)
	for iteration in range(max_iter):
		for i in range(n):
			xnew[i] = -(sum([A[i, j]*x[j] for j in range(n) if j != i]) - b[i])/A[i, i]
		if all(abs(xnew - x) < threshold):
			break
		else:
			x = np.copy(xnew)
	return iteration, np.around(xnew, decimals-1)

def gauss_seidel_method(A, b, initial_guess, max_iter=100, decimals=6):
	"""
	Solve a system of linear equations given the form of Ax = b 
	by simple iteration method similar to Jacobi's method but 
	new values of `xnew` (solution guess) are applied within
	the same iteration. This method is known as the Gauss-Seidel's
	method.

	Args:
	    A (ndarray) : 2D array containing the coefficients of each 
	                  variable in each linear equation
	    b (ndarray) : 1D array containing the constant terms
	    initial_guess (ndarray) : 1D array containing the guess of solution
	    max_iter (int) : the maximum iteration
	    decimals (int) : The desired decimal accuracy

	Returns:
	    ndarray : 1D array containing the solution to each variable

	>>> A = np.array([[4, 1, 2, -1], \
	                  [3, 6, -1, 2], \
	                  [2, -1, 5, -3], \
	                  [4, 1, -3, -8]], float)
	>>> b = np.array([2, -1, 3, 2], float)
	>>> initial_guess = np.full(len(b), 1., float)
	>>> gauss_seidel_method(A, b, initial_guess)[1]
	array([ 0.36501, -0.23379,  0.28507, -0.20362])
	"""
	n = len(b)
	x = initial_guess
	xnew = np.empty(n, float)
	threshold = 10**(-decimals)
	for iteration in range(max_iter):
		for i in range(n):
			xnew[i] = -(sum([A[i, j]*x[j] for j in range(n) if j != i]) - b[i])/A[i, i]
			xdiff = abs(xnew[i] - x[i])
			x[i] = xnew[i]
		if xdiff < threshold:
			break
	return iteration, np.around(xnew, decimals-1)