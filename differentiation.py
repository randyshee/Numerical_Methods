import numpy as np
from math import factorial

def n_choose_i(n, i):
	"""
	A math tool that returns the value of `n` choose `i`.
	"""
	return factorial(n)/(factorial(n-i)*factorial(i))

def forward_finite_differences(fn, x, n=1, h=0.05):
	"""
	This function finds the `n`-th derivative of a polynomial
	`fn` at a given value `x` by a differentiation method 
	based  on Taylor expansion  that consider the forward 
	finite difference. The expected error is O(h) where `h` 
	is the step size.

	Args:
	    fn : a user defined or lambda function that returns a float
	    x (float) : the x value (position) to differentiate
	    n (int) : the degree of derivative
	    h (float) : step size

	Returns:
	    float : the derivative at a given point

	>>> fn = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x -0.2
	>>> np.around(forward_finite_differences(fn, 0.1), 6)
	0.090632
	>>> np.around(forward_finite_differences(fn, 0.1, n=2), 6)
	-0.172875
	"""
	return sum([(-1)**(n-i)*n_choose_i(n, i)*fn(x+i*h)/h**n for i in range(n+1)])

def backward_finite_differences(fn, x, n=1, h=0.05):
	"""
	This function finds the `n`-th derivative of a polynomial
	`fn` at a given value `x` by a differentiation method 
	based  on Taylor expansion  that consider the backward 
	finite difference. The expected error is O(h) where `h` 
	is the step size.

	Args:
	    fn : a user defined or lambda function that returns a float
	    x (float) : the x value (position) to differentiate
	    n (int) : the degree of derivative
	    h (float) : step size

	Returns:
	    float : the derivative at a given point

	>>> fn = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x -0.2
	>>> np.around(backward_finite_differences(fn, 0.1), 6)
	0.096519
	>>> np.around(backward_finite_differences(fn, 0.1, n=2), 6)
	-0.059625
	"""
	return sum([(-1)**i*n_choose_i(n, i)*fn(x-i*h)/h**n for i in range(n+1)])

def central_finite_differences(fn, x, n=1, h=0.05):
	"""
	This function finds the `n`-th derivative of a polynomial
	`fn` at a given value `x` by a differentiation method 
	based  on Taylor expansion  that consider the central 
	finite difference. The expected error is O(h^2) where `h` 
	is the step size.

	Args:
	    fn : a user defined or lambda function that returns a float
	    x (float) : the x value (position) to differentiate
	    n (int) : the degree of derivative
	    h (float) : step size

	Returns:
	    float : the derivative at a given point

	>>> fn = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x -0.2
	>>> np.around(central_finite_differences(fn, 0.1), 6)
	0.093931
	>>> np.around(central_finite_differences(fn, 0.1, n=2), 6)
	-0.11775
	"""
	return sum([(-1)**i*n_choose_i(n, i)*fn(x+(n/2-i)*h)/h**n for i in range(n+1)])

# some differentiation functions in scipy package 
# from scipy.misc import derivative

#fn = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x -0.2
#computes the n-th derivative of a given function fn at a given
#point x0 using a central difference formula with spacing dx
#yn = derivative(fn, x0, dx, n)
#y2 = derivative(fn, 0.1, 0.01, 2)
