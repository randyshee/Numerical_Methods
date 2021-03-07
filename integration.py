import numpy as np

def trapezoidal_rule(fn, range_x, n=5):
	"""
	An integration method that calculate the area under a curve 
	`fn` that is divided into  vertical taperzoids with `n` 
	divisions  within a range of x values `range_x`

	Args:
	    fn : a user defined or lambda function that returns a float
	    range_x ([float, float]) : lower and upper limit of x values
	    n (int) : the number of divisions

	Returns:
	    float : the approximated integrated value at a given range

	>>> fn = lambda x: x*np.sin(x)
	>>> range_x = [0, np.pi/2] # analytical result = 1
	>>> np.around(trapezoidal_rule(fn, range_x, n=5), 6)
	1.008265
	>>> np.around(trapezoidal_rule(fn, range_x, n=10), 6)
	1.002059
	>>> np.around(trapezoidal_rule(fn, range_x, n=100), 6)
	1.000021
	"""
	# l and r denote left (lower) and right (upper) limits
	l, r = range_x[0], range_x[1]
	h = (r - l)/n
	return 0.5*h*(fn(l + r)) + sum([h*fn(l + i*h) for i in range(1, n)])

def simpson_1_3_rule(fn, range_x, n=6):
	"""
	This functino uses Simpson's 1/3 rule to calculate the area 
	under a curve `fn` with `n` divisions  within a range of x 
	values `range_x`. The Simpson's 1/3 rule uses weighing factors 
	to  minimize the error from  trapezoidal method with the formula
	for the first two strips being A=h*[f(x0)+4*f(x1)+f(x2)]/3.

	Args:
	    fn : a user defined or lambda function that returns a float
	    range_x ([float, float]) : lower and upper limit of x values
	    n (int) : the number of divisions (has to be even)

	Returns:
	    float : the approximated integrated value at a given range

	>>> fn = lambda x: x*np.sin(x)
	>>> range_x = [0, np.pi/2] # analytical result = 1
	>>> np.around(simpson_1_3_rule(fn, range_x, n=6), 6)
	0.999921
	>>> np.around(simpson_1_3_rule(fn, range_x, n=10), 6)
	0.99999
	>>> np.around(simpson_1_3_rule(fn, range_x, n=100), 6)
	1.0
	"""
	assert n%2 == 0, 'The number of divisions has to be even!'
	l, r = range_x[0], range_x[1]
	h = (r - l)/n
	return h/3*(fn(l) + fn(r) + \
		   4*sum([fn(l + i*h) for i in range(1, n, 2)]) + \
		   2*sum([fn(l + i*h) for i in range(2, n, 2)]))

def simpson_3_8_rule(fn, range_x, n=6):
	"""
	This functino uses Simpson's 3/8 rule to calculate the area 
	under a curve `fn` with `n` divisions  within a range of x 
	values `range_x`. The Simpson's 3/8 rule uses weighing factors 
	to  minimize the error from  trapezoidal method with the formula
	for the first three strips being A=3*h*[f(x0)+3*f(x1)+3*f(x2)+f(x3)]/8.

	Args:
	    fn : a user defined or lambda function that returns a float
	    range_x ([float, float]) : lower and upper limit of x values
	    n (int) : the number of divisions (has to multiple of three)

	Returns:
	    float : the approximated integrated value at a given range

	>>> fn = lambda x: x*np.sin(x)
	>>> range_x = [0, np.pi/2] # analytical result = 1
	>>> np.around(simpson_3_8_rule(fn, range_x, n=6), 6)
	0.999819
	>>> np.around(simpson_3_8_rule(fn, range_x, n=9), 6)
	0.999965
	>>> np.around(simpson_3_8_rule(fn, range_x, n=99), 6)
	1.0
	"""
	assert n%3 == 0, 'The number of divisions has to be multiple of 3!'
	l, r = range_x[0], range_x[1]
	h = (r - l)/n
	return 3*h/8*(fn(l) + fn(r) + \
		   3*sum([fn(l + i*h) + fn(l + (i + 1)*h) for i in range(1, n, 3)]) \
		   + 2*sum([fn(l + i*h) for i in range(3, n, 3)]))

def double_integration():
	pass