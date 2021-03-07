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
	>>> np.around(trapezoidal_rule(fn, range_x), 6)
	1.008265
	>>> np.around(trapezoidal_rule(fn, range_x, n=10), 6)
	1.002059
	>>> np.around(trapezoidal_rule(fn, range_x, n=100), 6)
	1.000021
	"""
	# l and r denote left (lower) and right (upper) limits
	l, r = range_x[0], range_x[1]
	h = (r - l)/n
	return 0.5*h*(fn(l+r)) + sum([h*fn(l+i*h) for i in range(1, n)])

def simpson_1_3_rule():
	pass

def simpson_1_8_rule():
	pass