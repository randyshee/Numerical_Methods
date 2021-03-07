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

def double_integration(fn, range_x, range_y, n_x=10, n_y=10):
	"""
	This functino uses Simpson's 1/3 rule to calculate the double integral
	of `fn` with `n_x` divisions within a range of x values `range_x` and 
	`n_y` divisions within a range of y values `range_y`.

	Args:
	    fn : a user defined or lambda function that returns a float
	    range_x ([float, float]) : lower and upper limit of x values
	    range_y ([float, float]) : lower and upper limit of y values
	    n_x (int) : the number of divisions for x (has to be even)
	    n_y (int) : the number of divisions for y (has to be even)

	Returns:
	    float : the approximated integrated value at given x and y ranges

	>>> fn = lambda x, y: x**2*y + x*y**2
	>>> range_x = [1, 2]
	>>> range_y = [-1, 1] # analytical result = 1
	>>> np.around(double_integration(fn, range_x, range_y), 6)
	1.0
	"""
	# l and r denote left (lower) and right (upper) limits of x or y
	l_x, r_x, l_y, r_y = range_x[0], range_x[1], range_y[0], range_y[1]
	h_x = (r_x - l_x)/n_x
	h_y = (r_y - l_y)/n_y
	S = 0
	for i in range(n_y+1):
		if i == 0 or i == n_y:
			p = 1
		elif i%2 == 1:
			p = 4
		else:
			p = 2
		for j in range(n_x+1):
			if j == 0 or j == n_x:
				q = 1
			elif j%2 == 1:
				q = 4
			else:
				q = 2
			S += p*q*fn(l_x + j*h_x, l_y + i*h_y)
	return h_x*h_y*S/9

# some integration functions in scipy package 
#from scipy.integrate import quad, dblquad, nquad

#fn = lambda x: 0.1*x**5 - 0.2*x**3 + 0.1*x -0.2
#integral, error = quad(fn, a, b)
#integral, error = dblquad(fn, ax, bx, lambda y: ay, lambda y: by)
#integral, error = nquad(fn, [[a1, b1], [a2, b2],..., [an, bn]])
