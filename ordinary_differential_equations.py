import numpy as np

def euler_method(dy, domain, y, h=0.5):
	"""
	Find the numerical solution of differential equation `dy` over
	the `domain` with given stepsize `h` and initial value `y` using
	the Euler's method (Taylor series).

	Args:
	    dy : a differential equation (user defined or lambda function)
	    domain ([float, float]) : lower and upper limit of x values
	    y (float) : the initial y value (y value when x is at the lower
	                limit of the domain)
	    h (float) : stepsize

	Returns:
	    [(float, float)] : the data points (x, y) in the domain

	>>> dy = lambda x, y: x*y
	>>> domain = [0., 2.]
	>>> y = 1.
	>>> euler_method(dy, domain, y)
	[(0.0, 1.0), (0.5, 1.0), (1.0, 1.25), (1.5, 1.875), (2.0, 3.28125)]
	"""
	x, x_n = domain[0], domain[1]
	n = int((x_n - x)/h)
	data_points = [(x, y)]
	for i in range(n):
		y += dy(x, y)*h
		x += h
		data_points.append((x, y))
	return data_points