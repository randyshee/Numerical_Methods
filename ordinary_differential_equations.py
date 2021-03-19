import numpy as np

def euler_method(dy, domain, y, h=0.5):
	"""
	Find the numerical solution of first order differential equation `dy` 
	over the `domain` with given stepsize `h` and initial value `y` using
	the Euler's method (using 1st derivative of Taylor series).

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

def second_runge_kutta(dy, domain, y, h=0.5):
	"""
	Find the numerical solution of first order differential equation `dy` 
	over the `domain` with given stepsize `h` and initial value `y` using
	the second-order Runge-Kutta's method (including 2nd derivative of 
	Taylor series).

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
	>>> data_points = second_runge_kutta(dy, domain, y)
	>>> data_points = ['(%.3f, %.3f)' % (x, y) for (x, y) in data_points]
	>>> data_points = [eval(item) for item in data_points]
	>>> data_points
	[(0.0, 1.0), (0.5, 1.125), (1.0, 1.6), (1.5, 2.849), (2.0, 6.277)]
	"""
	x, x_n = domain[0], domain[1]
	n = int((x_n - x)/h)
	data_points = [(x, y)]
	for i in range(n):
		K1 = h*dy(x, y)
		K2 = h*dy(x + h/2, y + K1/2)
		y += K2
		x += h
		data_points.append((x, y))
	return data_points

def forth_runge_kutta(dy, domain, y, h=0.5):
	"""
	Find the numerical solution of first order differential equation `dy` 
	over the `domain` with given stepsize `h` and initial value `y` using
	the forth-order Runge-Kutta's method (including 4th derivative of 
	Taylor series).

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
	>>> data_points = forth_runge_kutta(dy, domain, y)
	>>> data_points = ['(%.3f, %.3f)' % (x, y) for (x, y) in data_points]
	>>> data_points = [eval(item) for item in data_points]
	>>> data_points
	[(0.0, 1.0), (0.5, 1.133), (1.0, 1.649), (1.5, 3.078), (2.0, 7.367)]
	"""
	x, x_n = domain[0], domain[1]
	n = int((x_n - x)/h)
	data_points = [(x, y)]
	for i in range(n):
		K1 = h*dy(x, y)
		K2 = h*dy(x + h/2, y + K1/2)
		K3 = h*dy(x + h/2, y + K2/2)
		K4 = h*dy(x + h, y + K3)
		y += (K1 + 2*K2 + 2*K3 + K4)/6
		x += h
		data_points.append((x, y))
	return data_points

# some ODE solution functions in scipy package 
#from scipy.integrate import odeint

#y = odeint(dy, y0, domain_x)
