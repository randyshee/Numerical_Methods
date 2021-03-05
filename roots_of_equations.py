import numpy as np

def simple_iteration(a, b, c, initial_guess=0, max_iter=100, decimals=6):
	"""
	Solve quadratic equations in the form of ax^2 + bx + c = 0 
	iteratively with for loop that has an `initial_guess` and 
	ends at `max_iter` iterations or a threshold given by `decimals`

	Args:
	    a (float) : coefficient of x^2
	    b (float) : coefficient of x
	    c (float) : constant coefficient
	    initial_guess (float, optional) : initial guess of the root
	    max_iter (int, optional) : the maximum iteration of the for loop
	    decimals (int, optional) : the desired decimal accuracy 

	Returns:
	    (int, float) : the first number is the iteration used
	                   and the second number is the numerically
	                   sovled root rounded at the given `decimals`
	"""
	x = initial_guess
	threshold = 10**(-decimals)
	for iteration in range(1, max_iter):
		xnew = (a*x**2 + c)/(-b)
		print(iteration, x)
		if abs(xnew - x) < threshold:
			break
		x = xnew
	return iteration, np.around(xnew, decimals-1)

def newton_raphson(a, b, c, initial_guess=0, max_iter=100, decimals=6):
	"""
	Solve quadratic equations in the form of ax^2 + bx + c = 0 
	using the Newton-Raphson's method that has an `initial_guess` and 
	ends at `max_iter` iterations or a threshold given by `decimals`

	Args:
	    a (float) : coefficient of x^2
	    b (float) : coefficient of x
	    c (float) : constant coefficient
	    initial_guess (float, optional) : initial guess of the root
	    max_iter (int, optional) : the maximum iteration of the for loop
	    decimals (int, optional) : the desired decimal accuracy 

	Returns:
	    (int, float) : the first number is the iteration used
	                   and the second number is the numerically
	                   sovled root rounded at the given `decimals`
	"""
	x = initial_guess
	threshold = 10**(-decimals)
	for iteration in range(1, max_iter):
		# The Newton-Raphson's formula
		# xnew = x - f(x)/f'(x)
		xnew = x - (a*x**2 + b*x + c)/(2*a*x + b)
		print(iteration, x)
		if abs(xnew - x) < threshold:
			break
		x = xnew
	return iteration, np.around(xnew, decimals-1) 

# An example of what fn should look like
def fn(x): 
	# analytical roots are 1 and 1.5
	return 2*x**2 - 5*x + 3

def bisection_method(fn, x1, x2, max_iter=100, decimals=6):
	"""
	Find roots of equations in the form of fn(x) = 0 using the 
	bisection method where `x1` and `x2` are the lower and upper 
	limit of the expected interval. The process ends at `max_iter` 
	iterations or a threshold given by `decimals`

	Args:
	    fn : a user defined or lambda function that returns a float
	    x1 (float) : lower limit of the expected interval
	    x2 (float) : upper limit of the expected interval
	    max_iter (int, optional) : the maximum iteration of the for loop
	    decimals (int, optional) : the desired decimal accuracy 

	Returns:
	    (int, float) : the first number is the number of bisection 
	                   performed and the second number is the numerically
	                   sovled root rounded at the given `decimals`
	"""

	threshold = 10**(-decimals)
	y1, y2 = fn(x1), fn(x2)
	if y1 == 0:
		return 0, x1
	if y2 == 0:
		return 0, x2
	assert y1*y2 < 0, "The signs of the two y values should be opposite"
	for bisection in range(1, max_iter):
		xh = (x1 + x2)/2
		yh = fn(xh)
		y1 = fn(x1)
		if abs(y1) < threshold:
			break
		elif y1*yh < 0:
			x2 = xh
		else:
			x1 = xh
	return bisection, np.around(x1, decimals-1)

def regula_falsi(fn, x1, x2, max_iter=100, decimals=6):
	"""
	Find roots of equations in the form of fn(x) = 0 using the 
	regula falsi method where `x1` and `x2` are the lower and 
	upper limit of the expected interval. The process ends at 
	`max_iter` iterations or a threshold given by `decimals`

	Args:
	    fn : a user defined or lambda function that returns a float
	    x1 (float) : lower limit of the expected interval
	    x2 (float) : upper limit of the expected interval
	    max_iter (int, optional) : the maximum iteration of the for loop
	    decimals (int, optional) : the desired decimal accuracy 

	Returns:
	    (int, float) : the first number is the number of iteration used 
	                   and the second number is the numerically sovled 
	                   root rounded at the given `decimals`
	"""
	threshold = 10**(-decimals)
	y1, y2 = fn(x1), fn(x2)
	if abs(y1) < threshold:
		return 0, x1
	if abs(y2) < threshold:
		return 0, x2
	assert y1*y2 < 0, "The signs of the two y values should be opposite"
	for iteration in range(1, max_iter):
		xh = x2 - (x2 - x1)/(y2 - y1) * y2
		yh = fn(xh)
		if abs(yh) < threshold:
			break
		elif y1*yh < 0:
			x2 = xh
			y2 = yh
		else:
			x1 = xh
			y1 = yh
	return iteration, np.around(xh, decimals-1)


def secant_method(fn, x1, x2, max_iter=100, decimals=6):
	"""
	Find roots of equations in the form of fn(x) = 0 
	using the secant method where `x1` and `x2` are 
	the merely initial values where roots are not 
	necessary to be in this interval. However, they 
	should not be the same (x1 cannot equal x2), and
	fn(x1) also shouldn't be the same as fn(x1).

	Args:
	    fn : a user defined or lambda function that returns a float
	    x1 (float) : lower limit of the expected interval
	    x2 (float) : upper limit of the expected interval
	    max_iter (int, optional) : the maximum iteration of the for loop
	    decimals (int, optional) : the desired decimal accuracy 

	Returns:
	    (int, float) : the first number is the number of iteration used 
	                   and the second number is the numerically sovled 
	                   root rounded at the given `decimals`
	"""
	threshold = 10**(-decimals)
	if abs(fn(x1)) < threshold:
		return 0, x1
	if abs(fn(x2)) < threshold:
		return 0, x2
	for iteration in range(1, max_iter):
		xnew = x2 - (x2 - x1)/(fn(x2) - fn(x1)) * fn(x2)
		if abs(fn(xnew)) < threshold:
			break
		else:
			x1, x2 = x2, xnew
	return iteration, np.around(xnew, decimals-1)


# some root-finding functions in scipy package 
#from scipy.optimize import newton, bisect, fsolve, root
#newton(fn, 0)
#bisect(fn, 0, 1.2)
#fsolve(fn, 0)
#fsolve(fn, [-1, 0, 1, 2, 3])
#root(fn, 0)
#root(fn, [-1, 0, 1, 2, 3])











