import numpy as np

def linear_interpolation(xp, xlist, ylist):
	"""
	Predict the y value of the given `xp` by 
	linear interpolation between the closest 
	intervel in `xlist`

	>>> time = [0, 20, 40, 60, 80, 100]
	>>> temp = [26.0, 48.6, 61.6, 71.2, 74.8, 75.2]
	>>> linear_interpolation(50, time, temp)
	66.4
	"""
	def prediction(xp, x1, x2, y1, y2):
		return y1 + (y2 - y1)/(x2 - x1)*(xp - x1)

	for i, xi in enumerate(xlist):
		if i != 0 and xp < xi:
			return prediction(xp, xlist[i-1], xlist[i], ylist[i-1], ylist[i])
	print("Given `xp` is out of the range of `xlist`")

def lagrange_method(xp, xlist, ylist):
	"""
	Note: 
	Lagrange's method predict the y value of a given `xp` 
	by  creating a interpolation polynomial of  degree n 
	and it'll need n+1 points.

	Interpolation polynomial: 
	y(x) = sum(y_i*l_i(x)) for i from 1 to n+1
	l_i(x) = prod((x-x_j)/(x_i-x_j)) for j from 1 to n+1 and j!=i

	>>> time = [0, 20, 40, 60, 80, 100]
	>>> temp = [26.0, 48.6, 61.6, 71.2, 74.8, 75.2]
	>>> np.around(lagrange_method(50, time, temp), 1)
	66.9
	"""
	# Degree of interpolation polynomial
	n = len(xlist) - 1
	def l(x, i):
		value = 1
		for j in range(n+1):
			if i != j:
				value *= (x - xlist[j])/(xlist[i] - xlist[j])
		return value
	def y(x):
		value = 0
		for i in range(n+1):
			value += ylist[i]*l(x, i)
		return value
	return y(xp)

def newton_method(xp, xlist, ylist):
	"""
	Note: 
	Newton's method predict the y value of a given `xp` by 
	creating a interpolation  polynomial of  degree n in 
	the form of a0+a1(x-x1)+a2(x-x1)(x-x2)+...
	and it'll need n+1 points.

	>>> xlist = [0.0, 1.5, 2.8, 4.4, 6.1, 8.0]
	>>> ylist = [0.0, 0.9, 2.5, 6.6, 7.7, 8.0]
	>>> np.around(newton_method(0.0, xlist, ylist), 1)
	0.0
	>>> np.around(newton_method(1.5, xlist, ylist), 1)
	0.9
	>>> np.around(newton_method(2.8, xlist, ylist), 1)
	2.5
	>>> np.around(newton_method(4.4, xlist, ylist), 1)
	6.6
	>>> np.around(newton_method(6.1, xlist, ylist), 1)
	7.7
	>>> np.around(newton_method(8.0, xlist, ylist), 1)
	8.0
	>>> np.around(newton_method(4, xlist, ylist), 1)
	5.6
	"""
	n = len(xlist) - 1
	# Construct the divided difference table
	Dy = np.zeros((n+1, n+1))
	Dy[:,0] = ylist
	for j in range(n):
		for i in range(j+1, n+1):
			Dy[i, j+1] = (Dy[i, j] - Dy[j, j])/(xlist[i] - xlist[j])
	def x_product(xp, degree):
		product = 1
		for j in range(degree):
			product *= xp - xlist[j]
		return product
	return sum([Dy[i, i]*x_product(xp, i) for i in range(n+1)])




