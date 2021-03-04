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
	Lagrande's method predict the y value by 
	creating a interpolation polynomial of  
	degree n and it'll need n+1 points.

	Interpolation polynomial: 
	y(x) = sum(y_i*l_i(x)) for i from 1 to n+1
	l_i(x) = prod((x-x_j)/(x_i-x_j)) for j from 1 to n+1 and j!=i

	>>> time = [0, 20, 40, 60, 80, 100]
	>>> temp = [26.0, 48.6, 61.6, 71.2, 74.8, 75.2]
	>>> 
	"""
	# Degree of interpolation polynomial
	pass