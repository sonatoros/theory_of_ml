#reference https://machinelearningmastery.com/gradient-descent-optimization-from-scratch/

from numpy import asarray
from numpy.random import rand
import math
import numpy as np
from random import randrange

# objective function
def objective(x,y,a,b):
    L = 0
    for i in range(len(a)):
        L +=l(x,y,a[i],b[i])
    return L/(2*len(a))

def l(x,y,ai,bi):
    return (x-ai)**2 + (y-bi)**2


def gradient_hx(x,ai):
    return 2*(x-ai)

def gradient_hy(y,bi):
    return 2*(y-bi)



# gradient descent algorithm
def gradient_descent( n_iter, a,b, learning_rate):
	# generate an initial point
	# w = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    x = np.zeros(n_iter+1)
    y = np.zeros(n_iter+1)
    x[0] = 1
    y[0] = 1
	# run the gradient descent
    for j in range(n_iter):
		# calculate gradient
        i = randrange(2*n)
        gxt = gradient_hx(x[j], a[i])
        gyt = gradient_hy(y[j], b[i])
		# take a step
        # learning_rate = 0.1/(j+1)
        x[j+1] = x[j] - learning_rate * gxt
        y[j+1] = y[j] - learning_rate * gyt


		# evaluate candidate point
        f_eval = objective(x[j+1],y[j+1],a,b)
		# report progress
        print('%d & f(%.5f,%.5f) & %.5f \\\\' % (j, x[j+1], y[j+1], f_eval))
    return [x, y, f_eval]

# define the total iterations
n_iter = 200
n = int(n_iter/2)
# define the step size
learning_rate = 0.1
a = np.zeros(n_iter)
b = np.zeros(n_iter)
for i in range(n):
    a[i] = (i+1)/n
    b[i] = -1

for i in range(n,2*n):
    a[i] = (i+1-n)/n
    b[i] = 1

gradient_descent(n_iter, a,b, 0.1)

# # perform the gradient descent search
# best, score = gradient_descent(objective, derivative, n_iter, step_size, x,y)
# print('Done!')
# print('f(%s) = %f' % (best, score))
