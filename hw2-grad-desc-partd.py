#reference https://machinelearningmastery.com/gradient-descent-optimization-from-scratch/


from numpy import asarray
from numpy.random import rand
import math
import numpy

# objective function
def objective(w,x,y):
    L = 0
    for i in range(len(x)):
        L +=l(x[i],y[i],w)
    return L

def l(xi,yi,w):
    # print(yi,xi,w)
    return math.log((1+math.exp(-1*yi*xi*w)),2)

# derivative of objective function
def derivative(w,x,y):
    Ld = 0
    for i in range(len(x)):
        Ld +=ld(x[i],y[i],w)
    return Ld

def ld(w,xi,yi):
    x = -1*xi*yi*w
    return -1*yi*xi*math.exp(x)/(1+math.exp(x))



# gradient descent algorithm
def gradient_descent(objective, derivative, n_iter, step_size,x,y):
	# generate an initial point
	# w = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    w = -1
	# run the gradient descent
    for i in range(n_iter):
		# calculate gradient
        gradient = derivative(w,x,y)
		# take a step
        w = w - step_size * gradient
		# evaluate candidate point
        w_eval = objective(w,x,y)

		# report progress
        print('%d & f(%s) & %.5f \\\\' % (i, w, w_eval))

    return [w, w_eval]

# define the total iterations
n_iter = 100
# define the step size
step_size = 0.005

x = list(range(-50,51))
y = numpy.tile([-1,1], 100)
for i in range(5):
    y[i] = -1*y[i]
    y[100-i] = -1*y[i]
    print(x[i], x[100-i],y[i],y[100-i])
# perform the gradient descent search
best, score = gradient_descent(objective, derivative, n_iter, step_size, x,y)
print('Done!')
print('f(%s) = %f' % (best, score))
