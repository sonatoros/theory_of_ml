import numpy as np
def gradient_descent(A, b, T, stepsize):
    step = 0.1
    m,n = A.shape
    print(A.shape)
    theta = np.zeros((n,1))
    print("theta shape", theta.shape)
    f = np.zeros(T)

    for i in range(T):
        f[i] = np.linalg.norm(A.dot(theta)-b)**2   #norm of (Ax-b)^2
        g = 2*np.transpose(A).dot(A.dot(theta) - b)  #gradient
        # print("gshape", g.shape)
        k = stepsize*g
        # print(k.shape)
        theta = theta - stepsize*g
    return theta, f




#part C; defining A and vectors
n = 500
m=2*n

A = np.random.uniform(-1,1, (m,n))
x_star = np.random.uniform(-1,1,(n,1))
eta = np.random.normal(0, 0.5, (m,1))
print(eta.shape)
b = A.dot(x_star) + eta
print(b.shape)
# print(b+eta)

T = 50 #number of steps
theta, f = gradient_descent(A,b, T, 1/10)
print(theta, f)
