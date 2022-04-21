import numpy as np
import random
import math
import matplotlib.pyplot as plt

T = math.pow(10,6)

x1 = 0
x2 = 1
ls = list()
ls.append(x1)
ls.append(x2)
# print(ls)
# total_loss = 0.0
# k = 1
loss_list = list()
eta_list = list()

def choose(eta):
    denom = np.exp(-1*eta*g1) + np.exp(-1*eta*g2)
    x1_prob = np.exp(-1*eta*g1)/denom
    x2_prob = np.exp(-1*eta*g2)/denom
    # choose either 0 or 1 based on their probabilities
    return random.choices(ls, weights=(x1_prob, x2_prob), k=1)[0]

iter = 5
for k in range(1,10001,200):
    if(k < 0):
        break
    print(k)
    eta = 1/float(k)
    total = 0.0
    for j in range(iter):
        total_loss = 0.0
        g1 = 0
        g2 = 0
        for i in range(int(T)):
            current_x = choose(eta)
            # print("x: ", current_x)
            # if even number
            if(i%2==0):
                total_loss =total_loss + -2*current_x
                g2 += -2
            else:
                total_loss =total_loss + 2 * current_x
                g2 += 2
        total+=total_loss
    # print(total_loss/5)
    loss_list.append(abs(total_loss/iter))
    eta_list.append(k)

print(total_loss)
# print(eta)
plt.plot(eta_list, loss_list)
plt.yscale("log")
plt.xlabel("k value")
plt.ylabel("Total Loss")
plt.show()
