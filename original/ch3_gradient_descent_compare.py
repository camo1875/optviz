# 3.5. Comparing gradient descent methods on linear regression
# Restored from the report's code (pp.19-25), copied as printed.
# Only what prevented it from running was fixed; those lines are marked [restoration fix].
# Bugs that change the results were deliberately left in and marked [known issue]. See optviz/ for the improved version.

import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.random.rand(12)*5  # 12 x values between 0 and 5
y = 3*x+np.random.randn(12) + 1  # 12 y values: 3x+1 plus random noise
# storage for the predictions of the three methods
# y_t[0] : BGD predictions, y_t[1] : SGD predictions, y_t[2] : mBGD predictions
# [known issue] filled with integer 0, so the dtype is int. Predictions stored later lose their fractional
#               part, and the bias b is learned larger than the optimum.
y_t = np.full((3, 12), 0)
print(f"x:{x}")
print(f"y:{y}")
print(f"y_t:{y_t}")

init_w = 0.2  # initial weight
init_b = 0.8  # initial bias
w = np.full(3, init_w)  # w[0] : BGD weight, w[1] : SGD weight, w[2] : mBGD weight
b = np.full(3, init_b)  # b[0] : BGD bias, b[1] : SGD bias, b[2] : mBGD bias
EPOCH = 100
ETA = 0.01
print(f"w:{w}")
print(f"b:{b}")

hist_BGD_w = []
hist_BGD_b = []
hist_SGD_w = []
hist_SGD_b = []
hist_mBGD_w = []
hist_mBGD_b = []


def batchW(x, y, y_t):
  err = 0
  for i in range(len(y)):
    err += -2*(y[i]-y_t[i])*x[i]  # [restoration fix] -2(...) -> -2*(...)
  return (1/len(y))*err  # [restoration fix] missing closing parenthesis


def batchB(y, y_t):
  err = 0
  for i in range(len(y)):
    err += -2*(y[i]-y_t[i])  # [restoration fix] -2(...) -> -2*(...)
  return (1/len(y))*err


def stocW(x, y, y_t):
  err = -2*(y-y_t)*x
  return err


def stocB(y, y_t):
  err = -2*(y-y_t)
  return err


batch_size = 3


def miniW(x, y, y_t, minibatch):
  err = 0
  for i in minibatch:
    err += -2*(y[i]-y_t[i])*x[i]
  return (1/batch_size)*err


def miniB(y, y_t, minibatch):
  err = 0
  for i in minibatch:
    err += -2*(y[i]-y_t[i])
  return (1/batch_size)*err


for e in range(EPOCH):
  s_count = list(range(len(x)))
  random.shuffle(s_count)
  mb_count = list(range(len(x)))
  random.shuffle(mb_count)
  mb_count = np.array(mb_count)
  mb_count = np.reshape(mb_count, (int(len(x)/batch_size), batch_size))
  mb_count.tolist
  # [known issue] all three methods update 12 times per EPOCH, so BGD sees the whole
  #               data set 12 times and mBGD sees each mini-batch 3 times.
  for j in range(len(y)):
    #BGD
    y_t[0] = w[0]*x+b[0]
    w[0] = w[0]-ETA*batchW(x, y, y_t[0])
    b[0] = b[0]-ETA*batchB(y, y_t[0])
    # bgdG.set_ydata(w[0]*x+b[0])
    #SGD
    y_t[1] = w[1]*x+b[1]
    i = s_count[j]
    w[1] = w[1]-ETA*stocW(x[i], y[i], y_t[1][i])
    b[1] = b[1]-ETA*stocB(y[i], y_t[1][i])
    # sgdG.set_ydata(w[1]*x+b[1])
    #Mini Batch GD
    y_t[2] = w[2]*x+b[2]

    i = j % (int(len(x)/batch_size))

    w[2] = w[2]-ETA*miniW(x, y, y_t[2], mb_count[i])
    b[2] = b[2]-ETA*miniB(y, y_t[2], mb_count[i])
    # mbgdG.set_ydata(w[2]*x+b[2])
  hist_BGD_w.append(w[0])
  hist_BGD_b.append(b[0])
  hist_SGD_w.append(w[1])
  hist_SGD_b.append(b[1])
  hist_mBGD_w.append(w[2])
  hist_mBGD_b.append(b[2])
  if(e % 5 == 4):

    print(f"EPOCH : {e+1}")
    print(f"BGD : w:{w[0]}, b:{b[0]}, deltaw:{init_w-w[0]}, deltab:{init_b-b[0]}")
    print(f"SGD : w:{w[1]}, b:{b[1]}, deltaw:{init_w-w[1]}, deltab:{init_b-b[1]}")
    print(f"mBGD : w:{w[2]}, b:{b[2]}, deltaw:{init_w-w[2]}, deltab:{init_b-b[2]}")

epochs = range(1, EPOCH+1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, hist_BGD_w, linestyle='-', color='red', label='BGD_w', alpha=0.5)
plt.plot(epochs, hist_BGD_b, linestyle='-', color='red', label='BGD_b', alpha=0.5)
plt.plot(epochs, hist_SGD_w, linestyle='-', color='blue', label='SGD_w', alpha=0.5)
plt.plot(epochs, hist_SGD_b, linestyle='-', color='blue', label='SGD_b', alpha=0.5)
plt.plot(epochs, hist_mBGD_w, linestyle='-', color='green', label='mBGD_w', alpha=0.5)
plt.plot(epochs, hist_mBGD_b, linestyle='-', color='green', label='mBGD_b', alpha=0.5)
plt.title(f'Title')
plt.xlabel('Eopchs')
plt.ylabel('delta of w, b')
plt.legend()
plt.show()
