# 3.5. 선형 회귀에서 경사 하강법들의 비교
# 보고서 19~25쪽 코드를 그대로 옮겨 복원한 것.
# 실행이 안 되는 부분만 최소한으로 고쳤고, 고친 곳에는 [복원 수정] 주석을 달았다.
# 동작상의 문제(버그)는 일부러 고치지 않고 [알려진 문제] 주석으로 표시했다. -> 개량판은 optviz/ 참고

import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.random.rand(12)*5  # 0~5 사이의 12개의 x데이터
y = 3*x+np.random.randn(12) + 1  # 3x+1로부터 무작위 오차를 준 12개의 y데이터
# 세 가지 방법의 예측값을 저장하기 위한 공간 마련.
# y_t[0] : BGD의 예측값, y_t[1] : SGD의 예측값, y_t[2] : mBGD의 예측값
# [알려진 문제] 0(정수)으로 채워서 dtype이 int가 된다. 이후 저장되는 예측값의 소수점 아래가 잘려
#              편향 b가 최적값보다 크게 학습된다.
y_t = np.full((3, 12), 0)
print(f"x:{x}")
print(f"y:{y}")
print(f"y_t:{y_t}")

init_w = 0.2  # 가중치 초깃값
init_b = 0.8  # 편향 초깃값
w = np.full(3, init_w)  # w[0] : BGD의 가중치, w[1] : SGD의 가중치, w[2] : mBGD의 가중치
b = np.full(3, init_b)  # b[0] : BGD의 편향, b[1] : SGD의 편향, b[2] : mBGD의 편향
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
    err += -2*(y[i]-y_t[i])*x[i]  # [복원 수정] -2(...) -> -2*(...)
  return (1/len(y))*err  # [복원 수정] 닫는 괄호 누락


def batchB(y, y_t):
  err = 0
  for i in range(len(y)):
    err += -2*(y[i]-y_t[i])  # [복원 수정] -2(...) -> -2*(...)
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
  # [알려진 문제] 세 방법 모두 한 EPOCH에 12번씩 업데이트한다.
  #              BGD는 전체 데이터를 12번, mBGD는 같은 미니 배치를 3번씩 보게 된다.
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
