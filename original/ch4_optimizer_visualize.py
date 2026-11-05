# 4.6. 옵티마이저들의 시각화
# 보고서 30~36쪽 코드를 그대로 옮겨 복원한 것.
# 실행이 안 되는 부분만 최소한으로 고쳤고, 고친 곳에는 [복원 수정] 주석을 달았다.
# 동작상의 문제(버그)는 일부러 고치지 않고 [알려진 문제] 주석으로 표시했다. -> 개량판은 optviz/ 참고

import random
import time
import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt

#목표 함수 설정
x, y = sp.symbols('x y')
target_f = x**4 - 1/2*x**3 - 3/4*x**2 - 1/8*x + y**2 - 1/2  # [복원 수정] '–'(en dash) -> '-'
dfx = sp.diff(target_f, x)
dfy = sp.diff(target_f, y)


# [알려진 문제] 매 호출마다 sympy의 subs로 기호 계산을 해서 매우 느리고,
#              결과가 sympy Float(object dtype) 배열이다.
def Gradient(point):
  dx = dfx.subs({x: point[0], y: point[1]})
  dy = dfy.subs({x: point[0], y: point[1]})
  return np.array([dx, dy])


f_lambdified = sp.lambdify((x, y), target_f, 'numpy')  #이후 계산을 위한 람다화

start_point = np.array([-0.9, 0.5])
EPOCH = 500
ETA = 0.01
EPSILON = 10**-8
MOMENTUM_MOMENT = 0.9
RMSP_GAMMA = 0.9
ADAM_GAMMA = 0.999
ADAM_MOMENT = 0.9  # 보고서 37쪽: 0.99로 바꾸면 Adam이 최솟값을 찾아간다
doGD, doMoment, doNAG, doAdagrad, doRMSP, doAdam = 1, 1, 1, 1, 1, 1

#3차원상 기존함수 처리
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# [알려진 문제] [-5, 5] 범위에서는 z가 최대 약 600까지 커져서,
#              실제 경로가 움직이는 [-1, 1] 근처가 거의 보이지 않는다.
x_map = np.linspace(-5, 5, 20)
y_map = np.linspace(-5, 5, 20)
x_grid, y_grid = np.meshgrid(x_map, y_map)
z_vals = f_lambdified(x_grid, y_grid)
ax.plot_surface(x_grid, y_grid, z_vals, alpha=0.1)
ax.set_xlim((-5, 5))
ax.set_ylim((-5, 5))

#데이터 처리
if doGD:
  gradient_delta = np.empty((EPOCH, 2))
  gradient_sets = np.empty((1, 2))
  x_GD = gradient_sets[:, 0]
  y_GD = gradient_sets[:, 1]
  z_GD = f_lambdified(x_GD, y_GD)
  gdG, = ax.plot(x_GD, y_GD, z_GD, linewidth="3", label="GD")
if doMoment:
  momentum_delta = np.empty((EPOCH, 2))
  momentum_sets = np.empty((1, 2))
  x_Moment = momentum_sets[:, 0]
  y_Moment = momentum_sets[:, 1]
  z_Moment = f_lambdified(x_Moment, y_Moment)
  momentG, = ax.plot(x_Moment, y_Moment, z_Moment, linewidth="3", label="Momentum")
if doNAG:
  nag_delta = np.empty((EPOCH, 2))
  nag_sets = np.empty((1, 2))
  x_NAG = nag_sets[:, 0]
  y_NAG = nag_sets[:, 1]
  z_NAG = f_lambdified(x_NAG, y_NAG)
  nagG, = ax.plot(x_NAG, y_NAG, z_NAG, linewidth="3", label="NAG")
if doAdagrad:
  adagrad_g = np.empty((EPOCH, 2))
  adagrad_delta = np.empty((EPOCH, 2))
  adagrad_sets = np.empty((1, 2))
  x_AdaGrad = adagrad_sets[:, 0]
  y_AdaGrad = adagrad_sets[:, 1]
  z_AdaGrad = f_lambdified(x_AdaGrad, y_AdaGrad)
  adagradG, = ax.plot(x_AdaGrad, y_AdaGrad, z_AdaGrad, linewidth="3", label="AdaGrad")
if doRMSP:
  rmsprop_g = np.empty((EPOCH, 2))
  rmsprop_delta = np.empty((EPOCH, 2))
  rmsprop_sets = np.empty((1, 2))
  x_RMSProp = rmsprop_sets[:, 0]
  y_RMSProp = rmsprop_sets[:, 1]
  z_RMSProp = f_lambdified(x_RMSProp, y_RMSProp)
  rmspropG, = ax.plot(x_RMSProp, y_RMSProp, z_RMSProp, linewidth="3", label="RMSProp")
if doAdam:
  adam_m = np.empty((EPOCH, 2))
  adam_g = np.empty((EPOCH, 2))
  adam_delta = np.empty((EPOCH, 2))
  adam_sets = np.empty((1, 2))
  x_Adam = adam_sets[:, 0]
  y_Adam = adam_sets[:, 1]
  z_Adam = f_lambdified(x_Adam, y_Adam)
  adamG, = ax.plot(x_Adam, y_Adam, z_Adam, linewidth="3", label="Adam")

plt.xlabel("x axis")
plt.ylabel("y axis")
plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left')
plt.show()

for e in range(EPOCH):
  if(e == 0):
    if doGD:
      gradient_delta[0] = -(Gradient(start_point)*ETA)
      gradient_sets[0] = start_point+gradient_delta[0]
    if doMoment:
      momentum_delta[0] = -Gradient(start_point)*ETA
      momentum_sets[0] = start_point+momentum_delta[0]
    if doNAG:
      nag_delta[0] = -Gradient(start_point)*ETA
      # [알려진 문제] nag_delta가 아니라 momentum_delta를 쓴다.
      #              doMoment = 0이면 초기화되지 않은 값을 쓰게 된다.
      nag_sets[0] = start_point+momentum_delta[0]
    if doAdagrad:
      adagrad_g[0] = Gradient(start_point)*Gradient(start_point)
      adagrad_delta[0] = -(Gradient(start_point) / np.sqrt(adagrad_g[0] + EPSILON))*ETA
      adagrad_sets[0] = start_point+adagrad_delta[0]
    if doRMSP:
      rmsprop_g[0] = Gradient(start_point)*Gradient(start_point)*(1-RMSP_GAMMA)
      rmsprop_delta[0] = -(Gradient(start_point) / np.sqrt(rmsprop_g[0] + EPSILON))*ETA
      rmsprop_sets[0] = start_point + rmsprop_delta[0]
    if doAdam:
      adam_m[0] = Gradient(start_point)*(1-ADAM_MOMENT)
      edit_adam_moment = adam_m[0]/(1-ADAM_MOMENT)
      adam_g[0] = (Gradient(start_point)*Gradient(start_point))*(1-ADAM_GAMMA)
      edit_adam_g = adam_g[0]/(1-ADAM_GAMMA)
      adam_delta[0] = -(edit_adam_moment / np.sqrt(edit_adam_g + EPSILON))*ETA
      adam_sets[0] = start_point + adam_delta[0]
  else:
    if doGD:
      gradient_delta[e] = -(Gradient(gradient_sets[e-1])*ETA)
      gradient_sets = np.append(gradient_sets, gradient_sets[e-1]+gradient_delta[e].reshape(1, 2), axis=0)
    if doMoment:
      momentum_delta[e] = momentum_delta[e-1]*MOMENTUM_MOMENT - Gradient(momentum_sets[e-1])*ETA
      momentum_sets = np.append(momentum_sets, (momentum_sets[e-1]+momentum_delta[e]).reshape(1, 2), axis=0)

    if doNAG:
      nag_delta[e] = nag_delta[e-1]*MOMENTUM_MOMENT - Gradient(nag_sets[e-1]+nag_delta[e-1]*MOMENTUM_MOMENT)*ETA
      nag_sets = np.append(nag_sets, (nag_sets[e-1]+nag_delta[e]).reshape(1, 2), axis=0)
    if doAdagrad:
      adagrad_g[e] = adagrad_g[e-1] + (Gradient(adagrad_sets[e-1])*Gradient(adagrad_sets[e-1]))
      adagrad_delta[e] = -(Gradient(adagrad_sets[e-1]) / np.sqrt(adagrad_g[e] + EPSILON))*ETA
      adagrad_sets = np.append(adagrad_sets, (adagrad_sets[e-1]+adagrad_delta[e]).reshape(1, 2), axis=0)

    if doRMSP:
      rmsprop_g[e] = rmsprop_g[e-1]*RMSP_GAMMA + (Gradient(rmsprop_sets[e-1])*Gradient(rmsprop_sets[e-1]))*(1-RMSP_GAMMA)
      rmsprop_delta[e] = -(Gradient(rmsprop_sets[e-1]) / np.sqrt(rmsprop_g[e]+EPSILON))*ETA
      rmsprop_sets = np.append(rmsprop_sets, (rmsprop_sets[e-1]+rmsprop_delta[e]).reshape(1, 2), axis=0)
    if doAdam:
      adam_m[e] = adam_m[e-1]*ADAM_MOMENT + Gradient(adam_sets[e-1])*(1-ADAM_MOMENT)
      edit_adam_moment = adam_m[e] / (1-(ADAM_MOMENT**(e+1)))
      adam_g[e] = adam_g[e-1]*ADAM_GAMMA + (Gradient(adam_sets[e-1])*Gradient(adam_sets[e-1]))*(1-ADAM_GAMMA)
      edit_adam_g = adam_g[e] / (1-(ADAM_GAMMA**(e+1)))
      adam_delta[e] = -(edit_adam_moment / np.sqrt(edit_adam_g + EPSILON))*ETA
      adam_sets = np.append(adam_sets, (adam_sets[e-1]+adam_delta[e]).reshape(1, 2), axis=0)
  if doGD:
    x_GD = gradient_sets[:, 0]
    y_GD = gradient_sets[:, 1]
    z_GD = f_lambdified(x_GD, y_GD)
    gdG.set_data(x_GD, y_GD)
    gdG.set_3d_properties(z_GD)
  if doMoment:
    x_Moment = momentum_sets[:, 0]
    y_Moment = momentum_sets[:, 1]
    z_Moment = f_lambdified(x_Moment, y_Moment)
    momentG.set_data(x_Moment, y_Moment)
    momentG.set_3d_properties(z_Moment)

  if doNAG:
    x_NAG = nag_sets[:, 0]
    y_NAG = nag_sets[:, 1]
    z_NAG = f_lambdified(x_NAG, y_NAG)
    nagG.set_data(x_NAG, y_NAG)
    nagG.set_3d_properties(z_NAG)
  if doAdagrad:
    x_AdaGrad = adagrad_sets[:, 0]
    y_AdaGrad = adagrad_sets[:, 1]
    z_AdaGrad = f_lambdified(x_AdaGrad, y_AdaGrad)
    adagradG.set_data(x_AdaGrad, y_AdaGrad)
    adagradG.set_3d_properties(z_AdaGrad)
  if doRMSP:
    x_RMSProp = rmsprop_sets[:, 0]
    y_RMSProp = rmsprop_sets[:, 1]
    z_RMSProp = f_lambdified(x_RMSProp, y_RMSProp)
    rmspropG.set_data(x_RMSProp, y_RMSProp)
    rmspropG.set_3d_properties(z_RMSProp)
  if doAdam:
    x_Adam = adam_sets[:, 0]
    y_Adam = adam_sets[:, 1]
    z_Adam = f_lambdified(x_Adam, y_Adam)
    adamG.set_data(x_Adam, y_Adam)
    adamG.set_3d_properties(z_Adam)
  fig.canvas.flush_events()
  time.sleep(0.05)

# [복원 수정] 보고서 코드에는 없음. 반복이 끝난 뒤 창이 바로 닫히지 않도록 추가
print("최종 위치")
for name, on, sets in [("GD", doGD, "gradient_sets"), ("Momentum", doMoment, "momentum_sets"),
                       ("NAG", doNAG, "nag_sets"), ("AdaGrad", doAdagrad, "adagrad_sets"),
                       ("RMSProp", doRMSP, "rmsprop_sets"), ("Adam", doAdam, "adam_sets")]:
  if on:
    print(f"  {name:9s}: {np.array(globals()[sets][-1], dtype=float)}")
plt.ioff()
plt.show()
