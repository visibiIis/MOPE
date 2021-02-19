from random import randint

# input data
A0 = 1
A1 = 2
A2 = 3
A3 = 4

Nodes = 8
Factors = 3

def gen_exp(factors, nodes):
    return [[randint(0, 20) for _ in range(nodes)] for _ in range(factors)]

def calc_y(x1, x2, x3):
    return A0 + A1 * x1 + A2 * x2 + A3 * x3

def calc_x0(x):
    return (max(x) + min(x)) / 2

def calc_dx(x0, x):
    return x0 - min(x)

def calc_xn(x0, x, dx):
    return [round((x[i] - x0) / dx, 2) for i in range(len(x))]

def decision(y, yet):
    return [(y[i] - yet) ** 2 for i in range(len(y))]

X1, X2, X3 = gen_exp(Factors, Nodes)

Y = [calc_y(X1[i], X2[i], X3[i]) for i in range(Nodes)]

X01 = calc_x0(X1)
X02 = calc_x0(X2)
X03 = calc_x0(X3)

Yet = calc_y(X01, X02, X03)

dx1 = calc_dx(X01, X1)
dx2 = calc_dx(X02, X2)
dx3 = calc_dx(X03, X3)

Xn1 = calc_xn(X01, X1, dx1)
Xn2 = calc_xn(X02, X2, dx2)
Xn3 = calc_xn(X03, X3, dx3)

Y_decision = decision(Y, Yet)
dec_min = min(Y_decision)
dec_index = Y_decision.index(dec_min)
Y_optimal = Y[dec_index]
X_optimal = [X1[dec_index], X2[dec_index], X3[dec_index]]

print("    X1    X2    X3     Y   |   Xn1     Xn2     Xn3")
for i in range(Nodes):
    print(f"--  {X1[i]:<6}{X2[i]:<6}{X3[i]:<6}{Y[i]:<5}|  {Xn1[i]:<8}{Xn2[i]:<8}{Xn3[i]:<8}")
print(f"X0  {X01:<5} {X02:<5} {X03:<6}{'--':<5}|  {'--':<8}{'--':<8}{'--':<8}")
print(f"dx  {dx1:<5} {dx2:<5} {dx3:<6}{'--':<5}|  {'--':<8}{'--':<8}{'--':<8}")
print(f"\nYет = {Yet}")
print(f"\nmin(Y - Yет)² = {dec_min}")
print(f"Yопт = {Y_optimal} --> Y{X_optimal}")
