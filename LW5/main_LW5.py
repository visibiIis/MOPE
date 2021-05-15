from random import randint
from functools import reduce
from _decimal import Decimal
from scipy.stats import f, t
import numpy as np

# Cramer's rule
def cramer(arr, ins, pos):
    matrix = np.insert(np.delete(arr, pos, 1), pos, ins, 1)

    return np.linalg.det(matrix) / np.linalg.det(arr)

# Method to get dispersion
def getDispersion(y, y_r):
    return [round(sum([(y[i][j] - y_r[i]) ** 2 for j in range(len(y[i]))]) / 3, 3) for i in range(len(y_r))]

def generate_factors_table(raw_array):
    new_list = [row + [row[0] * row[1], row[0] * row[2], row[1] * row[2], row[0] * row[1] * row[2]]
                + list(map(lambda x: round(x ** 2, 5), row))
                for row in raw_array]
    return np.array(new_list)

def m_ij(*arrays):
    return np.average(reduce(lambda accum, el: accum * el, arrays))

# Cochran criteria
def cochran(disp, m):
    def get_cochran_value(f1, f2, q):
        partResult1 = q / f2
        params = [partResult1, f1, (f2 - 1) * f1]
        fisherc = f.isf(*params)
        result = fisherc / (fisherc + (f2 - 1))
        return Decimal(result).quantize(Decimal('.0001')).__float__()

    f_1 = m - 1
    f_2 = len(disp)
    p = 0.95
    q_ = 1 - p

    Gp = max(disp) / sum(disp)
    Gt = get_cochran_value(f_1, f_2, q_)

    return [round(Gp, 4), Gt[m - 2]]

# Student criteria
def student(disp, m, y_r, x_nT):
    def get_student_value(f3, q):
        return Decimal(abs(t.ppf(q / 2, f3))).quantize(Decimal('.0001')).__float__()

    N = len(y_r)
    f3 = (m - 1) * N
    q = 0.05
    t_our = get_student_value(f3, q)

    Sb = sum(disp) / len(y_r)
    Sbeta = (Sb / (m * N)) ** (1 / 2)

    beta = [sum([y_r[j] * x_nT[i][j] for j in range(N)]) / N for i in range(N)]
    t_i = [abs(beta[i]) / Sbeta for i in range(len(beta))]

    importance = [True if el > t_our else False for el in list(t_i)]

    return importance

# Fisher criteria
def fisher(y_r, y_st, b_det, disp, m):
    def get_fisher_value(f3, f4, q):
        return Decimal(abs(f.isf(q, f4, f3))).quantize(Decimal('.0001')).__float__()

    N = len(y_r)
    Sb = sum(disp) / N
    d = 0
    for b in b_det:
        if b:
            d += 1

    f4 = N - d
    f3 = N * (m - 1)
    q = 0.05
    Sad = (m / f4) * sum([(y_st[i] - y_r[i]) ** 2 for i in range(N)])
    Fap = Sad / Sb
    Ft = get_fisher_value(f3, f4, q)

    if Fap < Ft:
        return f"\nРівняння регресії адекватно оригіналу:\nFap < Ft: {round(Fap, 2)} < {Ft}"
    else:
        return f"\nРівняння регресії неадекватно оригіналу: \nFap > Ft: {round(Fap, 2)} > {Ft}"

# Main function
def experiment(m, min_x1, max_x1, min_x2, max_x2, min_x3, max_x3):
    y_min = round((min_x1 + min_x2 + min_x3) / 3) + 200
    y_max = round((max_x1 + max_x2 + max_x3) / 3) + 200

    x_norm_prt = [
        [-1, -1, -1],
        [-1, -1, +1],
        [-1, +1, -1],
        [-1, +1, +1],
        [+1, -1, -1],
        [+1, -1, +1],
        [+1, +1, -1],
        [+1, +1, +1],
        [-1.215, 0, 0],
        [+1.215, 0, 0],
        [0, -1.215, 0],
        [0, +1.215, 0],
        [0, 0, -1.215],
        [0, 0, +1.215],
        [0, 0, 0]
    ]

    x0i = []
    xi = []

    for i in [[min_x1, max_x1], [min_x2, max_x2], [min_x3, max_x3]]:
        x0 = round((i[1] + i[0]) / 2, 3)
        x0i.append(x0)
        xi.append([round(1.215 * (i[1] - x0) + x0, 3), round(-1.215 * (i[1] - x0) + x0, 3)])

    x_prt = [
        [-min_x1, -min_x2, -min_x3],
        [-min_x1, -max_x2, max_x3],
        [-max_x1, min_x2, -max_x3],
        [-max_x1, max_x2, min_x3],
        [min_x1, -min_x2, -max_x3],
        [min_x1, -max_x2, min_x3],
        [max_x1, min_x2, -min_x3],
        [max_x1, max_x2, max_x3],
        [xi[0][0], x0i[1], x0i[2]],
        [xi[0][1], x0i[1], x0i[2]],
        [x0i[0], xi[1][0], x0i[2]],
        [x0i[0], xi[1][1], x0i[2]],
        [x0i[0], x0i[1], xi[2][0]],
        [x0i[0], x0i[1], xi[2][1]],
        [x0i[0], x0i[1], x0i[2]]
    ]

    x_norm = generate_factors_table(x_norm_prt)
    x = generate_factors_table(x_prt)
    # for i in x:
    #     print(i)

    # print()
    x_normT = x_norm.T
    xT = x.T

    N = len(x)
    y = [[randint(y_min, y_max) for _ in range(m)] for _ in range(N)]
    y_r = [round(sum(y[i]) / len(y[i]), 2) for i in range(N)]

    disp = getDispersion(y, y_r)
    print(disp)
    cochran_cr = cochran(disp, m)

    # Get coefficients
    x1 = xT[0]
    x2 = xT[1]
    x3 = xT[2]
    yi = np.array(y_r)

    x_tmp = [N, sum(x1), sum(x2), sum(x3), sum(x1 * x2), sum(x1 * x3), sum(x2 * x3), sum(x1 * x2 * x3),
             sum(x1**2), sum(x2**2), sum(x3**2)]

    x_i = [x_tmp] + [[x_tmp[i]*x_tmp[j] for j in range(len(x_tmp))] for i in range(1, len(x_tmp))]
    y_free = [round(sum(yi), 3)] + [round(sum(yi) * x_tmp[i], 3) for i in range(1, len(x_tmp))]
    beta = np.linalg.solve(x_i, y_free)

    x1_norm = x_normT[0]
    x2_norm = x_normT[1]
    x3_norm = x_normT[2]

    b_norm = [sum(yi) / N, sum(yi * x1_norm / N), sum(yi * x2_norm) / N, sum(yi * x3_norm) / N,
              sum(yi * x1_norm * x2_norm) / N, sum(yi * x1_norm * x3_norm) / N, sum(yi * x2_norm * x3_norm) / N,
              sum(yi * x1 * x2 * x3_norm) / N]

    b_det = student(disp, m, y_r, x_normT)
    b_cut = b.copy()

    # Simplified equations
    if b_det is None:
        return
    else:
        for i in range(N):
            if not b_det[i]:
                b_cut[i] = 0

        y_st = [round(sum([b_cut[0]] + [x[i][j] * b_cut[j + 1] for j in range(N - 1)]), 2) for i in range(N)]

    # Calculate F-test
    fisher_cr = fisher(y_r, y_st, b_det, disp, m)

    # Print out results
    print(f"\nМатриця планування для m = {m}:")
    for i in range(m):
        print(f"Y{i + 1} - {np.array(y).T[i]}")

    print(f"\nСередні значення функції відгуку за рядками:\nY_R: {y_r}")
    print(f"\nКоефіцієнти рівняння регресії:")
    for i in range(len(b)):
        print(f"b{i} = {round(b[i], 3)}")

    if cochran_cr[0] < cochran_cr[1]:
        print(f"\nЗа критерієм Кохрена дисперсія однорідна:\nGp < Gt - {cochran_cr[0]} < {cochran_cr[1]}")
    else:
        print(f"\nЗа критерієм Кохрена дисперсія неоднорідна:\nGp > Gt - {cochran_cr[0]} > {cochran_cr[1]}"
              f"\nСпробуйте збільшити кілкість експериментів.")
        return

    print(f"\nЗа критерієм Стьюдента коефіцієнти ", end="")
    for i in range(len(b_det)):
        if not b_det[i]:
            print(f"b{i} ", end="")
    print("приймаємо незначними")

    print(f"\nОтримані функції відгуку зі спрощеними коефіцієнтами:\nY_St - {y_st}")
    print(fisher_cr)

    return True

if __name__ == '__main__':
    Min_x1, Max_x1 = -6, 6
    Min_x2, Max_x2 = -5, 5
    Min_x3, Max_x3 = -10, 8

    M = 3

    experiment(M, Min_x1, Max_x1, Min_x2, Max_x2, Min_x3, Max_x3)
