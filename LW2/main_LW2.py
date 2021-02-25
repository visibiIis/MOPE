import math
from random import randint
import numpy as np

def solveEquation(m, min_y, max_y, min_x1, max_x1, min_x2, max_x2):
    xn = [[-1, -1], [1, -1], [-1, 1]]

    y = [[randint(min_y, max_y) for _ in range(m)] for _ in range(3)]
    avg_y = [round(sum(y[i][j] for j in range(m)) / m, 2) for i in range(3)]

    # Dispersion for each Y
    disp_list = [sum([(j - avg_y[i]) ** 2 for j in y[i]]) / m for i in range(len(y))]

    main_dev = math.sqrt((2 * (2 * m - 2)) / (m * (m - 4)))

    def calc_fuv(u, v):
        if u >= v:
            return u / v
        else:
            return v / u

    # Fuv
    f_1 = calc_fuv(disp_list[0], disp_list[1])
    f_2 = calc_fuv(disp_list[2], disp_list[0])
    f_3 = calc_fuv(disp_list[2], disp_list[1])

    # Deviation for each Fuv
    dev_list = [((m - 2) / m) * f for f in [f_1, f_2, f_3]]

    # Ruv for each deviation
    r_list = [abs(dev_list[i] - 1) / main_dev for i in range(len(dev_list))]

    rkr = 2
    for ruv in r_list:
        if ruv > rkr:
            m += 1
            break

    # Coefficients
    mx_1 = (xn[0][0] + xn[1][0] + xn[2][0]) / 3
    mx_2 = (xn[0][1] + xn[1][1] + xn[2][1]) / 3
    my = sum(avg_y) / 3

    a1 = (xn[0][0] ** 2 + xn[1][0] ** 2 + xn[2][0] ** 2) / 3
    a2 = (xn[0][0] * xn[0][1] + xn[1][0] * xn[1][1] + xn[2][0] * xn[2][1]) / 3
    a3 = (xn[0][1] ** 2 + xn[1][1] ** 2 + xn[2][1] ** 2) / 3
    a11 = (xn[0][0] * avg_y[0] + xn[1][0] * avg_y[1] + xn[2][0] * avg_y[2]) / 3
    a22 = (xn[0][1] * avg_y[0] + xn[1][1] * avg_y[1] + xn[2][1] * avg_y[2]) / 3

    inv_matrix = np.linalg.inv([[1, mx_1, mx_2], [mx_1, a1, a2], [mx_2, a2, a3]])

    # Get determinants from scalar multiplication
    b0 = np.linalg.det(np.dot([[my, mx_1, mx_2], [a11, a1, a2], [a22, a2, a3]], inv_matrix))
    b1 = np.linalg.det(np.dot([[1, my, mx_2], [mx_1, a11, a2], [mx_2, a22, a3]], inv_matrix))
    b2 = np.linalg.det(np.dot([[1, mx_1, my], [mx_1, a1, a11], [mx_2, a2, a22]], inv_matrix))

    y_r1 = b0 + b1 * xn[0][0] + b2 * xn[0][1]
    y_r2 = b0 + b1 * xn[1][0] + b2 * xn[1][1]
    y_r3 = b0 + b1 * xn[2][0] + b2 * xn[2][1]

    # Coefficient naturalization
    dx1 = math.fabs(max_x1 - min_x1) / 2
    dx2 = math.fabs(max_x2 - min_x2) / 2
    x10 = (max_x1 + min_x1) / 2
    x20 = (max_x2 + min_x2) / 2

    A0 = b0 - b1 * x10 / dx1 - b2 * x20 / dx2
    A1 = b1 / dx1
    A2 = b2 / dx2

    # Naturalized equations
    y1n = A0 + A1 * min_x1 + A2 * min_x2
    y2n = A0 + A1 * max_x1 + A2 * min_x2
    y3n = A0 + A1 * min_x1 + A2 * max_x2

    # Print out results
    if avg_y == [round(y1n, 2), round(y2n, 2), round(y3n, 2)] == [round(y_r1, 2), round(y_r2, 2), round(y_r3, 2)]:
        print(f"\nМатриця планування для m = {len(y[0])}:")
        for i in range(3):
            print(f"Y{i + 1} - {y[i]}")

        print("\nОтримані значення критерію Романовського:")
        for i in range(3):
            print(f"Ruv{i + 1} = {round(r_list[i], 3)}")

        print("\nКоефіцієнти регресійного рівняння та перевірка:")
        print(f"b0 = {round(b0, 3)}")
        print(f"b1 = {round(b1, 3)}")
        print(f"b2 = {round(b2, 3)}")

        print("\nНатуралізовані коефіцієнти:")
        print("a0:", A0)
        print("a1:", A1)
        print("a2:", A2)

        print("\nПеревірка:\n")
        print(f"{'Середні значення:':<23} {avg_y}")
        print(f"{'Рівняння регресії:':<23} {[round(y_r1, 2), round(y_r2, 2), round(y_r3, 2)]}")
        print(f"{'Нормалізоване рівняння:':<23} {[round(y1n, 2), round(y2n, 2), round(y3n, 2)]}")
    else:
        return False

    return True

if __name__ == '__main__':
    M = 5
    Min_y, Max_y = 60, 160

    Min_x1, Max_x1 = -25, 75
    Min_x2, Max_x2 = 5, 40

    success = False

    while not success:
        success = solveEquation(M, Min_y, Max_y, Min_x1, Max_x1, Min_x2, Max_x2)
        M += 1
