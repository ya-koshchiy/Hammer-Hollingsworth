# Метод Хаммера-Холінгсуорта

Ця програма чисельно інтегрує жорстку систему звичайних диференціальних рівнянь Лотки-Вольтерри з насиченням хижака методом Хаммера-Холінгсуорта (неявним методом Рунге-Кутти 4-го порядку) з автоматичною зміною кроку інтегрування.

Коефіцієнти _k_ знаходяться методом простої ітерації.


_Python code:_

import matplotlib.pyplot as plt
import sympy as smp
import math

epsilon, h, T = 10 ** -6, 1, 10
a, b, c, d, s = 2, 1, 1, 1, 0.1
t, x_i, y_i = [0], [2], [1]
errors, steps = [0], [h]

print("\n" + "Швидкість розмноження популяції жертви за відсутності хижака:   a = " + str(a))
print("Питома швидкість споживання популяцією хижака популяції жертви за одиничної щільності обох популяцій:   b = "
      + str(b))
print("Питома смертність хижака:   c = " + str(c))
print("Коефіцієнт переробки спожитої хижаком біомаси у власну біомасу: " + str(d / b) + ",   де d = " + str(d) +
      ", b = " + str(b))
print("Коефіцієнт насичення хижака:   s = " + str(s) + "\n")

kx1, kx2 = smp.Symbol('kx1'), smp.Symbol('kx2')
ky1, ky2 = smp.Symbol('ky1'), smp.Symbol('ky2')

print("Початкові умови: t = " + str(t[-1]))
print("                 (x; y) = (" + str(x_i[0]) + "; " + str(y_i[0]) + ")" + "\n")


def Runge_Kutta(step):
    x_step_1 = x_i[-1] + kx1 / 4 + kx2 * (1 / 4 - math.sqrt(3) / 6)
    y_step_1 = y_i[-1] + ky1 / 4 + ky2 * (1 / 4 - math.sqrt(3) / 6)

    x_step_2 = x_i[-1] + kx1 * (1 / 4 + math.sqrt(3) / 6) + kx2 / 4
    y_step_2 = y_i[-1] + ky1 * (1 / 4 + math.sqrt(3) / 6) + ky2 / 4

    f_kx1 = step * (a * x_step_1 - (b * x_step_1 * y_step_1) / (1 + s * x_step_1))
    f_kx2 = step * (a * x_step_2 - (b * x_step_2 * y_step_2) / (1 + s * x_step_2))
    f_ky1 = step * (-c * y_step_1 + (d * x_step_1 * y_step_1) / (1 + s * x_step_1))
    f_ky2 = step * (-c * y_step_2 + (d * x_step_2 * y_step_2) / (1 + s * x_step_2))

    f_kx1 = smp.lambdify([kx1, kx2, ky1, ky2], f_kx1)
    f_kx2 = smp.lambdify([kx1, kx2, ky1, ky2], f_kx2)
    f_ky1 = smp.lambdify([ky1, ky2, kx1, kx2], f_ky1)
    f_ky2 = smp.lambdify([ky1, ky2, kx1, kx2], f_ky2)

    k_approx, residual = [0, 0, 0, 0], [1, 1, 1, 1]
    while max(residual) > 0.001:
        k_approx = [f_kx1(k_approx[0], k_approx[1], k_approx[2], k_approx[3]),
                    f_kx2(k_approx[0], k_approx[1], k_approx[2], k_approx[3]),
                    f_ky1(k_approx[0], k_approx[1], k_approx[2], k_approx[3]),
                    f_ky2(k_approx[0], k_approx[1], k_approx[2], k_approx[3])]
        residual = [k_approx[0] - f_kx1(k_approx[0], k_approx[1], k_approx[2], k_approx[3]),
                    k_approx[1] - f_kx2(k_approx[0], k_approx[1], k_approx[2], k_approx[3]),
                    k_approx[2] - f_ky1(k_approx[0], k_approx[1], k_approx[2], k_approx[3]),
                    k_approx[3] - f_ky2(k_approx[0], k_approx[1], k_approx[2], k_approx[3])]

    return [x_i[-1] + (k_approx[0] + k_approx[1]) / 2, y_i[-1] + (k_approx[2] + k_approx[3]) / 2]


print("Обчислення запущено...")
while t[-1] < T:
    solve_whole = Runge_Kutta(h)

    solve_half = Runge_Kutta(h / 2)
    x_i.append(solve_half[0])
    y_i.append(solve_half[1])
    solve_half = Runge_Kutta(h / 2)
    x_i.remove(x_i[-1])
    y_i.remove(y_i[-1])

    Runge_rule_error = max(abs(solve_whole[0] - solve_half[0]) / 15, abs(solve_whole[1] - solve_half[1]) / 15)

    if Runge_rule_error < epsilon:
        while Runge_rule_error < epsilon:
            h = 2 * h

            solve_half = solve_whole
            x_i.append(solve_half[0])
            y_i.append(solve_half[1])
            solve_half = Runge_Kutta(h / 2)
            x_i.remove(x_i[-1])
            y_i.remove(y_i[-1])

            solve_whole = Runge_Kutta(h)

            Runge_rule_error = max(abs(solve_whole[0] - solve_half[0]) / 15, abs(solve_whole[1] - solve_half[1]) / 15)

    if Runge_rule_error > epsilon:
        while Runge_rule_error > epsilon:
            h = h / 2

            solve_whole = Runge_Kutta(h)

            solve_half = Runge_Kutta(h / 2)
            x_i.append(solve_half[0])
            y_i.append(solve_half[1])
            solve_half = Runge_Kutta(h / 2)
            x_i.remove(x_i[-1])
            y_i.remove(y_i[-1])

            Runge_rule_error = max(abs(solve_whole[0] - solve_half[0]) / 15, abs(solve_whole[1] - solve_half[1]) / 15)

    if (x_i[-1] > epsilon) & (y_i[-1] > epsilon):
        steps.append(h)
        t.append(t[-1] + h)

        x_i.append(solve_half[0])
        y_i.append(solve_half[1])

        errors.append(Runge_rule_error)

    if (t[-1] >= T / 4) & (t[-2] < T / 4):
        print("25% . . .")
    elif (t[-1] >= T / 2) & (t[-2] < T / 2):
        print("50% . . .")
    elif (t[-1] >= T * 3 / 4) & (t[-2] < T * 3 / 4):
        print("75% . . .")

    if x_i[-1] <= epsilon:
        print("Популяція жертви вимерла.")
        break
    if y_i[-1] <= epsilon:
        print("Популяція хижака вимерла.")
        break

if (t[-1] > T) & (x_i[-1] > 0) & (y_i[-1] > 0):
    rudiment = t[-1] - T
    steps[-1] = steps[-1] - rudiment
    t[-1] = t[-2] + steps[-1]

    solve = Runge_Kutta(h)
    x_i[-1] = solve[0]
    y_i[-1] = solve[1]

if (x_i[-1] > epsilon) & (y_i[-1] > epsilon):
    print("Систему проінтегровано!\n")

print("Максимальна похибка (" + str(max(errors)) + ") не перевищує допустиму (" + str(epsilon) + "): " +
      str(max(errors) <= epsilon))

plt.plot(t, x_i)
plt.plot(t, y_i)
plt.title("Графіки функцій x(t) i y(t)")
plt.show()

plt.plot(x_i, y_i)
plt.title("Фазовий портрет")
plt.show()

plt.plot(steps)
plt.title("Графік зміни кроку h")
plt.show()

plt.plot(t, errors)
plt.title("Графіки оцінок локальних похибок")
plt.show()

