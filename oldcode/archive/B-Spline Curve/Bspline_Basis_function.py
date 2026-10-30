# importing the required module
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

TOTAL_POINTS = 200


def get_points(ini, fin, tot):
    points = []
    dif = (fin - ini) / tot
    points.append(ini)
    for item in range(tot - 1):
        ini += dif
        points.append(ini)
    return points


def zeroth(u_in, all_points):
    final_N_zero = []
    for item in range(len(u_in) - 1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        plt.ylim(0, 1.1)
        plt.xlim(u_in[0], u_in[-1])

        initial = u_in[item]
        final = u_in[item + 1]
        N0 = []
        if initial == final:
            for p in all_points:
                N0.append(0)
        else:
            for p in all_points:
                if initial <= p < final:
                    N0.append(1)
                else:
                    N0.append(0)

        plt.plot(all_points, N0)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        title = "N(" + str(item) + ",0)"
        plt.title(title)
        plt.show()
        final_N_zero.append(N0)
    return final_N_zero


def zero_plus(N_prev, u_in, all_points, p):
    final_N_zero_plus = []
    for num in range(len(u_in) - p - 1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        plt.ylim(0, 1.1)
        plt.xlim(u_in[0], u_in[-1])

        Np = []
        first = u_in[num]
        second = u_in[num + 1]
        third = u_in[num + p]
        final = u_in[num + p + 1]

        for n, point in enumerate(all_points):
            numerator1 = point - first
            denominator1 = third - first
            multiplier1 = N_prev[num][n]    #doubt
            numerator2 = final - point
            denominator2 = final - second
            multiplier2 = N_prev[num + 1][n]

            if multiplier1 == 0 and multiplier2 != 0:
                N_value = (numerator2 * multiplier2) / denominator2
            elif multiplier1 != 0 and multiplier2 == 0:
                N_value = (numerator1 * multiplier1) / denominator1
            elif multiplier1 == 0 and multiplier2 == 0:
                N_value = 0
            else:
                N_value = (numerator1 * multiplier1) / denominator1 + (numerator2 * multiplier2) / denominator2
            Np.append(N_value)

        final_N_zero_plus.append(Np)
        plt.plot(all_points, Np)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        title = "N(" + str(num) + "," + str(p) + ")"
        plt.title(title)
        plt.show()
    return final_N_zero_plus


if __name__ == "__main__":
    u = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]  # Non-Uniform knot Vector
    plot_points = get_points(u[0], u[-1], TOTAL_POINTS)
    p = int(input("ENTER DEGREE(p)   :"))
    print("Plotting .....")
    N_prev = []
    for item in range(p + 1):
        if item == 0:
            N_prev = zeroth(u, plot_points)
        else:
            N_prev = zero_plus(N_prev, u, plot_points, item)
