import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

resultsdir = Path(__file__).parent.joinpath("results")
os.makedirs(resultsdir, exist_ok=True)

def henon_map(x, y):
    x_next = 1 - 1.4 * x * x + y
    y_next = 0.3 * x
    return x_next, y_next

def henon_propagator(x, y):
    return np.array([
            [-2.8*x,  1.],
            [0.3,     0.]
        ])

def l2diff(x1, x2, y1, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def henon_tlm(x, y, dx, dy):
    M = henon_propagator(x, y)
    epsilon = np.array([dx, dy])
    return M @ epsilon

def question1():
    print("\n--- Question 1 ---")
    x_values = []
    y_values = []
    x, y = 1., 1.
    x_values.append(x)
    y_values.append(y)

    print("Simulating...")
    for _ in range(1_000_000):
        x, y = henon_map(x, y)
        x_values.append(x)
        y_values.append(y)
    print("Simulation complete.")

    plt.scatter(x_values[:1_000_000], y_values[:1_000_000], s=0.05, color="tab:blue")
    plt.plot(x_values[:7], y_values[:7], linewidth=1, color="tab:orange")
    plt.title("Henon attractor: N=1 million, (X0,Y0)=(1,1)\nPath of first 7 points marked in orange)")
    plt.savefig(resultsdir / "q1_henon_with_init.png")
    plt.close()
    print("Plotted attractor plus orbit initial path.")

    plt.scatter(x_values[100:], y_values[100:], s=0.05)
    plt.title("Henon attractor: N=1 million, (X0,Y0)=(1,1)\nFirst 100 points discarded)")
    plt.savefig(resultsdir / "q1_henon_attractor_full.png")
    plt.close()
    print("Plotted full attractor.")

    plt.scatter(x_values[100:], y_values[100:], s=0.2)
    plt.ylim(0.207, 0.213)
    plt.xlim(0.302, 0.311)
    plt.title("Henon attractor, zoomed in")
    plt.savefig(resultsdir / "q1_henon_attractor_zoomed.png")
    plt.close()
    print("Plotted zoomed attractor.")

def questions2through4():
    print("\n--- Question 2 ---")

    x, y = -1., 0.36
    M = henon_propagator(x, y)

    dx, dy = 0.2, 0.2
    xp, yp = x + dx, y + dy

    dx2, dy2 = 0.02, 0.02
    xp2, yp2 = x + dx2, y + dy2

    tlm_xp_arr_1 = []
    tlm_yp_arr_1 = []
    actual_xp_arr_1 = []
    actual_yp_arr_1 = []
    tlm_xp_arr_1.append(xp)
    tlm_yp_arr_1.append(yp)
    actual_xp_arr_1.append(xp)
    actual_yp_arr_1.append(yp)

    tlm_xp_arr_2 = []
    tlm_yp_arr_2 = []
    actual_xp_arr_2 = []
    actual_yp_arr_2 = []
    tlm_xp_arr_2.append(xp2)
    tlm_yp_arr_2.append(yp2)
    actual_xp_arr_2.append(xp2)
    actual_yp_arr_2.append(yp2)

    for _ in range(14):
        dx, dy = henon_tlm(x, y, dx, dy)
        dx2, dy2 = henon_tlm(x, y, dx2, dy2)
        x, y = henon_map(x, y)
        xp, yp = henon_map(xp, yp)
        xp2, yp2 = henon_map(xp2, yp2)
        M = henon_propagator(x, y) @ M

        tlm_xp_arr_1.append(x + dx)
        tlm_yp_arr_1.append(y + dy)
        actual_xp_arr_1.append(xp)
        actual_yp_arr_1.append(yp)

        tlm_xp_arr_2.append(x + dx2)
        tlm_yp_arr_2.append(y + dy2)
        actual_xp_arr_2.append(xp2)
        actual_yp_arr_2.append(yp2)

    print(f"Propagated (dx, dy) = ({dx:.3f}, {dy:.3f})")
    print(f"Final value (x, y) = ({x:.3f}, {y:.3f})")

    print("\n--- Question 3 ---")
    print("For (dx0, dy0) = (0.2, 0.2):")
    print(f"\tTLM estimate perturbed final value (x+dx, y+dy) ~= ({(x+dx):.3f}, {(y+dy):.3f})")
    print(f"\tActual perturbed final value (x+dx, y+dy) = ({xp:.3f}, {yp:.3f})")
    print("For (dx0, dy0) = (0.02, 0.02):")
    print(f"\tTLM estimate perturbed final value (x+dx, y+dy) ~= ({(x+dx2):.3f}, {(y+dy2):.3f})")
    print(f"\tActual perturbed final value (x+dx, y+dy) = ({xp2:.3f}, {yp2:.3f})")

    differences1 = [l2diff(x1, x2, y1, y2) for x1, y1, x2, y2 in zip(tlm_xp_arr_1, tlm_yp_arr_1, actual_xp_arr_1, actual_yp_arr_1)]
    differences2 = [l2diff(x1, x2, y1, y2) for x1, y1, x2, y2 in zip(tlm_xp_arr_2, tlm_yp_arr_2, actual_xp_arr_2, actual_yp_arr_2)]
    plt.plot(range(15), differences1, label="(dx0, dy0)=(0.2, 0.2)")
    plt.plot(range(15), differences2, label="(dx0, dy0)=(0.02, 0.02)")
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("||TLM-Actual||")
    plt.title("Henon map perturbation evolution: TLM estimate vs actual\nEuclidean norm of difference")
    plt.savefig(resultsdir / "q3_difference.png")

    print("Plotted TLM estimate skill.")

    print("\n--- Question 4 ---")
    print(f"Composite M = {M}")
    adj_sens = M.T @ np.array([1., 1.])
    print(f"Adjoint sensitivity dJ/dX0 = M^T dJ/dXt = {adj_sens}")
    pert_est = adj_sens @ np.array([0.2, 0.2])
    print(f"Adjoint-estimated forecast perturbation to J: {pert_est}")

def question5():
    print("\n--- Question 5 ---")
    print("For (x0, y0)=(1,1) over 14 timesteps...")
    x, y = 1., 1.
    M = henon_propagator(x, y)
    for _ in range(14):
        x, y = henon_map(x, y)
        M = henon_propagator(x, y) @ M
    U, s, Vt = np.linalg.svd(M)
    leading_singular_vector = U[:, 0]
    amplification = s[0]
    print(f"Leading left singular vector: {leading_singular_vector}")
    print(f"Amplification factor (singular value): {amplification}")

if __name__ == "__main__":
    question1()
    questions2through4()
    question5()
