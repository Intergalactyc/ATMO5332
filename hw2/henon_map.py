import numpy as np
import matplotlib.pyplot as plt

def henon_map(x, y):
    x_next = 1 - 1.4 * x * x + y
    y_next = 0.3 * x
    return x_next, y_next

def henon_propagator(x, y):
    return np.array([
            [-2.8*x,  1.],
            [0.3,     0.]
        ])

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
    plt.savefig("results/henon_with_init.png")
    plt.close()
    print("Plotted attractor plus orbit initial path.")

    plt.scatter(x_values[100:], y_values[100:], s=0.05)
    plt.title("Henon attractor: N=1 million, (X0,Y0)=(1,1)\nFirst 100 points discarded)")
    plt.savefig("results/henon_attractor_full.png")
    plt.close()
    print("Plotted full attractor.")

    plt.scatter(x_values[100:], y_values[100:], s=0.2)
    plt.ylim(0.207, 0.213)
    plt.xlim(0.302, 0.311)
    plt.title("Henon attractor, zoomed in")
    plt.savefig("results/henon_attractor_zoomed.png")
    plt.close()
    print("Plotted zoomed attractor.")

def questions2and3():
    x, y = 1., 0.36
    dx, dy = 0.2, 0.2
    xp, yp = x + dx, y + dy
    for _ in range(14):
        dx, dy = henon_tlm(x, y, dx, dy)
        x, y = henon_map(x, y)
        xp, yp = henon_map(xp, yp)
    print("\n--- Question 2 ---")
    print(f"Propagated (dx, dy) = ({dx:.3f}, {dy:.3f})")
    print(f"Final value (x, y) = ({x:.3f}, {y:.3f})")
    print(f"TLM estimate perturbed final value (x+dx, y+dy) ~= ({(x+dx):.3f}, {(y+dy):.3f})")
    print(f"Actual perturbed final value (x+dx, y+dy) = ({xp:.3f}, {yp:.3f})")

if __name__ == "__main__":
    # question1()
    questions2and3()
    # question4()
    # question5()
