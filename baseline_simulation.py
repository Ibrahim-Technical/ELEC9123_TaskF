import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. System parameters from the design task
# ============================================================

X_MIN = 0.0
X_MAX = 100.0
Y_MIN = 0.0
Y_MAX = 100.0
Z_MIN = 0.0
Z_H = 50.0

K = 2                 # number of users
N_T = 8               # number of transmit antennas
P_T = 0.5             # total transmit power budget in Watts
ETA = 2.5             # path loss exponent
FREQ = 2.4e9          # 2.4 GHz
C = 3e8               # speed of light

NOISE_DB = -100       # noise power in dB
NOISE_POWER = 10 ** (NOISE_DB / 10)

START = np.array([0.0, 0.0, Z_H])
END = np.array([80.0, 80.0, Z_H])

L = 200               # episode length / number of time steps


# ============================================================
# 2. Helper functions
# ============================================================

def generate_users(seed=1):
    """
    Generate K users randomly on the ground plane z = 0.
    """
    rng = np.random.default_rng(seed)

    x_users = rng.uniform(X_MIN, X_MAX, K)
    y_users = rng.uniform(Y_MIN, Y_MAX, K)
    z_users = np.zeros(K)

    users = np.column_stack((x_users, y_users, z_users))
    return users


def straight_line_trajectory():
    """
    Generate deterministic UAV trajectory from START to END.
    """
    trajectory = np.zeros((L + 1, 3))

    for t in range(L + 1):
        alpha = t / L
        trajectory[t] = START + alpha * (END - START)

    return trajectory


def compute_distances(uav_position, users):
    """
    Compute distances from UAV to all users.
    """
    diff = users - uav_position
    distances = np.linalg.norm(diff, axis=1)
    return distances


def compute_channel_power(distances):
    """
    Compute |h_k|^2 = L0 / d_k^eta.
    """
    wavelength = C / FREQ
    L0 = (wavelength / (4 * np.pi)) ** 2

    channel_power = L0 / (distances ** ETA)
    return channel_power


def compute_snr(channel_power):
    """
    Compute SNR for each user.

    First simple assumption:
    total power is equally divided between users.
    """
    power_per_user = P_T / K
    snr = channel_power * power_per_user / NOISE_POWER
    return snr


def compute_throughput(snr):
    """
    Compute R_k = log2(1 + SNR_k)
    """
    return np.log2(1 + snr)


# ============================================================
# 3. Run baseline simulation
# ============================================================

def run_baseline():
    users = generate_users(seed=10)
    trajectory = straight_line_trajectory()

    all_user_rates = []
    sum_rates = []

    for uav_position in trajectory:
        distances = compute_distances(uav_position, users)
        channel_power = compute_channel_power(distances)
        snr = compute_snr(channel_power)
        rates = compute_throughput(snr)

        all_user_rates.append(rates)
        sum_rates.append(np.sum(rates))

    all_user_rates = np.array(all_user_rates)
    sum_rates = np.array(sum_rates)

    return users, trajectory, all_user_rates, sum_rates


# ============================================================
# 4. Plot results
# ============================================================

def plot_trajectory(users, trajectory):
    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], label="UAV trajectory")
    plt.scatter(users[:, 0], users[:, 1], marker="x", s=100, label="Users")
    plt.scatter(START[0], START[1], marker="o", s=100, label="Start")
    plt.scatter(END[0], END[1], marker="s", s=100, label="End")

    plt.xlim(X_MIN, X_MAX)
    plt.ylim(Y_MIN, Y_MAX)
    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Deterministic Straight-Line UAV Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_throughput(all_user_rates, sum_rates):
    time = np.arange(L + 1)

    plt.figure()
    plt.plot(time, sum_rates)
    plt.xlabel("Time step")
    plt.ylabel("Sum throughput")
    plt.title("Sum Throughput Along Straight-Line Trajectory")
    plt.grid(True)
    plt.show()

    plt.figure()
    for k in range(K):
        plt.plot(time, all_user_rates[:, k], label=f"User {k + 1}")

    plt.xlabel("Time step")
    plt.ylabel("Individual throughput")
    plt.title("Individual User Throughput Along Straight-Line Trajectory")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    users, trajectory, all_user_rates, sum_rates = run_baseline()

    print("User positions:")
    print(users)

    print("\nTotal episode throughput:")
    print(np.sum(sum_rates))

    plot_trajectory(users, trajectory)
    plot_throughput(all_user_rates, sum_rates)
