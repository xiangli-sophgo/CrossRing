import numpy as np
import matplotlib.pyplot as plt

# Figure 1: Average and worst hop count vs N
Ns = np.arange(3, 11)  # 3 .. 10
avg_mesh = 2 * (Ns - 1) / 3
avg_cr = Ns / 2
worst_mesh = 2 * (Ns - 1)
worst_cr = Ns  # for even N -- approximation

plt.figure()
plt.plot(Ns, avg_mesh, marker="o", label="Mesh – Avg")
plt.plot(Ns, avg_cr, marker="s", label="CrossRing – Avg")
plt.plot(Ns, worst_mesh, marker="^", label="Mesh – Worst")
plt.plot(Ns, worst_cr, marker="v", label="CrossRing – Worst")
plt.title("Figure 1  •  Hop Count vs Network Size")
plt.xlabel("Network size N (NxN nodes)")
plt.ylabel("Hop count (hops)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Figure 2: Latency vs injection rate for N=8
N = 8
H_mesh = 2 * (N - 1) / 3
H_cr = N / 2
λ_vals = np.linspace(0.0, 0.24, 13)  # up to Mesh saturation 0.25
ρ_mesh = 4 * λ_vals  # ρ_mesh = N*λ/2  (N=8)
ρ_cr = 2 * λ_vals  # ρ_cr   = N*λ/4  (N=8)


def md1_latency(H, ρ):
    return H * (1 + ρ / (2 * (1 - ρ)))


lat_mesh = md1_latency(H_mesh, ρ_mesh)
lat_cr = md1_latency(H_cr, ρ_cr)

plt.figure()
plt.plot(λ_vals, lat_mesh, marker="o", label="Mesh (N=8)")
plt.plot(λ_vals, lat_cr, marker="s", label="CrossRing (N=8)")
plt.axvline(x=0.25, linestyle="--", label="Mesh λ_sat≈0.25")
plt.title("Figure 2  •  Average Latency vs Injection Rate (N=9)")
plt.xlabel("Per-node injection rate λ (pkts/cycle)")
plt.ylabel("Average latency (cycles)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Figure 3: Saturation injection rate vs N
λ_sat_mesh = 2 / Ns
λ_sat_cr = 4 / Ns

plt.figure()
plt.plot(Ns, λ_sat_mesh, marker="o", label="Mesh λ_sat")
plt.plot(Ns, λ_sat_cr, marker="s", label="CrossRing λ_sat")
plt.title("Figure 3  •  Saturation Injection Rate vs Network Size")
plt.xlabel("Network size N (NxN nodes)")
plt.ylabel("λ_sat  (pkts/cycle/node)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
