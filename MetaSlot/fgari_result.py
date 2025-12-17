import matplotlib.pyplot as plt

# FG-ARI without TTT
baseline = 47.09

# TTT 步数
steps = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# 各个 lr 对应的 FG-ARI
results = {
    "2e-5":  [49.55, 52.13, 53.52, 54.53, 53.90, 55.78, 54.17, 53.64, 55.14, 54.91],
    "6e-5":  [49.47, 50.05, 53.42, 51.30, 52.91, 52.72, 53.69, 54.41, 53.42, 53.46],
    "8e-5":  [50.05, 52.24, 51.87, 52.91, 53.86, 54.29, 53.90, 53.02, 53.85, 53.55],
    "1e-4":  [49.21, 50.18, 51.74, 51.03, 52.56, 52.10, 52.65, 53.95, 51.72, 52.77],
    "1.2e-4":[49.21, 48.32, 49.40, 50.63, 49.76, 50.63, 50.33, 50.59, 50.88, 50.62],
    "1e-5":  [50.03, 50.97, 53.34, 53.51, 52.72, 54.14, 55.90, 53.72, 55.10, 55.68],
    "8e-6":  [49.20, 50.41, 52.69, 52.70, 53.08, 54.75, 54.03, 54.96, 54.26, 53.76],
    "6e-6":  [48.85, 50.15, 52.01, 53.16, 53.21, 53.82, 53.18, 54.45, 53.96, 54.79],
    "4e-6":  [48.55, 49.80, 51.36, 51.88, 52.71, 52.64, 51.04, 51.91, 51.14, 53.52],
    "2e-6":  [49.07, 50.23, 49.62, 51.37, 51.57, 51.87, 51.52, 52.73, 52.41, 52.73],
}

plt.figure(figsize=(10, 6))

# 画每条 lr 曲线
for lr, vals in results.items():
    plt.plot(steps, vals, marker="o", label=f"lr={lr}")

# baseline 水平线
plt.axhline(baseline, linestyle="--", linewidth=1, label=f"no TTT ({baseline})")

plt.xlabel("TTT step")
plt.ylabel("FG-ARI")
plt.title("FG-ARI vs TTT Steps for Different Learning Rates")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

plt.savefig("fg_ari_vs_ttt_steps.png", dpi=300)

plt.show()
