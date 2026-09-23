import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import norm

# ===============================================================
# HELPERS
# ===============================================================
def dprime_from_auc(auc):
    """Invert AUC = Phi(d'/sqrt(2)) to get d'."""
    return np.sqrt(2) * norm.ppf(auc)


def simulate_rater_from_auc(y_true, target_auc, rng):
    """
    Simulate a rater's binary responses so that the expected AUC
    is approximately `target_auc`.

    Returns:
        preds    : binary responses (1 = "Real", 0 = "Synthetic")
        evidence : the latent continuous evidence (for plotting)
        d        : the d' used internally
    """
    d = dprime_from_auc(target_auc)
    evidence = rng.normal(
        loc=np.where(y_true == 1, d / 2, -d / 2),
        scale=1.0
    )
    preds = (evidence > 0).astype(int)
    return preds, evidence, d


# ===============================================================
# 1. Ground truth: 300 items (150 real = 1, 150 synthetic = 0)
# ===============================================================
rng = np.random.default_rng(42)
n_real, n_synth = 150, 150
ground_truth = np.array([1] * n_real + [0] * n_synth)

# ===============================================================
# 2. Simulate 3 raters with TARGET AUCs
# ===============================================================
n_raters = 3
target_aucs = [0.60, 0.78, 0.89]           # low, medium, high
rater_names = [f"Rater {i+1}" for i in range(n_raters)]

rater_preds = {}
rater_evidence = {}
rater_dprimes = {}
for name, target_auc in zip(rater_names, target_aucs):
    preds, evidence, d = simulate_rater_from_auc(ground_truth, target_auc, rng)
    rater_preds[name] = preds
    rater_evidence[name] = evidence
    rater_dprimes[name] = d

# ===============================================================
# 3. Per-rater AUC + bootstrap 95% CI
# ===============================================================
def bootstrap_auc(y_true, y_pred, n_boot=2000, seed=0):
    rng_b = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng_b.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return np.percentile(aucs, [2.5, 97.5])


results = {}
for name, preds in rater_preds.items():
    auc = roc_auc_score(ground_truth, preds)
    lo, hi = bootstrap_auc(ground_truth, preds, seed=hash(name) % 2**32)
    results[name] = {"auc": auc, "ci": (lo, hi), "preds": preds}

print(f"{'Rater':<10} {'Target':>7} {'d-prime':>8} {'Obs AUC':>9} {'95% CI':>18}")
print("-" * 58)
for name, target in zip(rater_names, target_aucs):
    r = results[name]
    print(f"{name:<10} {target:>7.2f} {rater_dprimes[name]:>8.2f} "
          f"{r['auc']:>9.3f}   [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")

# ===============================================================
# 4. Build 3-panel figure
# ===============================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

names = list(results.keys())
aucs  = [results[n]["auc"] for n in names]
los   = [results[n]["auc"] - results[n]["ci"][0] for n in names]
his   = [results[n]["ci"][1] - results[n]["auc"] for n in names]

colors = ["#e74c3c", "#f39c12", "#27ae60"]   # red / orange / green

# ---------------------------------------------------------------
# Panel 1: Per-rater AUC bar chart with error bars
# ---------------------------------------------------------------
ax = axes[0]
bars = ax.bar(names, aucs, yerr=[los, his], capsize=8,
              color=colors, edgecolor="black", alpha=0.9, width=0.55)
ax.axhline(0.5, color="black", linestyle="--", lw=1.2,
           label="Chance (AUC = 0.5)")
ax.set_ylabel("AUC", fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_title("Per-Rater AUC\n(300 items, 95% CI via bootstrap)", fontsize=11)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)

for bar, auc, tgt, d in zip(bars, aucs, target_aucs,
                             [rater_dprimes[n] for n in names]):
    ax.text(bar.get_x() + bar.get_width()/2, auc + 0.03,
            f"AUC {auc:.2f}\nd' {d:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

# ---------------------------------------------------------------
# Panel 2: ROC curves per rater
# ---------------------------------------------------------------
ax = axes[1]
for name, c in zip(names, colors):
    fpr, tpr, _ = roc_curve(ground_truth, results[name]["preds"])
    ax.plot(fpr, tpr, color=c, lw=2.5,
            label=f"{name} (AUC = {results[name]['auc']:.2f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves per Rater", fontsize=11)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

# ---------------------------------------------------------------
# Panel 3: Evidence distributions (real vs synthetic) per rater
# ---------------------------------------------------------------
ax = axes[2]

# Shared x-range across the three sub-distributions
x_min, x_max = -5, 5
x = np.linspace(x_min, x_max, 500)

# Offset each rater's distribution vertically so they don't overlap
vertical_offsets = [2.0, 1.0, 0.0]   # Rater 3 on top, Rater 1 at bottom

for name, c, offset in zip(names, colors, vertical_offsets):
    d = rater_dprimes[name]
    mu_real, mu_synth = d / 2, -d / 2
    sigma = 1.0

    pdf_real  = norm.pdf(x, mu_real,  sigma)
    pdf_synth = norm.pdf(x, mu_synth, sigma)

    # Normalize peak height to ~0.8 so panels look tidy
    scale = 0.8 / max(pdf_real.max(), pdf_synth.max())

    # Synthetic (left, hatched)
    ax.fill_between(x, offset, offset + pdf_synth * scale,
                    color=c, alpha=0.25, hatch="///", edgecolor=c, lw=1)
    # Real (right, solid)
    ax.fill_between(x, offset, offset + pdf_real * scale,
                    color=c, alpha=0.55, edgecolor=c, lw=1.5)

    # Mean markers
    ax.plot([mu_synth, mu_synth], [offset, offset + pdf_synth.max() * scale],
            color=c, lw=1.5, ls=":")
    ax.plot([mu_real, mu_real], [offset, offset + pdf_real.max() * scale],
            color=c, lw=1.5, ls=":")

    # Decision criterion at 0
    ax.axvline(0, color="black", lw=0.8, alpha=0.35)

    # Labels
    ax.text(x_min + 0.2, offset + 0.55,
            f"{name}\nd' = {d:.2f}   AUC = {results[name]['auc']:.2f}",
            fontsize=10, fontweight="bold", color=c, va="top")

    # d' arrow between the two means
    y_arrow = offset + 0.15
    ax.annotate("", xy=(mu_real, y_arrow), xytext=(mu_synth, y_arrow),
                arrowprops=dict(arrowstyle="<->", color=c, lw=1.8))
    ax.text((mu_real + mu_synth) / 2, y_arrow + 0.05, "d'",
            ha="center", va="bottom", fontsize=9, color=c, fontweight="bold")

# Legend for the two distribution types (drawn once, top-right)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="grey", alpha=0.55, edgecolor="grey", label="Real (truth = 1)"),
    Patch(facecolor="grey", alpha=0.25, edgecolor="grey", hatch="///",
          label="Synthetic (truth = 0)"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.1, 3.4)
ax.set_yticks([])                      # hide y ticks — vertical axis is arbitrary
ax.set_xlabel("Latent evidence  (rater's internal signal)", fontsize=12)
ax.set_title("Evidence Distributions: Real vs. Synthetic", fontsize=11)
ax.axvline(0, color="black", lw=1.0, alpha=0.6)
ax.text(0.05, 3.28, "decision criterion (evidence > 0 → 'Real')",
        fontsize=8, color="black", alpha=0.7)
ax.grid(axis="x", alpha=0.2)

plt.tight_layout()
plt.savefig("per_rater_auc_300_items_3raters.png", dpi=150, bbox_inches="tight")
plt.show()
