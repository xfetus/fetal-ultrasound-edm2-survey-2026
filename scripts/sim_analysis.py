import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from scipy.stats import norm
from itertools import combinations
from matplotlib.patches import Patch

# ===============================================================
# HELPERS
# ===============================================================
def dprime_from_auc(auc):
    """Invert AUC = Phi(d'/sqrt(2)) to get d'."""
    return np.sqrt(2) * norm.ppf(auc)


def simulate_rater_from_auc(y_true, target_auc, rng):
    """Simulate a rater's binary responses targeting a given AUC."""
    d = dprime_from_auc(target_auc)
    evidence = rng.normal(
        loc=np.where(y_true == 1, d / 2, -d / 2),
        scale=1.0
    )
    preds = (evidence > 0).astype(int)
    return preds, evidence, d


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


def bootstrap_kappa(y1, y2, n_boot=2000, seed=0):
    """Bootstrap CI for Cohen's kappa."""
    rng_b = np.random.default_rng(seed)
    n = len(y1)
    ks = []
    for _ in range(n_boot):
        idx = rng_b.integers(0, n, n)
        if len(np.unique(y1[idx])) < 2 or len(np.unique(y2[idx])) < 2:
            continue
        ks.append(cohen_kappa_score(y1[idx], y2[idx]))
    return np.percentile(ks, [2.5, 97.5])


# ===============================================================
# NEW: PABAK + prevalence / bias indices
# ===============================================================
def pabak_pairwise(y1, y2):
    """
    Prevalence-Adjusted Bias-Adjusted Kappa (Byrt et al., 1993).
        PABAK = 2 * Po - 1
    where Po = observed proportion of agreement.
    """
    Po = (y1 == y2).mean()
    return 2 * Po - 1, Po


def pabak_fleiss(responses):
    """
    Group-level PABAK for n_raters raters on binary labels.

    Two common definitions exist. We use the *pairwise-averaged* form:
        PABAK = 2 * Po_avg - 1
    where Po_avg is the mean pairwise observed agreement.
    This matches the spirit of Fleiss' κ (which is also a multi-rater
    average) and reduces to the standard pairwise PABAK when n_raters = 2.
    """
    n_items, n_raters = responses.shape
    pair_agreements = []
    for i, j in combinations(range(n_raters), 2):
        pair_agreements.append((responses[:, i] == responses[:, j]).mean())
    Po_avg = np.mean(pair_agreements)
    return 2 * Po_avg - 1, Po_avg


def prevalence_and_bias_index(y1, y2):
    """
    Byrt et al. (1993) indices explaining κ vs PABAK divergence.

    Pindex = |p1 - p2|                (difference in positive rates)
    Bindex = (p1 + p2) / 2            (average positive rate)

    Interpretation:
      Pindex ≈ 0        → no prevalence asymmetry → κ ≈ PABAK
      Pindex large      → prevalence skews κ downward → PABAK > κ
      Bindex near 0/1   → extreme base rate → both distorted
    """
    p1 = y1.mean()
    p2 = y2.mean()
    Pindex = abs(p1 - p2)
    Bindex = (p1 + p2) / 2
    return Pindex, Bindex, p1, p2


def kappa_label(k):
    """Landis & Koch (1977) qualitative benchmarks."""
    if k < 0.00:  return "Poor"
    if k < 0.20:  return "Slight"
    if k < 0.40:  return "Fair"
    if k < 0.60:  return "Moderate"
    if k < 0.80:  return "Substantial"
    return "Almost perfect"


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
target_aucs = [0.60, 0.78, 0.89]
rater_names = [f"Rater {i+1}" for i in range(n_raters)]

rater_preds, rater_evidence, rater_dprimes = {}, {}, {}
for name, target_auc in zip(rater_names, target_aucs):
    preds, evidence, d = simulate_rater_from_auc(ground_truth, target_auc, rng)
    rater_preds[name] = preds
    rater_evidence[name] = evidence
    rater_dprimes[name] = d

# ===============================================================
# 3. Per-rater AUC + bootstrap CI
# ===============================================================
results = {}
for name, preds in rater_preds.items():
    auc = roc_auc_score(ground_truth, preds)
    lo, hi = bootstrap_auc(ground_truth, preds, seed=abs(hash(name)) % 2**32)
    results[name] = {"auc": auc, "ci": (lo, hi), "preds": preds}

print(f"{'Rater':<10} {'Target':>7} {'d-prime':>8} {'Obs AUC':>9} {'95% CI':>18}")
print("-" * 58)
for name, target in zip(rater_names, target_aucs):
    r = results[name]
    print(f"{name:<10} {target:>7.2f} {rater_dprimes[name]:>8.2f} "
          f"{r['auc']:>9.3f}   [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")

# ===============================================================
# 3b. Inter-rater agreement: Cohen's κ, Fleiss' κ, PABAK
# ===============================================================
responses = np.column_stack([rater_preds[n] for n in rater_names])

# --- Fleiss' κ (all raters) ---
counts, _ = aggregate_raters(responses)
fleiss = fleiss_kappa(counts, method="fleiss")

# --- Group-level PABAK (pairwise-averaged) ---
pabak_group, Po_group = pabak_fleiss(responses)

# --- Pairwise Cohen's κ + PABAK + indices ---
pairwise = {}
for r1, r2 in combinations(rater_names, 2):
    k = cohen_kappa_score(rater_preds[r1], rater_preds[r2])
    pab, Po = pabak_pairwise(rater_preds[r1], rater_preds[r2])
    Pidx, Bidx, p1, p2 = prevalence_and_bias_index(rater_preds[r1],
                                                    rater_preds[r2])
    k_lo, k_hi = bootstrap_kappa(rater_preds[r1], rater_preds[r2],
                                 seed=abs(hash(r1 + r2)) % 2**32)
    pairwise[f"{r1} vs {r2}"] = {
        "kappa": k, "kappa_ci": (k_lo, k_hi),
        "pabak": pab, "Po": Po,
        "Pindex": Pidx, "Bindex": Bidx,
        "p1": p1, "p2": p2,
    }

# --- Unanimous agreement ---
unanimous = np.all(responses == responses[:, [0]], axis=1).mean()

# --- Print full report ---
print("\n" + "=" * 78)
print("INTER-RATER AGREEMENT")
print("=" * 78)
print(f"Fleiss' κ (all 3 raters):        {fleiss:>6.3f}   "
      f"[{kappa_label(fleiss)}]")
print(f"PABAK (group, pairwise-avg):     {pabak_group:>6.3f}   "
      f"(Po_avg = {Po_group:.3f})")
print(f"Unanimous agreement (all 3):     {unanimous*100:>5.1f}%")

print("\n" + "-" * 78)
print(f"{'Pair':<22} {'κ':>7} {'95% CI':>16} {'PABAK':>7} "
      f"{'Pindex':>8} {'Bindex':>8}")
print("-" * 78)
for pair, m in pairwise.items():
    ci = f"[{m['kappa_ci'][0]:.2f}, {m['kappa_ci'][1]:.2f}]"
    print(f"{pair:<22} {m['kappa']:>7.3f} {ci:>16} "
          f"{m['pabak']:>7.3f} {m['Pindex']:>8.3f} {m['Bindex']:>8.3f}")

print(f"\nFleiss' κ interpretation: {kappa_label(fleiss)}")
print(f"PABAK interpretation:     {kappa_label(pabak_group)}")

# ===============================================================
# 4. Build 4-panel figure
# ===============================================================
fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))

names = list(results.keys())
aucs  = [results[n]["auc"] for n in names]
los   = [results[n]["auc"] - results[n]["ci"][0] for n in names]
his   = [results[n]["ci"][1] - results[n]["auc"] for n in names]
colors = ["#e74c3c", "#f39c12", "#27ae60"]

# ---------------------------------------------------------------
# Panel 1: Per-rater AUC
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
# Panel 2: ROC curves
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
# Panel 3: Evidence distributions
# ---------------------------------------------------------------
ax = axes[2]
x = np.linspace(-5, 5, 500)
vertical_offsets = [2.0, 1.0, 0.0]

for name, c, offset in zip(names, colors, vertical_offsets):
    d = rater_dprimes[name]
    mu_real, mu_synth = d / 2, -d / 2
    pdf_real  = norm.pdf(x, mu_real,  1.0)
    pdf_synth = norm.pdf(x, mu_synth, 1.0)
    scale = 0.8 / max(pdf_real.max(), pdf_synth.max())

    ax.fill_between(x, offset, offset + pdf_synth * scale,
                    color=c, alpha=0.25, hatch="///", edgecolor=c, lw=1)
    ax.fill_between(x, offset, offset + pdf_real * scale,
                    color=c, alpha=0.55, edgecolor=c, lw=1.5)
    ax.plot([mu_synth, mu_synth], [offset, offset + pdf_synth.max() * scale],
            color=c, lw=1.5, ls=":")
    ax.plot([mu_real, mu_real], [offset, offset + pdf_real.max() * scale],
            color=c, lw=1.5, ls=":")
    ax.axvline(0, color="black", lw=0.8, alpha=0.35)
    ax.text(-4.8, offset + 0.55,
            f"{name}\nd' = {d:.2f}   AUC = {results[name]['auc']:.2f}",
            fontsize=10, fontweight="bold", color=c, va="top")
    y_arrow = offset + 0.15
    ax.annotate("", xy=(mu_real, y_arrow), xytext=(mu_synth, y_arrow),
                arrowprops=dict(arrowstyle="<->", color=c, lw=1.8))
    ax.text((mu_real + mu_synth) / 2, y_arrow + 0.05, "d'",
            ha="center", va="bottom", fontsize=9, color=c, fontweight="bold")

legend_handles = [
    Patch(facecolor="grey", alpha=0.55, edgecolor="grey", label="Real (truth = 1)"),
    Patch(facecolor="grey", alpha=0.25, edgecolor="grey", hatch="///",
          label="Synthetic (truth = 0)"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)
ax.set_xlim(-5, 5)
ax.set_ylim(-0.1, 3.4)
ax.set_yticks([])
ax.set_xlabel("Latent evidence  (rater's internal signal)", fontsize=12)
ax.set_title("Evidence Distributions: Real vs. Synthetic", fontsize=11)
ax.axvline(0, color="black", lw=1.0, alpha=0.6)
ax.grid(axis="x", alpha=0.2)

# ---------------------------------------------------------------
# Panel 4: Cohen's κ vs PABAK (grouped bar chart)
# ---------------------------------------------------------------
ax = axes[3]

# Build κ and PABAK matrices
kappa_matrix  = np.ones((n_raters, n_raters))
pabak_matrix  = np.ones((n_raters, n_raters))
for i, r1 in enumerate(rater_names):
    for j, r2 in enumerate(rater_names):
        if i != j:
            kappa_matrix[i, j] = cohen_kappa_score(rater_preds[r1],
                                                    rater_preds[r2])
            pabak_matrix[i, j], _ = pabak_pairwise(rater_preds[r1],
                                                    rater_preds[r2])

# Grouped bar chart: for each pair, show κ and PABAK side-by-side
pair_labels = [p.replace(" vs ", "\nvs ") for p in pairwise.keys()]
kappa_vals  = [pairwise[p]["kappa"] for p in pairwise.keys()]
pabak_vals  = [pairwise[p]["pabak"] for p in pairwise.keys()]
kappa_lo    = [pairwise[p]["kappa"] - pairwise[p]["kappa_ci"][0]
               for p in pairwise.keys()]
kappa_hi    = [pairwise[p]["kappa_ci"][1] - pairwise[p]["kappa"]
               for p in pairwise.keys()]

x_pos = np.arange(len(pair_labels))
width = 0.38

bars_k = ax.bar(x_pos - width/2, kappa_vals, width, yerr=[kappa_lo, kappa_hi],
                capsize=5, color="#3498db", edgecolor="black",
                alpha=0.9, label="Cohen's κ")
bars_p = ax.bar(x_pos + width/2, pabak_vals, width,
                color="#9b59b6", edgecolor="black",
                alpha=0.9, label="PABAK")

# Reference lines
ax.axhline(0.0, color="black", lw=1, ls="--", alpha=0.6, label="Chance (κ = 0)")
ax.axhline(fleiss, color="#3498db", lw=1.5, ls=":",
           label=f"Fleiss' κ = {fleiss:.2f}")
ax.axhline(pabak_group, color="#9b59b6", lw=1.5, ls=":",
           label=f"Group PABAK = {pabak_group:.2f}")

# Annotate values
for bar, val in zip(bars_k, kappa_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val >= 0 else -0.08),
            f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
            fontsize=9, fontweight="bold")
for bar, val in zip(bars_p, pabak_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val >= 0 else -0.08),
            f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
            fontsize=9, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(pair_labels, fontsize=9)
ax.set_ylabel("Agreement (chance-corrected)", fontsize=12)
ax.set_ylim(-0.3, 1.05)
ax.set_title("Cohen's κ vs PABAK per Rater Pair", fontsize=11)
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0, color="black", lw=0.5)

plt.tight_layout()
plt.savefig("per_rater_auc_300_items_3raters.png", dpi=150, bbox_inches="tight")
plt.show()
