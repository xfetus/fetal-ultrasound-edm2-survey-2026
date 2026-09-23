import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from scipy.stats import norm, chi2_contingency
from itertools import combinations
from matplotlib.patches import Patch

# ===============================================================
# HELPERS
# ===============================================================
def dprime_from_auc(auc):
    return np.sqrt(2) * norm.ppf(auc)


def bootstrap_auc(y_true, y_pred, n_boot=2000, seed=0):
    rng_b = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng_b.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def bootstrap_kappa(y1, y2, n_boot=2000, seed=0):
    rng_b = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        idx = rng_b.integers(0, len(y1), len(y1))
        if len(np.unique(y1[idx])) < 2 or len(np.unique(y2[idx])) < 2:
            continue
        ks.append(cohen_kappa_score(y1[idx], y2[idx]))
    return np.percentile(ks, [2.5, 97.5])


def bootstrap_rate(detected, n_boot=2000, seed=0):
    """Bootstrap CI for a binomial proportion."""
    rng_b = np.random.default_rng(seed)
    n = len(detected)
    if n == 0:
        return (np.nan, np.nan)
    rates = []
    for _ in range(n_boot):
        idx = rng_b.integers(0, n, n)
        rates.append(detected[idx].mean())
    return np.percentile(rates, [2.5, 97.5])


def pabak_pairwise(y1, y2):
    Po = (y1 == y2).mean()
    return 2 * Po - 1, Po


def pabak_fleiss(responses):
    n_items, n_raters = responses.shape
    pair_agreements = [(responses[:, i] == responses[:, j]).mean()
                       for i, j in combinations(range(n_raters), 2)]
    Po_avg = np.mean(pair_agreements)
    return 2 * Po_avg - 1, Po_avg


def prevalence_and_bias_index(y1, y2):
    p1, p2 = y1.mean(), y2.mean()
    return abs(p1 - p2), (p1 + p2) / 2, p1, p2


def kappa_label(k):
    if k < 0.00:  return "Poor"
    if k < 0.20:  return "Slight"
    if k < 0.40:  return "Fair"
    if k < 0.60:  return "Moderate"
    if k < 0.80:  return "Substantial"
    return "Almost perfect"


# ===============================================================
# ARTEFACT TAXONOMY (with explicit "unsure" meta-category)
# ===============================================================
ARTEFACT_TAXONOMY = [
    # code,        name,                            base_rate, detect_w, category
    ("smoothing",  "Over-smoothing",                  0.62,     0.55, "texture"),
    ("speckle",    "Speckle / noise pattern",         0.48,     0.70, "texture"),
    ("anatomy",    "Anatomical implausibility",       0.31,     1.10, "structural"),
    ("boundary",   "Boundary / edge artefact",        0.27,     0.85, "structural"),
    ("contrast",   "Contrast inconsistency",          0.22,     0.60, "photometric"),
    ("symmetry",   "Unnatural symmetry",              0.18,     0.75, "structural"),
    ("repetition", "Repeated local pattern",          0.15,     0.65, "texture"),
    ("blur",       "Localized blur",                  0.12,     0.50, "photometric"),
    ("unsure",     "Unsure — unidentified artefact",  0.20,     0.35, "meta"),
]

CODES          = [a[0] for a in ARTEFACT_TAXONOMY]
CATEGORIES     = [a[4] for a in ARTEFACT_TAXONOMY]
WEIGHTS        = np.array([a[3] for a in ARTEFACT_TAXONOMY])
UNSURE_IDX     = CODES.index("unsure")
NAMED_IDX      = [i for i, c in enumerate(CATEGORIES) if c != "meta"]
UNSURE_PROB    = ARTEFACT_TAXONOMY[UNSURE_IDX][2]
N_ARTEFACTS    = len(ARTEFACT_TAXONOMY)


# ===============================================================
# ARTEFACT ASSIGNMENT (with consistency between named and 'unsure')
# ===============================================================
def assign_artefacts(n_synth, rng):
    """
    Sample named artefacts per synthetic image, then model 'unsure'
    flags with a realistic dependency:
      - If no named artefact is present → 'unsure' is *boosted*
        (annotator sees something is off but can't name it).
      - If a named artefact is present → 'unsure' can still occur
        but at a lower rate.

    Guarantees every synthetic image has ≥1 flag (named or 'unsure').
    """
    named_probs = np.array([ARTEFACT_TAXONOMY[i][2] for i in NAMED_IDX])
    named_presence = rng.random((n_synth, len(NAMED_IDX))) < named_probs
    has_named = named_presence.any(axis=1)

    # Base 'unsure' draw
    unsure = rng.random(n_synth) < UNSURE_PROB
    # Boost 'unsure' where no named artefact was found
    boost = (~has_named) & (rng.random(n_synth) < 0.55)
    unsure = unsure | boost
    # Guarantee every image has at least one flag
    unsure = unsure | (~has_named & ~unsure)

    presence = np.zeros((n_synth, N_ARTEFACTS), dtype=bool)
    for j, i in enumerate(NAMED_IDX):
        presence[:, i] = named_presence[:, j]
    presence[:, UNSURE_IDX] = unsure
    return presence


def artefact_evidence_boost(presence):
    return presence @ WEIGHTS


# ===============================================================
# 1. Ground truth: 300 items (150 real = 1, 150 synthetic = 0)
# ===============================================================
rng = np.random.default_rng(42)
n_real, n_synth = 150, 150
ground_truth = np.array([1] * n_real + [0] * n_synth)
is_synthetic = (ground_truth == 0)

artefact_presence = assign_artefacts(n_synth, rng)
artefact_boost = np.zeros(len(ground_truth))
artefact_boost[is_synthetic] = artefact_evidence_boost(artefact_presence)

# ===============================================================
# 2. Simulate 3 raters — artefact-aware
# ===============================================================
n_raters = 3
target_aucs = [0.60, 0.78, 0.89]
rater_names = [f"Rater {i+1}" for i in range(n_raters)]

rater_preds, rater_evidence, rater_dprimes = {}, {}, {}
for name, target_auc in zip(rater_names, target_aucs):
    d = dprime_from_auc(target_auc)
    base = np.where(ground_truth == 1, d / 2, -d / 2)
    evidence = base - 0.5 * artefact_boost + rng.normal(0, 1, len(base))
    rater_preds[name] = (evidence > 0).astype(int)
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
# 3b. Inter-rater agreement
# ===============================================================
responses = np.column_stack([rater_preds[n] for n in rater_names])
counts, _ = aggregate_raters(responses)
fleiss = fleiss_kappa(counts, method="fleiss")
pabak_group, Po_group = pabak_fleiss(responses)
unanimous = np.all(responses == responses[:, [0]], axis=1).mean()

pairwise = {}
for r1, r2 in combinations(rater_names, 2):
    k = cohen_kappa_score(rater_preds[r1], rater_preds[r2])
    pab, Po = pabak_pairwise(rater_preds[r1], rater_preds[r2])
    Pidx, Bidx, p1, p2 = prevalence_and_bias_index(rater_preds[r1],
                                                    rater_preds[r2])
    k_lo, k_hi = bootstrap_kappa(rater_preds[r1], rater_preds[r2],
                                 seed=abs(hash(r1 + r2)) % 2**32)
    pairwise[f"{r1} vs {r2}"] = {
        "kappa": k, "kappa_ci": (k_lo, k_hi), "pabak": pab,
        "Po": Po, "Pindex": Pidx, "Bindex": Bidx,
    }

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

# ===============================================================
# 3c. ARTEFACT TAXONOMY — frequencies, detection, bootstrap CIs
# ===============================================================
print("\n" + "=" * 78)
print("ARTEFACT TAXONOMY (synthetic images only, n = 150)")
print("=" * 78)

artefact_freqs = artefact_presence.sum(axis=0)
artefact_prev  = artefact_presence.mean(axis=0)

detection_rate = np.zeros((n_raters, N_ARTEFACTS))
detection_ci   = np.full((n_raters, N_ARTEFACTS, 2), np.nan)
for r_idx, name in enumerate(rater_names):
    preds_synth = rater_preds[name][is_synthetic]
    detected_all = (preds_synth == 0).astype(int)
    for a_idx in range(N_ARTEFACTS):
        mask = artefact_presence[:, a_idx]
        if mask.sum() > 0:
            detection_rate[r_idx, a_idx] = detected_all[mask].mean()
            detection_ci[r_idx, a_idx] = bootstrap_rate(
                detected_all[mask], seed=r_idx * 100 + a_idx
            )

print(f"\n{'Code':<12} {'Name':<30} {'n':>5} {'%':>7} {'Cat.':<12} "
      f"{'Detect R1':>16} {'Detect R2':>16} {'Detect R3':>16}")
print("-" * 125)
for a_idx, (code, name, base, weight, cat) in enumerate(ARTEFACT_TAXONOMY):
    n = artefact_freqs[a_idx]
    pct = artefact_prev[a_idx] * 100
    cells = []
    for r_idx in range(n_raters):
        dr = detection_rate[r_idx, a_idx]
        lo, hi = detection_ci[r_idx, a_idx]
        cells.append(f"{dr:.3f} [{lo:.2f},{hi:.2f}]")
    print(f"{code:<12} {name:<30} {n:>5} {pct:>6.1f}% {cat:<12} "
          f"{cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")

# By category (excluding 'meta')
print(f"\n{'Category':<14} {'n images':>10} {'% of synthetic':>16}")
print("-" * 42)
for cat in ["texture", "structural", "photometric", "meta"]:
    idx = [i for i, c in enumerate(CATEGORIES) if c == cat]
    n_images = artefact_presence[:, idx].any(axis=1).sum()
    print(f"{cat:<14} {n_images:>10} {n_images/n_synth*100:>15.1f}%")

n_per_image = artefact_presence.sum(axis=1)
print(f"\nArtefacts per synthetic image: "
      f"mean = {n_per_image.mean():.2f}, "
      f"median = {np.median(n_per_image):.0f}, "
      f"range = [{n_per_image.min()}, {n_per_image.max()}]")

# ===============================================================
# 3d. Named vs 'unsure' vs neither — 3-way annotation confidence
# ===============================================================
named_presence  = artefact_presence[:, NAMED_IDX]
unsure_presence = artefact_presence[:, UNSURE_IDX]
has_named  = named_presence.any(axis=1)
has_unsure = unsure_presence

confidence_categories = [
    ("Named only",       has_named & ~has_unsure),
    ("'Unsure' only",    ~has_named & has_unsure),
    ("Named + 'unsure'", has_named & has_unsure),
    ("Neither (clean)",  ~has_named & ~has_unsure),
]

print("\n" + "=" * 78)
print("ANNOTATION CONFIDENCE (synthetic images, n = 150)")
print("=" * 78)
print(f"\n{'Category':<20} {'n':>5} {'%':>7} "
      f"{'Detect R1':>10} {'Detect R2':>10} {'Detect R3':>10}")
print("-" * 78)
for label, mask in confidence_categories:
    n = mask.sum()
    pct = n / n_synth * 100
    if n == 0:
        print(f"{label:<20} {n:>5} {pct:>6.1f}%         —          —          —")
        continue
    drs = []
    for name in rater_names:
        preds_synth = rater_preds[name][is_synthetic]
        drs.append((preds_synth[mask] == 0).mean())
    print(f"{label:<20} {n:>5} {pct:>6.1f}% "
          f"{drs[0]:>10.3f} {drs[1]:>10.3f} {drs[2]:>10.3f}")

# χ² test: does detection rate depend on annotation confidence category?
print("\nChi-square test: detection rate × annotation category (per rater)")
for name in rater_names:
    preds_synth = rater_preds[name][is_synthetic]
    detected = (preds_synth == 0).astype(int)
    table = []
    for _, mask in confidence_categories[:3]:   # exclude 'Neither'
        if mask.sum() == 0:
            continue
        table.append([detected[mask].sum(), mask.sum() - detected[mask].sum()])
    if len(table) >= 2:
        chi2, p, dof, _ = chi2_contingency(np.array(table))
        print(f"  {name}: χ² = {chi2:.2f}, df = {dof}, p = {p:.4f}")

# ===============================================================
# 3e. Logistic regression: independent contribution per artefact
# ===============================================================
print("\n" + "=" * 78)
print("LOGISTIC REGRESSION: P(detected | artefacts, rater)")
print("=" * 78)

# Long-format design: rows = (synthetic image, rater) pairs
X_artefacts = np.tile(artefact_presence.astype(float), (n_raters, 1))
X_rater = np.repeat(np.arange(n_raters), n_synth)
X = np.column_stack([X_artefacts, X_rater])

y = np.concatenate([
    (rater_preds[name][is_synthetic] == 0).astype(int)
    for name in rater_names
])

# Drop 'unsure' column from predictors to avoid perfect collinearity?
# No — 'unsure' carries signal; keep it. Drop one rater dummy as reference.
clf = LogisticRegression(max_iter=2000, penalty=None)
clf.fit(X, y)

feature_names = [f"{c}" for c in CODES] + [f"rater_{i+1}" for i in range(n_raters)]
coefs = clf.coef_[0]
print(f"\n{'Feature':<14} {'coef':>8} {'OR = exp(coef)':>16}")
print("-" * 40)
for fname, c in zip(feature_names, coefs):
    print(f"{fname:<14} {c:>8.3f} {np.exp(c):>16.3f}")
print("\nNote: 'rater_1' is the reference level (its coefficient is ~0).")
print("OR > 1 → presence of that artefact increases odds of detection.")

# ===============================================================
# 4. Build 6-panel figure
# ===============================================================
fig, axes = plt.subplots(1, 6, figsize=(36, 5.5))
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
for name, c, offset in zip(names, colors, [2.0, 1.0, 0.0]):
    d = rater_dprimes[name]
    mu_real, mu_synth = d / 2, -d / 2
    pdf_real  = norm.pdf(x, mu_real,  1.0)
    pdf_synth = norm.pdf(x, mu_synth, 1.0)
    scale = 0.8 / max(pdf_real.max(), pdf_synth.max())
    ax.fill_between(x, offset, offset + pdf_synth * scale,
                    color=c, alpha=0.25, hatch="///", edgecolor=c, lw=1)
    ax.fill_between(x, offset, offset + pdf_real * scale,
                    color=c, alpha=0.55, edgecolor=c, lw=1.5)
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
ax.set_xlim(-5, 5); ax.set_ylim(-0.1, 3.4); ax.set_yticks([])
ax.set_xlabel("Latent evidence  (rater's internal signal)", fontsize=12)
ax.set_title("Evidence Distributions: Real vs. Synthetic", fontsize=11)
ax.grid(axis="x", alpha=0.2)

# ---------------------------------------------------------------
# Panel 4: Cohen's κ vs PABAK per pair
# ---------------------------------------------------------------
ax = axes[3]
pair_labels = [p.replace(" vs ", "\nvs ") for p in pairwise.keys()]
kappa_vals  = [pairwise[p]["kappa"] for p in pairwise.keys()]
pabak_vals  = [pairwise[p]["pabak"] for p in pairwise.keys()]
kappa_lo    = [pairwise[p]["kappa"] - pairwise[p]["kappa_ci"][0]
               for p in pairwise.keys()]
kappa_hi    = [pairwise[p]["kappa_ci"][1] - pairwise[p]["kappa"]
               for p in pairwise.keys()]
x_pos = np.arange(len(pair_labels)); width = 0.38
bars_k = ax.bar(x_pos - width/2, kappa_vals, width, yerr=[kappa_lo, kappa_hi],
                capsize=5, color="#3498db", edgecolor="black",
                alpha=0.9, label="Cohen's κ")
bars_p = ax.bar(x_pos + width/2, pabak_vals, width,
                color="#9b59b6", edgecolor="black", alpha=0.9, label="PABAK")
ax.axhline(0.0, color="black", lw=1, ls="--", alpha=0.6, label="Chance (κ = 0)")
ax.axhline(fleiss, color="#3498db", lw=1.5, ls=":",
           label=f"Fleiss' κ = {fleiss:.2f}")
ax.axhline(pabak_group, color="#9b59b6", lw=1.5, ls=":",
           label=f"Group PABAK = {pabak_group:.2f}")
for bar, val in zip(bars_k, kappa_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val >= 0 else -0.08),
            f"{val:.2f}", ha="center",
            va="bottom" if val >= 0 else "top", fontsize=9, fontweight="bold")
for bar, val in zip(bars_p, pabak_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val >= 0 else -0.08),
            f"{val:.2f}", ha="center",
            va="bottom" if val >= 0 else "top", fontsize=9, fontweight="bold")
ax.set_xticks(x_pos); ax.set_xticklabels(pair_labels, fontsize=9)
ax.set_ylabel("Agreement (chance-corrected)", fontsize=12)
ax.set_ylim(-0.3, 1.05)
ax.set_title("Cohen's κ vs PABAK per Rater Pair", fontsize=11)
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.grid(axis="y", alpha=0.3)

# ---------------------------------------------------------------
# Panel 5: Artefact prevalence vs detection (with 'unsure' highlighted)
# ---------------------------------------------------------------
ax = axes[4]
y_pos = np.arange(N_ARTEFACTS)
order = np.argsort(-artefact_prev)
codes_sorted  = [CODES[i] for i in order]
prev_sorted   = artefact_prev[order] * 100
detect_sorted = detection_rate[:, order]

bar_colors = ["#95a5a6" if CATEGORIES[i] != "meta" else "#e67e22"
              for i in order]
ax.barh(y_pos, prev_sorted, color=bar_colors, alpha=0.5,
        edgecolor="black", height=0.7,
        label="Prevalence (grey = named, orange = 'unsure')")

for r_idx, (name, c) in enumerate(zip(rater_names, colors)):
    ax.scatter(detect_sorted[r_idx] * 100, y_pos, color=c, s=70,
               edgecolor="black", zorder=3, label=f"{name} detection")

ax.axvline(50, color="black", ls="--", lw=1, alpha=0.5)
ax.set_yticks(y_pos); ax.set_yticklabels(codes_sorted, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Percent (%)", fontsize=12)
ax.set_xlim(0, 100)
ax.set_title("Artefact Prevalence vs Detection Rate\n"
             "(synthetic images, n = 150)", fontsize=11)
ax.legend(loc="lower right", fontsize=8, ncol=2)
ax.grid(axis="x", alpha=0.3)

# ---------------------------------------------------------------
# Panel 6: Annotation confidence breakdown
# ---------------------------------------------------------------
ax = axes[5]
conf_labels = [c[0] for c in confidence_categories]
conf_counts = [c[1].sum() for c in confidence_categories]
conf_colors = ["#27ae60", "#e67e22", "#8e44ad", "#bdc3c7"]
bars = ax.bar(conf_labels, conf_counts, color=conf_colors,
              edgecolor="black", alpha=0.85)
for bar, n in zip(bars, conf_counts):
    ax.text(bar.get_x() + bar.get_width()/2, n + 1,
            f"{n}\n({n/n_synth*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Number of synthetic images", fontsize=12)
ax.set_title("Annotation Confidence\n(n = 150 synthetic)", fontsize=11)
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("per_rater_auc_300_items_3raters_with_artefacts.png",
            dpi=150, bbox_inches="tight")
plt.show()
