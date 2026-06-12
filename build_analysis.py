"""Анализ A/B-теста: push-уведомление за 24ч до конца trial → конверсия в paid.

DGP с зашитым известным эффектом + полный пайплайн:
- Power analysis
- A/B на уровне пользователя: z-test, bootstrap CI
- Регрессия с ковариатами
- Synthetic Control: раскатка фичи на одну страну (Германия)

Сохраняет 3 PNG в ./images/ для использования в README.md.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

SEED = 42
rng = np.random.default_rng(SEED)
IMG = "images"
os.makedirs(IMG, exist_ok=True)

# =====================================================================
# 1. DGP - синтетика с реалистичной для маленького VPN бизнеса картиной
# =====================================================================
N_PER_ARM = 2000           # ~4000 пользователей в trial всего, набирали 2 недели
TRIAL_DAYS = 7
BASE_CONVERSION = 0.235    # типичный VPN trial-to-paid ~23-26%
TRUE_LIFT_PP = 0.050       # ground truth: +5 п.п. от пуша

# Признаки пользователя
def gen_users(n, country, group, base_rate, lift_pp):
    """latent quality + observable covariates, conversion ~ Bernoulli."""
    age = rng.integers(18, 60, n)
    device = rng.choice(["iOS", "Android"], n, p=[0.55, 0.45])
    paid_ads = rng.choice([0, 1], n, p=[0.5, 0.5])  # пришёл с платной рекламы?
    sessions_in_trial = rng.poisson(5, n).clip(0, 25)  # активность в trial

    # Латентное предрасположение к конверсии
    latent = rng.normal(0, 1, n)
    # Probit-like модель
    logit_p = (
        ss.norm.ppf(base_rate)
        + 0.18 * latent
        + 0.05 * (sessions_in_trial - 5) / 3
        - 0.10 * paid_ads  # привлеченные платно конвертируются хуже
        + (1 if group == "treatment" else 0) * ss.norm.ppf(base_rate + lift_pp) - (1 if group == "treatment" else 0) * ss.norm.ppf(base_rate)
    )
    p_conv = ss.norm.cdf(logit_p)
    converted = (rng.random(n) < p_conv).astype(int)

    return pd.DataFrame({
        "user_id": np.arange(n),
        "country": country,
        "group": group,
        "age": age,
        "device": device,
        "paid_ads": paid_ads,
        "sessions_in_trial": sessions_in_trial,
        "converted": converted,
    })


countries = ["DE", "US", "BR", "IN", "FR", "PL", "TR", "MX"]
country_weights = [0.18, 0.22, 0.10, 0.12, 0.10, 0.08, 0.10, 0.10]

frames = []
for grp in ["control", "treatment"]:
    countries_for_arm = rng.choice(countries, N_PER_ARM, p=country_weights)
    df = gen_users(
        N_PER_ARM, countries_for_arm, grp,
        base_rate=BASE_CONVERSION,
        lift_pp=TRUE_LIFT_PP,
    )
    df["user_id"] += {"control": 0, "treatment": 100_000}[grp]
    frames.append(df)
df = pd.concat(frames, ignore_index=True)

print(f"N total = {len(df):,}")
print(f"control conv:   {df[df.group=='control']['converted'].mean():.4f}")
print(f"treatment conv: {df[df.group=='treatment']['converted'].mean():.4f}")
print(f"observed Δ:     {df[df.group=='treatment']['converted'].mean() - df[df.group=='control']['converted'].mean():+.4f}")

# =====================================================================
# 2. Power analysis - сохраняем picture 1
# =====================================================================
analysis = NormalIndPower()
deltas = np.linspace(0.005, 0.10, 60)
powers = [analysis.power(
    effect_size=proportion_effectsize(BASE_CONVERSION + d, BASE_CONVERSION),
    nobs1=N_PER_ARM, alpha=0.05, ratio=1.0
) for d in deltas]
powers = np.array(powers)
mde = deltas[np.searchsorted(powers, 0.8)]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(deltas * 100, powers, lw=2.5, color="#1f4e79")
ax.axhline(0.8, color="red", ls="--", label="power = 0.8")
ax.axvline(mde * 100, color="green", ls=":",
           label=f"MDE ≈ {mde*100:.2f} п.п.")
ax.axvline(TRUE_LIFT_PP * 100, color="darkorange", ls=":",
           label=f"ожидаемый эффект ≈ {TRUE_LIFT_PP*100:.1f} п.п.")
ax.set_title(f"Power vs величина эффекта (n = {N_PER_ARM} на ветку)")
ax.set_xlabel("Δ conversion, п.п.")
ax.set_ylabel("Power")
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG}/power_curve.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\n=> MDE at power 0.8: {mde*100:.2f} п.п.")

# =====================================================================
# 3. A/B-анализ - SRM, balance, z-test, bootstrap CI
# =====================================================================
# SRM
counts = df["group"].value_counts().reindex(["control", "treatment"]).values
srm_chi2, srm_p = ss.chisquare(counts, [len(df)/2, len(df)/2])
print(f"\nSRM χ² = {srm_chi2:.3f}, p = {srm_p:.3f}")

# Balance
print("\nКовариатный баланс:")
for col in ["age", "sessions_in_trial", "paid_ads"]:
    a = df[df.group=="control"][col]
    b = df[df.group=="treatment"][col]
    t, p = ss.ttest_ind(a, b, equal_var=False)
    print(f"  {col:25s}: p = {p:.3f}")

# z-test для пропорций
n_t = (df.group == "treatment").sum()
n_c = (df.group == "control").sum()
k_t = df[df.group == "treatment"]["converted"].sum()
k_c = df[df.group == "control"]["converted"].sum()
p_t = k_t / n_t
p_c = k_c / n_c
delta = p_t - p_c

z_stat, z_p = proportions_ztest([k_t, k_c], [n_t, n_c])
print(f"\nz-test: z = {z_stat:.3f}, p = {z_p:.4f}")
print(f"Δ = {delta:+.4f} ({delta*100:+.2f} п.п.)")

# Bootstrap CI на Δ
B = 4000
boots = np.empty(B)
ctrl_conv = df[df.group == "control"]["converted"].values
treat_conv = df[df.group == "treatment"]["converted"].values
for i in range(B):
    a = rng.choice(ctrl_conv, len(ctrl_conv), replace=True)
    b = rng.choice(treat_conv, len(treat_conv), replace=True)
    boots[i] = b.mean() - a.mean()
ci_low, ci_hi = np.percentile(boots, [2.5, 97.5])
print(f"Bootstrap 95% CI: [{ci_low:+.4f}, {ci_hi:+.4f}]")

# Регрессия с ковариатами
df_reg = pd.get_dummies(df, columns=["device"], drop_first=True)
df_reg["treatment"] = (df_reg["group"] == "treatment").astype(int)
model = sm.OLS.from_formula(
    "converted ~ treatment + age + paid_ads + sessions_in_trial + C(country)",
    data=df_reg
).fit(cov_type="HC3")  # heteroskedasticity-robust
ate_reg = model.params["treatment"]
ci_reg = model.conf_int().loc["treatment"].values
print(f"\nРегрессия (с ковариатами): ATE = {ate_reg:+.4f}, CI [{ci_reg[0]:+.4f}, {ci_reg[1]:+.4f}]")

# Бизнес-метрика: дополнительный revenue
ARPU_MONTHLY = 4.99
ADDITIONAL_CONVERSIONS_PER_1000 = delta * 1000
EXTRA_MRR_PER_1000_USERS = ADDITIONAL_CONVERSIONS_PER_1000 * ARPU_MONTHLY
print(f"\nДополнительная MRR на каждые 1000 пользователей в trial: ${EXTRA_MRR_PER_1000_USERS:.0f}")

# Картинка 2: бары + bootstrap CI
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
bars = ax.bar(["control\n(no push)", "treatment\n(push -24h)"],
              [p_c * 100, p_t * 100],
              color=["#8a8a8a", "#1f6f3e"])
ax.set_ylabel("Conversion rate, %")
ax.set_title("Trial → Paid conversion")
ax.set_ylim(0, max(p_c, p_t) * 100 * 1.3)
for i, (val, n) in enumerate([(p_c * 100, n_c), (p_t * 100, n_t)]):
    ax.text(i, val + 0.7, f"{val:.2f}%\nn = {n:,}", ha="center", fontsize=11)

ax = axes[1]
ax.hist(boots * 100, bins=40, color="#4472C4", alpha=0.85, edgecolor="white")
ax.axvline(delta * 100, color="red", ls="--", lw=2,
           label=f"наблюдаемое Δ = {delta*100:+.2f} п.п.")
ax.axvspan(ci_low * 100, ci_hi * 100, alpha=0.18, color="orange",
           label=f"95% CI [{ci_low*100:+.2f}, {ci_hi*100:+.2f}] п.п.")
ax.axvline(0, color="black", lw=0.7)
ax.set_xlabel("Δ conversion, п.п.")
ax.set_title("Bootstrap-распределение Δ")
ax.legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(f"{IMG}/ab_results.png", dpi=120, bbox_inches="tight")
plt.close()

# =====================================================================
# 4. Synthetic Control: раскатка push на Германию как pilot
# =====================================================================
# Сценарий: после A/B решили катить фичу на одну страну (DE) с известной даты,
# а в остальных странах оставили старый flow. Смотрим эффект на недельный
# trial-to-paid conversion rate по странам.

N_WEEKS = 16
T0 = 10  # раскатка с 10-й недели
TREATED = "DE"
DONORS = [c for c in countries if c != TREATED]

weeks = np.arange(N_WEEKS)
# Базовые intercepts по странам + общие факторы
factor1 = 0.012 * np.sin(2 * np.pi * weeks / 8)   # квартальная сезонность
factor2 = 0.0006 * (weeks - N_WEEKS / 2)          # лёгкий growth trend

country_params = {
    "DE": dict(intercept=BASE_CONVERSION, w1=1.0, w2=1.0),
    "US": dict(intercept=BASE_CONVERSION + 0.01, w1=0.9, w2=1.1),
    "BR": dict(intercept=BASE_CONVERSION - 0.02, w1=1.1, w2=0.9),
    "IN": dict(intercept=BASE_CONVERSION - 0.04, w1=0.7, w2=1.3),
    "FR": dict(intercept=BASE_CONVERSION + 0.005, w1=1.0, w2=1.0),
    "PL": dict(intercept=BASE_CONVERSION - 0.01, w1=0.8, w2=1.2),
    "TR": dict(intercept=BASE_CONVERSION - 0.02, w1=1.2, w2=0.8),
    "MX": dict(intercept=BASE_CONVERSION - 0.015, w1=1.0, w2=1.0),
}

panel = {}
for c, p in country_params.items():
    series = p["intercept"] + p["w1"] * factor1 + p["w2"] * factor2 + rng.normal(0, 0.005, N_WEEKS)
    panel[c] = series

panel[TREATED] = panel[TREATED].copy()
panel[TREATED][T0:] += TRUE_LIFT_PP
panel_df = pd.DataFrame(panel, index=weeks)

# SC подбор весов
Y1 = panel_df[TREATED].values
Y0 = panel_df[DONORS].values
pre_mask = weeks < T0


def loss(w):
    return np.sum((Y1[pre_mask] - Y0[pre_mask] @ w) ** 2)


cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
bnds = [(0, 1)] * len(DONORS)
res = minimize(loss, np.ones(len(DONORS)) / len(DONORS),
               method="SLSQP", bounds=bnds, constraints=cons)
w = res.x
synth = Y0 @ w
gap = Y1 - synth
sc_effect = gap[T0:].mean()

print(f"\nSynthetic Control:")
print(f"  Веса доноров: {dict(zip(DONORS, w.round(3)))}")
print(f"  Средний gap пост-периода: {sc_effect:+.4f}  (ground truth = {TRUE_LIFT_PP:+.4f})")

# Placebo-in-space
placebos = []
all_countries = list(panel_df.columns)
for fake_treated in DONORS:
    fake_donors = [c for c in all_countries if c != fake_treated]
    Y1_f = panel_df[fake_treated].values
    Y0_f = panel_df[fake_donors].values

    def loss_f(w_):
        return np.sum((Y1_f[pre_mask] - Y0_f[pre_mask] @ w_) ** 2)
    res_f = minimize(loss_f, np.ones(len(fake_donors)) / len(fake_donors),
                     method="SLSQP", bounds=[(0, 1)] * len(fake_donors), constraints=cons)
    w_f = res_f.x
    gap_f = Y1_f - Y0_f @ w_f
    placebos.append(gap_f[T0:].mean())

placebos = np.array(placebos)
p_placebo = (np.abs(np.concatenate([[sc_effect], placebos])) >= abs(sc_effect)).mean()
print(f"  Placebo-in-space: pseudo-p = {p_placebo:.3f}")

# Картинка 3: SC + placebo
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5),
                          gridspec_kw={"width_ratios": [2.2, 1]})

ax = axes[0]
ax.plot(weeks, Y1 * 100, color="#c0392b", lw=2.5, label=f"факт: {TREATED}")
ax.plot(weeks, synth * 100, color="#2c3e50", lw=2, ls="--", label="синтетический контроль")
ax.axvline(T0, color="red", ls=":", alpha=0.7, label=f"раскатка push на {TREATED}")
ax.fill_between(weeks, synth * 100, Y1 * 100, where=(weeks >= T0),
                color="green", alpha=0.15)
ax.set_xlabel("неделя")
ax.set_ylabel("trial → paid, %")
ax.set_title("Synthetic Control: pilot push в Германии")
ax.legend(loc="upper left", fontsize=10)

ax = axes[1]
ax.hist(placebos * 100, bins=10, color="#7f8c8d", alpha=0.7, edgecolor="white")
ax.axvline(sc_effect * 100, color="red", lw=2.5,
           label=f"DE = {sc_effect*100:+.2f}")
ax.axvline(0, color="black", lw=0.7)
ax.set_xlabel("avg post-эффект, п.п.")
ax.set_title(f"Placebo-in-space\np = {p_placebo:.3f}")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{IMG}/synthetic_control.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nГотово. PNG сохранены в images/")
print("Финальные числа для README:")
print(f"  Δ conversion: {delta*100:+.2f} п.п.")
print(f"  95% CI: [{ci_low*100:+.2f}, {ci_hi*100:+.2f}] п.п.")
print(f"  p-value (z-test): {z_p:.4f}")
print(f"  Регрессия ATE: {ate_reg*100:+.2f} п.п., CI [{ci_reg[0]*100:+.2f}, {ci_reg[1]*100:+.2f}]")
print(f"  Synthetic Control: {sc_effect*100:+.2f} п.п. (placebo-p = {p_placebo:.3f})")
