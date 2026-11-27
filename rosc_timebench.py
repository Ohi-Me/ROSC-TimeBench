"""ROSC-TimeBench: how much does elapsed resuscitation time contribute to event-level versus
cycle-level (landmark) prediction of return of spontaneous circulation (ROSC)?

Data (public, CC BY 4.0; downloaded automatically):
  - Zenodo doi:10.5281/zenodo.20162201: OHCA patients resuscitated in an emergency department (time origin: start of ED CPR)
  - Mendeley Data doi:10.17632/mbwykx585m.1: CAAM trial chest-compression-fraction sub-study (time origin: ALS arrival)

Definitions (horizon H = 30 min, 2-minute cycles):
  cycle k covers (2(k-1), 2k] minutes; landmark L_k = 2(k-1) minutes is the elapsed time available when cycle k starts.
  A patient contributes cycles 1..C, C = ceil(min(duration, H) / 2). ROSC after H counts as no ROSC within H.
  Cycle level  : one prediction per patient-cycle, made at L_k, for ROSC during cycle k (the next 2 minutes).
  Event level  : one prediction per patient, at the patient's last cycle C, for ROSC during that cycle (= ROSC within H).
                 Its time input L_C is fixed by the completed resuscitation duration, known only once it has ended.
Models: L2-penalised logistic regression (C = 0.3); 10-fold cross-validation with folds assigned by patient, so all cycles
of a patient are in the same fold; standardisation and mean imputation are fitted on the training fold only.
Uncertainty: patient-level bootstrap (all cycles of a sampled patient are kept together) of the out-of-fold predictions,
2000 resamples, percentile 95% CIs; differences are paired (same resamples). Model-fitting variability is shown separately
by repeating the cross-validation with 20 different fold splits.

Usage:  python rosc_timebench.py [--data DIR] [--out results.csv]
"""
import argparse, os, glob, sys, time
import numpy as np, pandas as pd, requests, pyreadstat
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression as LR

ZENODO = "https://zenodo.org/records/20162201/files/Research_data.sav?download=1"
MENDELEY = "https://data.mendeley.com/public-api/datasets/mbwykx585m"
H = {"User-Agent": "Mozilla/5.0"}  # used for Mendeley Data only (it rejects some non-browser agents; Zenodo rejects this one)
B = 2000


# ---------------------------------------------------------------- data
def get(url, tries=5, headers=None):
    for i in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=120)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


def download(data):
    os.makedirs(data, exist_ok=True)
    z = os.path.join(data, "zenodo_20162201.sav")
    if not os.path.exists(z):
        open(z, "wb").write(get(ZENODO).content)  # Zenodo: default user agent
    c = glob.glob(os.path.join(data, "mbwykx585m_*.dta"))
    if not c:
        try:
            meta = get(MENDELEY, headers=H).json()
            f = [f for f in meta["files"] if f["filename"].lower().endswith(".dta")][0]
            p = os.path.join(data, "mbwykx585m_" + f["filename"].replace(" ", "_"))
            open(p, "wb").write(get(f["content_details"]["download_url"], headers=H).content)
            c = [p]
        except (requests.RequestException, ValueError, KeyError, IndexError):
            sys.exit("Could not reach Mendeley Data. Download the .dta file from https://data.mendeley.com/datasets/mbwykx585m/1 "
                     "and save it in '%s' with a name starting 'mbwykx585m_' (e.g. mbwykx585m_cleandataset_anonymous.dta), then rerun." % data)
    return z, c[0]


def load(data):
    z, c = download(data)
    df, meta = pyreadstat.read_sav(z)
    assert meta.variable_value_labels.get("ROSC", {}).get(1.0) == "Yes", "unexpected ROSC coding in the Zenodo file"
    Z = dict(t=df.ED_CPR_duration.values.astype(float), y=df.ROSC.values.astype(float),
             X=df[["AGE", "GENDER", "Location", "Bystander_CPR", "Shockable_Rythm", "Etiology_of_arrest"]].values.astype(float))
    d = pd.read_stata(c, convert_categoricals=False)
    d = d[d.id_caam.notna()]
    als = (d.stopRCP - d.time_ALS).dt.total_seconds() / 60  # minutes from ALS arrival to end of CPR
    rt = (d.firstROSC - d.time_ALS).dt.total_seconds() / 60  # minutes from ALS arrival to first ROSC
    als = als.where(als > -600, als + 1440)  # clock times past midnight
    rt = rt.where(rt > -600, rt + 1440)
    y = d.firstROSC.notna().astype(float)
    C = dict(t=rt.where(y == 1, als).values, y=y.values,
             X=d[["age", "gender", "witness", "CPRbyWITNESS", "shockable"]].values.astype(float))
    return Z, C


# ---------------------------------------------------------------- design
def build(t, y, X, cap):
    ok = np.isfinite(t) & np.isfinite(y) & (t > 0)
    t, y, X = t[ok], y[ok], X[ok]
    yy = ((y == 1) & (t <= cap)).astype(int)                      # ROSC within the horizon
    C = np.ceil(np.minimum(t, cap) / 2).astype(int).clip(1)       # cycles per patient
    pid = np.repeat(np.arange(len(t)), C)
    k = np.concatenate([np.arange(1, q + 1) for q in C])
    L = 2.0 * (k - 1)                                             # landmark: elapsed minutes when the cycle starts
    e = np.concatenate([np.r_[np.zeros(q - 1), v] for q, v in zip(C, yy)]).astype(int)  # ROSC during this cycle
    last = np.r_[np.cumsum(C) - 1]                                # row of each patient's last cycle
    return dict(n=len(t), X=X, y=yy, C=C, pid=pid, L=L, e=e, last=last, excluded=int((~ok).sum()))


def tf(L):
    return np.c_[L / 10, (L / 10) ** 2]


def scale(Xtr, Xte):
    m = np.nanmean(Xtr, 0); s = np.nanstd(Xtr, 0) + 1e-9
    f = lambda A: np.nan_to_num((A - m) / s)                      # missing -> training mean
    return f(Xtr), f(Xte)


def crossfit(D, seed):
    """Out-of-fold predictions for every model; folds by patient."""
    n, pid, L, X = D["n"], D["pid"], D["L"], D["X"]
    fold = np.random.default_rng(seed).permutation(n) % 10
    out = {k: np.zeros(n) for k in ("ev_full", "ev_base", "ev_time")}
    out.update({k: np.zeros(len(L)) for k in ("lm_full", "lm_time", "transfer")})
    Llast = L[D["last"]]
    for q in range(10):
        tr, te = fold != q, fold == q
        TR, TE = fold[pid] != q, fold[pid] == q
        Xtr, Xte = scale(X[tr], X[te])
        # event level: trained and scored on last-cycle rows only
        fit = lambda A, Bm: LR(C=.3, max_iter=1000).fit(A, D["y"][tr]).predict_proba(Bm)[:, 1]
        m_full = LR(C=.3, max_iter=1000).fit(np.c_[Xtr, tf(Llast[tr])], D["y"][tr])
        out["ev_full"][te] = m_full.predict_proba(np.c_[Xte, tf(Llast[te])])[:, 1]
        out["ev_base"][te] = fit(Xtr, Xte)
        out["ev_time"][te] = fit(tf(Llast[tr]), tf(Llast[te]))
        # event-level model applied at every cycle of the held-out patients (same target: ROSC in the next 2 min)
        _, Xc_te = scale(X[tr], X[pid[TE]])
        out["transfer"][TE] = m_full.predict_proba(np.c_[Xc_te, tf(L[TE])])[:, 1]
        # cycle level (landmark): trained on all cycles of training patients
        Xr_tr, Xr_te = scale(X[pid[TR]], X[pid[TE]])
        out["lm_full"][TE] = LR(C=.3, max_iter=1000).fit(np.c_[Xr_tr, tf(L[TR])], D["e"][TR]).predict_proba(np.c_[Xr_te, tf(L[TE])])[:, 1]
        out["lm_time"][TE] = LR(C=.3, max_iter=1000).fit(tf(L[TR]), D["e"][TR]).predict_proba(tf(L[TE]))[:, 1]
    return out


def cal_slope(p, y):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return LR(C=1e6, max_iter=1000).fit(lp[:, None], y).coef_[0, 0]


# ---------------------------------------------------------------- metrics with patient-level bootstrap
def metrics(D, P, rows_p, rows_c):
    y, e, last = D["y"], D["e"], D["last"]
    Llast = D["L"][last]
    v = {
        "event: elapsed time alone, fixed direction (shorter = ROSC)": auc(y[rows_p], -Llast[rows_p]),
        "event: elapsed-time model (CV)": auc(y[rows_p], P["ev_time"][rows_p]),
        "event: baseline-only model (CV)": auc(y[rows_p], P["ev_base"][rows_p]),
        "event: baseline + time model (CV)": auc(y[rows_p], P["ev_full"][rows_p]),
        "cycle: elapsed time alone, fixed direction (shorter = ROSC)": auc(e[rows_c], -D["L"][rows_c]),
        "cycle: elapsed-time landmark model (CV)": auc(e[rows_c], P["lm_time"][rows_c]),
        "cycle: baseline + time landmark model (CV)": auc(e[rows_c], P["lm_full"][rows_c]),
        "cycle: event-level model applied at every cycle": auc(e[rows_c], P["transfer"][rows_c]),
        "cycle: observed ROSC rate per cycle": e[rows_c].mean(),
        "cycle: landmark model, mean predicted probability": P["lm_full"][rows_c].mean(),
        "cycle: landmark model, calibration slope": cal_slope(P["lm_full"][rows_c], e[rows_c]),
        "cycle: landmark model, Brier score": ((P["lm_full"][rows_c] - e[rows_c]) ** 2).mean(),
        "cycle: event-level model at every cycle, mean predicted probability": P["transfer"][rows_c].mean(),
        "cycle: event-level model at every cycle, Brier score": ((P["transfer"][rows_c] - e[rows_c]) ** 2).mean(),
        "event: observed ROSC rate at the last cycle": y[rows_p].mean(),
        "event: baseline + time model, mean predicted probability": P["ev_full"][rows_p].mean(),
    }
    v["difference: event baseline+time minus event time alone"] = v["event: baseline + time model (CV)"] - v["event: elapsed-time model (CV)"]
    v["difference: cycle landmark baseline+time minus cycle time-only landmark"] = v["cycle: baseline + time landmark model (CV)"] - v["cycle: elapsed-time landmark model (CV)"]
    v["difference: event time-only AUROC minus cycle time-only landmark AUROC"] = v["event: elapsed-time model (CV)"] - v["cycle: elapsed-time landmark model (CV)"]
    return v


def analyse(name, d, cap=30, seed=0):
    D = build(d["t"], d["y"], d["X"], cap)
    P = crossfit(D, seed)
    idx = [np.where(D["pid"] == i)[0] for i in range(D["n"])]
    point = metrics(D, P, np.arange(D["n"]), np.arange(len(D["e"])))
    r = np.random.default_rng(seed + 100); bs = []
    for _ in range(B):
        s = r.integers(0, D["n"], D["n"])
        try:
            bs.append(metrics(D, P, s, np.concatenate([idx[i] for i in s])))
        except ValueError:
            pass
    bs = pd.DataFrame(bs)
    rep = pd.DataFrame([metrics(D, crossfit(D, 1000 + j), np.arange(D["n"]), np.arange(len(D["e"]))) for j in range(20)])
    rows = [dict(cohort=name, horizon=cap, n=D["n"], excluded=D["excluded"], rosc=int(D["y"].sum()), cycles=len(D["e"]), metric=k,
                 estimate=round(float(v), 4), ci_low=round(float(bs[k].quantile(.025)), 4), ci_high=round(float(bs[k].quantile(.975)), 4),
                 repeated_cv_sd=round(float(rep[k].std()), 4)) for k, v in point.items()]
    # cycle-level ROSC rate by landmark window (direction audit)
    for a, b in ((0, 8), (8, 16), (16, 24), (24, cap)):
        m = (D["L"] >= a) & (D["L"] < b)
        rows.append(dict(cohort=name, horizon=cap, n=D["n"], excluded=D["excluded"], rosc=int(D["y"].sum()), cycles=len(D["e"]),
                         metric="audit: ROSC rate per cycle, landmark %g-%g min (cycles %d)" % (a, min(b, 99), m.sum()),
                         estimate=round(float(D["e"][m].mean()), 4) if m.any() else np.nan, ci_low=np.nan, ci_high=np.nan, repeated_cv_sd=np.nan))
    for x in rows:
        print("%-12s %-80s %8.3f  [%.3f, %.3f]  sd(CV) %s" % (x["cohort"][:12], x["metric"][:80], x["estimate"], x["ci_low"], x["ci_high"], x["repeated_cv_sd"]))
    print(flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()
    Z, C = load(a.data)
    rows = analyse("Zenodo", Z) + analyse("CAAM", C)
    rows += analyse("Zenodo, no cap", Z, cap=1e9) + analyse("CAAM, no cap", C, cap=1e9)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
