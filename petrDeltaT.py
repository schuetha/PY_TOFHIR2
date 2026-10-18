#!/usr/bin/env python3
"""
petrDeltaT.py - Petr Gorbounov's test-beam dT analysis (SND@LHC Coll. meeting 24.09.2026), applied to simulation

His recipe (slides 3-5, 7, 9-10, 13), and what it becomes here:
  TOFHIR per hit:  T = time of the T1 crossing (TDC)          -> t = T1 crossing of the simulated waveform
                   E = QDC "energy" (charge ~ number of photons) -> E = Npe x --qdcPerPe  (optionally saturated)
  local time:      T - T0 (external trigger)                    -> t - t_muon (Geant4 time 0; + optional T0 jitter)
  time-walk:       fit "dt vs energy" per SiPM, subtract        -> profile (median t in E bins) fitted with
                                                                   f(E) = a + b lnE + c lnE^2 (or a + b/E^c),
                                                                   t_corr = t - f(E)
  selection:       both SiPMs in their E-band (red lines),       -> --eBand lo hi  or  --eBandPct lo hi (per SiPM)
                   |E1 - E2| < 40 (March) / 30 (July) QDC         -> --dE (absolute) or --dEfrac (fraction of <E>)
  result:          dT = t_corr(a) - t_corr(b), histogram -2..2 ns, Gaussian fit (red),
                   sigma(SiPM) = sigma(pair) / sqrt(2)              -> identical
  colours as on his slides: blue = dT before the walk correction, red = after ("dtcorr"), with its Gaussian fit

Output (-o folder): one page per SiPM pair laid out like his slides (E spectra with the E-band, dt vs E with the
walk fit, dT raw + corrected with the Gaussian fit box), a summary page, petr_dT.csv.

Not in the simulation (so sigma is a lower limit): electronic noise, dark counts, TDC binning, channel calibration.
--tdcSigma adds a Gaussian time smearing per hit [ps], --t0Sigma a common T0 jitter [ps] (cancels in dT).

Usage
  python3 petrDeltaT.py SiPMModel/createWaveforms/waveforms_SND_25um_HPK_T1_muon_beam_100000events_withDLED_0.58ns.root --thrPE 50 -o no_Ecuts_with50pE_threshold                                  # all events, no cuts
  python3 petrDeltaT.py SiPMModel/createWaveforms/waveforms_SND_25um_HPK_T1_muon_beam_100000events_withDLED_0.58ns.root --thrPE 50 --eBandPct 40 90 --dEfrac 0.07  -o Petr_like_cuts_with50pE_threshold   # Petr-like cuts
  python3 petrDeltaT.py waveforms.root --thrDAC 5 --tbRun run0305_e.root ...        # T1 threshold as ith1
"""

import argparse
import os
import numpy as np
import wfio
import tbunits
from thresholdScan import crossings

BLUE, RED, INK, MUTED = "#2a78d6", "#e34948", "#0b0b0b", "#898781"

def walk_fit(E, t, nb=25, func="log2"):
    """profile of t vs E (median t per E-quantile bin), fitted with
         func = "log2":  t(E) = a + b ln E + c (ln E)^2   (linear fit, always converges; default)
         func = "power": t(E) = a + b / E^c
       returns (f, profile x, profile y)"""
    from scipy.optimize import curve_fit
    ok = np.isfinite(E) & np.isfinite(t) & (E > 0)frankreich@frankreich-ASUS-TUF-Gaming-A15-FA507XI-FA507XI:~/TOFHIR_code$ 
    E, t = E[ok], t[ok]
    if len(E) < 50:
        return (lambda x: np.full_like(np.asarray(x, float), np.median(t) if len(t) else 0.)), np.array([]), np.array([])
    edges = np.unique(np.quantile(E, np.linspace(0, 1, nb + 1)))
    idx = np.clip(np.searchsorted(edges, E, side="right") - 1, 0, len(edges) - 2)
    px, py, pw = [], [], []
    for i in range(len(edges) - 1):
        s = idx == i
        if s.sum() >= 10:
            px.append(np.median(E[s])); py.append(np.median(t[s])); pw.append(s.sum())
    px, py, pw = np.array(px), np.array(py), np.array(pw)
    if func == "log2" and len(px) >= 3:
        cf = np.polyfit(np.log(px), py, 2, w=np.sqrt(pw))
        return (lambda x: np.polyval(cf, np.log(np.clip(np.asarray(x, float), 1e-6, None)))), px, py
    f = lambda x, a, b, c: a + b / np.power(x, c)
    try:
        a0 = py[-1]
        b0 = max((py[0] - a0) * px[0] ** 0.5, 1e-6)
        p, _ = curve_fit(f, px, py, p0=(a0, b0, 0.5), sigma=1 / np.sqrt(pw),
                         bounds=([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 4.]), maxfev=20000)
        return (lambda x: f(np.asarray(x, float), *p)), px, py
    except Exception:
        return (lambda x: np.interp(x, px, py)), px, py


def gaus_fit_full(dT, rng, binw):
    """ROOT-like h->Fit("gaus") over the full histogram range: returns dict and histogram"""
    from scipy.optimize import curve_fit
    edges = np.arange(-rng, rng + binw / 2, binw)
    h, _ = np.histogram(dT, bins=edges)
    c = 0.5 * (edges[1:] + edges[:-1])
    m = h > 0
    g = lambda x, A, mu, s: A * np.exp(-0.5 * ((x - mu) / s) ** 2)
    mu0, s0 = np.median(dT), 1.4826 * np.median(np.abs(dT - np.median(dT))) or 0.05
    try:
        p, cov = curve_fit(g, c[m], h[m], p0=(h.max(), mu0, s0), sigma=np.sqrt(h[m]), absolute_sigma=True)
        e = np.sqrt(np.diag(cov))
        chi2 = float(np.sum(((h[m] - g(c[m], *p)) / np.sqrt(h[m])) ** 2))
        res = dict(const=p[0], constErr=e[0], mean=p[1], meanErr=e[1], sigma=abs(p[2]), sigmaErr=e[2],
                   chi2=chi2, ndf=int(m.sum() - 3), entries=len(dT))
    except Exception:
        res = dict(const=np.nan, constErr=np.nan, mean=np.nan, meanErr=np.nan, sigma=np.nan, sigmaErr=np.nan,
                   chi2=np.nan, ndf=0, entries=len(dT))
    return res, h, edges, g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="waveform file(s) of ONE DLED delay (part files allowed)")
    ap.add_argument("--run", type=int, default=0, help="SiPM model (iRun) to analyse (default 0)")
    ap.add_argument("--thrPE", type=float, default=None, help="T1 threshold [p.e.]")
    ap.add_argument("--thr", type=float, default=None, help="T1 threshold [model units]")
    ap.add_argument("--pairs", nargs="*", default=None, help="pairs a-b (default: all neighbours i-(i+1))")
    ap.add_argument("--nCh", type=int, default=8)
    ap.add_argument("--qdcPerPe", type=float, default=1.0, help="E = Npe x this (to mimic QDC units)")
    ap.add_argument("--qdcMax", type=float, default=None, help="QDC saturation: E above this is set to it")
    ap.add_argument("--eBand", type=float, nargs=2, default=None, help="E-band [E units], same for all SiPMs")
    ap.add_argument("--eBandPct", type=float, nargs=2, default=None, help="E-band in percentiles, per SiPM")
    ap.add_argument("--dE", type=float, default=None, help="|E1-E2| < dE [E units]")
    ap.add_argument("--dEfrac", type=float, default=None, help="|E1-E2| < dEfrac x (E1+E2)/2")
    ap.add_argument("--tdcSigma", type=float, default=0., help="Gaussian smearing per hit [ps] (TDC/electronics)")
    ap.add_argument("--t0Sigma", type=float, default=0., help="Gaussian T0 jitter per event [ps]")
    ap.add_argument("--dtRange", type=float, default=2., help="dT histogram range +-[ns] (Petr: 2)")
    ap.add_argument("--dtBin", type=float, default=0.02, help="dT bin width [ns]")
    ap.add_argument("--walkFunc", choices=["log2", "power"], default="log2",
                    help="time-walk curve: log2 = a + b lnE + c lnE^2 (default), power = a + b/E^c")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-o", "--out", default="petr_dT")
    tbunits.add_args(ap)
    a = ap.parse_args()
    import uproot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(a.out, exist_ok=True)

    # ---------------- threshold (one, like a TOFHIR run)
    f0 = uproot.open(a.files[0])
    cfs = wfio.configs(f0)
    cf = cfs[a.run]
    t = wfio.time_axis(f0)
    dac = tbunits.DAC(a)
    if dac.on:
        if len(a.thrDAC) != 1:
            raise SystemExit("give ONE threshold (--thrDAC 5)")
        thr, thrLabel = dac.to_model(a.thrDAC[0], cf), f"ith1 = {a.thrDAC[0]:g}"
        for line in dac.table(a.thrDAC, cf):
            print("      " + line)
    elif a.thr is not None:
        thr, thrLabel = a.thr, f"{a.thr:g} model u."
    else:
        pe = a.thrPE if a.thrPE is not None else 5.
        thr, thrLabel = pe * cf["amp1pe"], f"{pe:g} p.e."
    print(f">>> {cf['name']}, DLED {cf['delay']:g} ns, T1 threshold {thrLabel} = {thr:.3f} model units")

    # ---------------- stream: per waveform only (evt, ch, Npe, t_T1)
    E_, C_, N_, T_ = [], [], [], []
    for fn in a.files:
        f = uproot.open(fn)
        if a.run not in wfio.configs(f):
            continue
        for d in wfio.iterate_long(f, ["evt", "ch", "Npe", "wf"], iRun=a.run):
            if len(d["evt"]) == 0:
                continue
            E_.append(d["evt"]); C_.append(d["ch"]); N_.append(d["Npe"])
            W = d["wf"]
            T_.append(np.concatenate([crossings(np.asarray(W[k:k + 5000], float), t, thr, True)
                                      for k in range(0, len(W), 5000)]))
    evt, ch, npe, tc = (np.concatenate(x) for x in (E_, C_, N_, T_))
    nEvt = int(evt.max()) + 1
    rng = np.random.default_rng(a.seed)
    T = np.full((a.nCh, nEvt), np.nan); Q = np.full((a.nCh, nEvt), np.nan)
    s = ch < a.nCh
    T[ch[s], evt[s]] = tc[s]
    Q[ch[s], evt[s]] = npe[s] * a.qdcPerPe
    if a.qdcMax:
        Q = np.minimum(Q, a.qdcMax)
    if a.tdcSigma > 0:
        T += rng.normal(0, a.tdcSigma * 1e-3, T.shape)
    if a.t0Sigma > 0:
        T -= rng.normal(0, a.t0Sigma * 1e-3, nEvt)[None, :]      # T - T0, T0 jittered
    print(f">>> {len(evt)} waveforms, {nEvt} events")

    # ---------------- per SiPM: walk fit and E-band
    walk, prof, band = {}, {}, {}
    for c in range(a.nCh):
        walk[c], px, py = walk_fit(Q[c], T[c], func=a.walkFunc)
        prof[c] = (px, py)
        q = Q[c][np.isfinite(Q[c])]
        if a.eBand:
            band[c] = tuple(a.eBand)
        elif a.eBandPct and len(q):
            band[c] = tuple(np.percentile(q, a.eBandPct))
        else:
            band[c] = (-np.inf, np.inf)
    Tc = np.vstack([T[c] - walk[c](Q[c]) for c in range(a.nCh)])

    pairs = ([tuple(int(v) for v in p.split("-")) for p in a.pairs] if a.pairs
             else [(i, i + 1) for i in range(a.nCh - 1)])
    cuts = []
    if a.eBand: cuts.append(f"E-band {a.eBand[0]:g}-{a.eBand[1]:g}")
    if a.eBandPct: cuts.append(f"E-band {a.eBandPct[0]:g}-{a.eBandPct[1]:g}% per SiPM")
    if a.dE is not None: cuts.append(f"|E1-E2|<{a.dE:g}")
    if a.dEfrac is not None: cuts.append(f"|E1-E2|<{a.dEfrac:g}x<E>")
    if a.tdcSigma: cuts.append(f"TDC {a.tdcSigma:g} ps")
    cutTxt = ", ".join(cuts) or "no cuts"

    rows = []
    pdf = PdfPages(os.path.join(a.out, "petr_dT.pdf"))
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": "#e1e0d9",
                         "grid.linestyle": ":"})
    for (p, q) in pairs:
        sel = np.isfinite(T[p]) & np.isfinite(T[q])
        for c in (p, q):
            sel &= (Q[c] >= band[c][0]) & (Q[c] <= band[c][1])
        dEv = np.abs(Q[p] - Q[q])
        if a.dE is not None:
            sel &= dEv < a.dE
        if a.dEfrac is not None:
            sel &= dEv < a.dEfrac * 0.5 * (Q[p] + Q[q])
        raw = T[p][sel] - T[q][sel]
        cor = Tc[p][sel] - Tc[q][sel]
        fr, hr, edges, g = gaus_fit_full(raw, a.dtRange, a.dtBin)
        fc, hc, _, _ = gaus_fit_full(cor, a.dtRange, a.dtBin)
        rows.append(dict(model=cf["name"], delayNs=cf["delay"], threshold=thrLabel, pair=f"{p}-{q}", cuts=cutTxt,
                         nEvents=int(sel.sum()), sigmaPairRaw_ps=1e3 * fr["sigma"], sigmaPair_ps=1e3 * fc["sigma"],
                         sigmaPairErr_ps=1e3 * fc["sigmaErr"], sigmaSiPM_ps=1e3 * fc["sigma"] / np.sqrt(2),
                         meanCorr_ns=fc["mean"], chi2=fc["chi2"], ndf=fc["ndf"]))

        # ---------- page like Petr's slides
        fig = plt.figure(figsize=(13, 6.4))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.38, wspace=0.28)
        for k, c in enumerate((p, q)):
            ax = fig.add_subplot(gs[0, 2 * k])
            qq = Q[c][np.isfinite(Q[c])]
            ax.hist(qq, bins=100, histtype="step", color="#1c3f8f", lw=1)
            for b in band[c]:
                if np.isfinite(b):
                    ax.axvline(b, color=RED, lw=1)
            ax.set_title(f"Energy, SiPM {c}", fontsize=9); ax.set_xlabel("E (Npe x qdcPerPe)")
            ax = fig.add_subplot(gs[0, 2 * k + 1])
            ok = np.isfinite(Q[c]) & np.isfinite(T[c])
            ax.plot(Q[c][ok][::max(1, ok.sum() // 4000)], T[c][ok][::max(1, ok.sum() // 4000)], ".",
                    ms=1.2, color="#1c3f8f", alpha=0.35)
            px, py = prof[c]
            ax.plot(px, py, "o", ms=3, color=INK)
            xx = np.linspace(max(qq.min(), 1e-3), qq.max(), 200) if len(qq) else []
            ax.plot(xx, walk[c](xx), color=RED, lw=1.6, label=("walk fit a+b lnE+c lnE²" if a.walkFunc == "log2" else "walk fit a+b/E^c"))
            if len(py):
                ax.set_ylim(np.nanmin(py) - 0.15, np.nanmax(py) + 0.25)
            ax.set_title(f"t(ns) vs energy, SiPM {c}", fontsize=9); ax.set_xlabel("E"); ax.set_ylabel("t (ns)")
            ax.legend(fontsize=7, loc="upper right")
        ax = fig.add_subplot(gs[1, 0:2])
        c_ = 0.5 * (edges[1:] + edges[:-1])
        ax.step(c_, hr, where="mid", color=BLUE, lw=1.1, label=f"raw (no walk corr.): sigma {1e3 * fr['sigma']:.0f} ps")
        ax.step(c_, hc, where="mid", color=RED, lw=1.1, label="walk-corrected (dtcorr)")
        xx = np.linspace(-a.dtRange, a.dtRange, 800)
        if np.isfinite(fc["sigma"]):
            ax.plot(xx, g(xx, fc["const"], fc["mean"], fc["sigma"]), color="black", lw=1.4, ls="--", label="Gaussian fit (corrected)")
        ax.set_xlim(-a.dtRange, a.dtRange); ax.set_xlabel("dT (ns)")
        ax.set_title(f"dtcorr({p}-{q})  {cutTxt}", fontsize=9)
        box = (f"dT({p}-{q})\n$\\chi^2$ / ndf   {fc['chi2']:.1f} / {fc['ndf']}\n"
               f"Constant   {fc['const']:.1f} ± {fc['constErr']:.1f}\nMean   {fc['mean']:.4f} ± {fc['meanErr']:.4f}\n"
               f"Sigma   {fc['sigma']:.4f} ± {fc['sigmaErr']:.4f}\nEntries   {fc['entries']}")
        ax.text(0.99, 0.97, box, transform=ax.transAxes, ha="right", va="top", fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="square", fc="white", ec=INK, lw=0.6))
        ax.legend(fontsize=7, loc="upper left")
        ax2 = fig.add_subplot(gs[1, 2:4]); ax2.axis("off")
        ax2.text(0.02, 0.85, f"dt {p}-{q}", fontsize=13, color=INK)
        ax2.text(0.02, 0.68, f"σ(pair) ≈ {1e3 * fc['sigma']:.0f} ps", fontsize=13, color=INK)
        ax2.text(0.02, 0.51, f"σ(SiPM) ≈ {1e3 * fc['sigma'] / np.sqrt(2):.0f} ps", fontsize=13, color=RED)
        # ax2.text(0.02, 0.28, f"{cf['name']}, DLED {cf['delay']:g} ns, T1 thr {thrLabel}\n{cutTxt}\n"
        #                      f"{int(sel.sum())} events; before walk correction σ(pair) = {1e3 * fr['sigma']:.0f} ps",
                #  fontsize=9, color=MUTED, va="top")
        fig.suptitle(f"Adjacent SiPMs {p} and {q} - simulation", fontsize=11)
        fig.savefig(os.path.join(a.out, f"petr_dT_{p}-{q}.png"), dpi=1200)
        pdf.savefig(fig); plt.close(fig)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "petr_dT.csv"), index=False, float_format="%.4g")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.sigmaPairRaw_ps / np.sqrt(2), 0.38, color=BLUE, label="before walk correction")
    ax.bar(x + 0.2, df.sigmaSiPM_ps, 0.38, color=RED, label="walk-corrected")
    ax.set_xticks(x, df.pair); ax.set_ylabel("σ(SiPM) = σ(pair)/√2  [ps]"); ax.set_xlabel("SiPM pair")
    ax.set_title(f"{cf['name']}, T1 {thrLabel}, {cutTxt}", fontsize=9); ax.legend(fontsize=8)
    for i, v in enumerate(df.sigmaSiPM_ps):
        ax.text(i + 0.2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "petr_dT_summary.png"), dpi=1200); pdf.savefig(fig)
    pdf.close()
    print(df[["pair", "nEvents", "sigmaPairRaw_ps", "sigmaPair_ps", "sigmaSiPM_ps", "chi2", "ndf"]]
          .round(1).to_string(index=False))
    print(f">>> wrote {a.out}/petr_dT.pdf, petr_dT_<pair>.png, petr_dT_summary.png, petr_dT.csv")


if __name__ == "__main__":
    main()