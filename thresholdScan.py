#!/usr/bin/env python3
"""
thresholdScan.py - offline threshold scan on the "waveforms" tree of createWaveformsFromNtuple.py

For every stored waveform and every threshold it finds
  tRise[i] : first rising crossing of thr[i] before the maximum (linear interpolation)
  tFall[i] : first falling crossing of thr[i] after the maximum
and writes them to a new file (tree "scan", one row per waveform, same order as "waveforms"),
together with iRun, evt, ch, Npe, amp. Thresholds that are not crossed give NaN.

With --t1 and --t2 it also prints the TOFHIR-like ToT grid
  ToT(T1_i, T2_j) = tFall(T2_j) - tRise(T1_i)       (median per pair)
which is what TOFHIR measures in the run-0305 configuration.

Thresholds are in the units of the waveform (model units, or uA once the fitted normalization is used),
or in the test-beam convention with --thrDAC (ith DAC steps, see tbunits.py; needs --run if models differ):
  python3 thresholdScan.py waveforms.root --run 0 --ch 3 --thrDAC 3 4 5 6

Usage
  python3 thresholdScan.py waveforms_SND_25um_HPK_T1.root --thr 2 5 10 20 50 -o scan.root
  python3 thresholdScan.py waveforms_SND_25um_HPK_T1.root --t1 1 2 3 --t2 4 6 10 --run 0 --ch 3
"""
import argparse
import numpy as np
import tbunits


def crossings(W, t, thr, rising):
    """W: (n, nS) waveforms, t: (nS,) times. Returns (n,) crossing times, NaN if none."""
    ip = np.argmax(W, axis=1)
    idx = np.arange(W.shape[1] - 1)[None, :]
    if rising:
        cond = (W[:, :-1] < thr) & (W[:, 1:] >= thr) & (idx + 1 <= ip[:, None])
    else:
        cond = (W[:, :-1] >= thr) & (W[:, 1:] < thr) & (idx >= ip[:, None])
    k = np.argmax(cond, axis=1)
    rows = np.arange(len(k))
    has = cond[rows, k]
    w0, w1 = W[rows, k], W[rows, k + 1]
    dt = t[1] - t[0]
    tc = t[k] + (thr - w0) / np.where(w1 != w0, w1 - w0, 1e-30) * dt
    return np.where(has, tc, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="waveform file(s), e.g. all part files of a run")
    ap.add_argument("--thr", type=float, nargs="*", default=[], help="thresholds for tRise / tFall")
    ap.add_argument("--t1", type=float, nargs="*", default=[], help="T1 thresholds for the ToT grid")
    ap.add_argument("--t2", type=float, nargs="*", default=[], help="T2 thresholds for the ToT grid")
    ap.add_argument("--run", type=int, default=None, help="only this iRun (SiPM config)")
    ap.add_argument("--ch", type=int, default=None, help="only this channel")
    ap.add_argument("-o", "--out", default=None, help="output file for the per-waveform crossing times")
    tbunits.add_args(ap)
    a = ap.parse_args()

    import uproot
    f = uproot.open(a.files[0])
    ax = f["axis"].arrays(library="np")
    t0, dt, nS = float(ax["t0"][0]), float(ax["dt"][0]), int(ax["nSamples"][0])
    t = t0 + dt * np.arange(nS)
    dac = tbunits.DAC(a)
    if dac.on:
        import wfio
        cfs = wfio.configs(f)
        cf = cfs[a.run] if a.run is not None else next(iter(cfs.values()))
        if a.run is None and len({(c["OV"], c["amp1pe"]) for c in cfs.values()}) > 1:
            raise SystemExit(">>> the file has several SiPM models: choose one with --run for --thrDAC")
        print(f">>> --thrDAC, {cf['name']}:")
        for line in dac.table(a.thrDAC, cf):
            print("      " + line)
        a.thr = list(a.thr) + [round(dac.to_model(v, cf), 6) for v in a.thrDAC]
    thr = sorted(set(a.thr) | set(a.t1) | set(a.t2))
    if not thr:
        raise SystemExit("give --thr and/or --t1/--t2")

    keep = {k: [] for k in ("iRun", "evt", "ch", "Npe", "amp")}
    tR = {x: [] for x in thr}
    tF = {x: [] for x in thr}
    import wfio
    for chunk in (c for fn in a.files for c in wfio.iterate_long(uproot.open(fn), ["iRun", "evt", "ch", "Npe", "amp", "wf"], iRun=a.run)):
        sel = np.ones(len(chunk["evt"]), dtype=bool)
        if a.run is not None: sel &= chunk["iRun"] == a.run
        if a.ch is not None:  sel &= chunk["ch"] == a.ch
        if not sel.any():
            continue
        W = np.asarray(chunk["wf"][sel], dtype=float)
        for k in keep:
            keep[k].append(chunk[k][sel])
        for x in thr:
            tR[x].append(crossings(W, t, x, True))
            tF[x].append(crossings(W, t, x, False))
    if not keep["evt"]:
        raise SystemExit("no waveforms selected")
    keep = {k: np.concatenate(v) for k, v in keep.items()}
    tR = {x: np.concatenate(v) for x, v in tR.items()}
    tF = {x: np.concatenate(v) for x, v in tF.items()}
    n = len(keep["evt"])
    print(f">>> {n} waveforms, window {t[0]:.2f}-{t[-1]:.2f} ns")

    print("\n    thr      crossed   median tRise   median tFall   median width(tFall-tRise)")
    for x in thr:
        ok = np.isfinite(tR[x])
        w = tF[x] - tR[x]
        print(f"    {x:7.3g}  {ok.mean():7.1%}   {np.nanmedian(tR[x]):10.3f}     {np.nanmedian(tF[x]):10.3f}"
              f"     {np.nanmedian(w):10.3f}")

    if a.t1 and a.t2:
        print("\n    median ToT(T1, T2) = tFall(T2) - tRise(T1)   [ns]   rows T1, cols T2 = " + " ".join(f"{x:g}" for x in a.t2))
        for x1 in a.t1:
            row = []
            for x2 in a.t2:
                d = tF[x2] - tR[x1]
                row.append(f"{np.nanmedian(d):7.3f}" if np.isfinite(d).sum() else "    -  ")
            print(f"    T1 {x1:6.3g}: " + "  ".join(row))

    if a.out:
        with uproot.recreate(a.out) as fo:
            fo["scan"] = {**keep,
                          "thresholds": np.tile(np.array(thr, dtype=np.float32), (n, 1)),
                          "tRise": np.stack([tR[x] for x in thr], axis=1).astype(np.float32),
                          "tFall": np.stack([tF[x] for x in thr], axis=1).astype(np.float32)}
        print(f"\n>>> wrote {a.out} (tree 'scan': tRise[nThr], tFall[nThr] per waveform)")


if __name__ == "__main__":
    main()