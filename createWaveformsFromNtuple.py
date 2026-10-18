#!/usr/bin/env python3
"""
createWaveformsFromNtuple.py - Python version of createWaveformsFromNtuple.cpp

Same cfg file and model as the C++ version. Output <plotFolder>/waveforms_<label>[_<tag>].root with
    one TTree PER SiPM MODEL, named after its cfg block (e.g. "SiPM_25um_OV4", "SiPM_15um_OV4").
    Every model tree has the SAME leaves, one row per (event, channel):
                       iRun, evt, ch, OV, gain, nPhot, Npe, amp, tAmp, SR, wf[nSamples]
                       wf = the waveform samples in the window [wfTMin, wfTMax) ns (float32)
                       (channels without photons are skipped)
    -> the same analysis code runs on any model, only the tree name changes, e.g. in ROOT
       SiPM_25um_OV4->Draw("amp", "ch==3")      SiPM_15um_OV4->Draw("amp", "ch==3")
    TTree "axis"       t0 (centre of the first stored sample), dt, nSamples  -> t_k = t0 + k*dt
    TTree "configs"    iRun, name (= tree name), OV, gain, Rq, Cq, Rd, Cd, Nc, Cg, RL, RF, delayNs, amp1pe
    g_SR, g_amp        slew rate / amplitude vs Npe x gain (all models, as before)
    hOuTot_Npe<Npe>__<nGen>  only if Output.saveHistos 1 (the old format)

With Output.filePerModel 1 (or --filePerModel) every model goes to its own file instead:
    <plotFolder>/waveforms_<label>[_<tag>]_<model>.root   (same tree name and leaves inside)

Optional cfg keys in <Output>:
    saveTree   1        write the model trees (default 1)
    filePerModel 0      1 = one output file per SiPM model (default 0 = all model trees in one file)
    saveHistos 0        also write one TH1D per waveform, as the C++ version (default 0)
    wfTMin     15.      start of the stored window [ns] (default 15; the SiPM pulse starts at 20 ns)
    wfTMax     65.      end of the stored window [ns]   (default 65)
    dledDelayNs 0.58    DLED delay [ns] = tau of the delayed (RC) copy subtracted from the direct one
                        (default 0.5746 = the MTD 277 MHz pole; 0 = no DLED). --delay overrides it.
The "configs" tree also stores delayNs and amp1pe (peak of the 1-p.e. response, to express thresholds in p.e.).

Usage
  python3 createWaveformsFromNtuple.py cfg/createWaveformsFromNtuple.cfg          # cores from cfg (default 1)
  python3 createWaveformsFromNtuple.py cfg/createWaveformsFromNtuple.cfg -j 8     # 8 parallel processes
  python3 createWaveformsFromNtuple.py cfg/createWaveformsFromNtuple.cfg -j 0     # all CPU cores

  # separate jobs (e.g. HTCondor), each its own event range and output file:
  python3 createWaveformsFromNtuple.py cfg.cfg --first 0    --last 5000  --tag part0
  python3 createWaveformsFromNtuple.py cfg.cfg --first 5000 --last 10000 --tag part1
  (the analysis scripts deltaT.py / plotWaveforms.py / thresholdScan.py take all part files together)

Parallel running: events are split into blocks that are simulated by separate processes
(multiprocessing); the main process writes them in the original order, so the output file is
identical to a single-process run. cfg key Input.nWorkers sets the default, -j overrides it.

Memory: the photon times are NOT loaded in advance - every worker reads only the events of its
own block from the ntuple; at most 2 x jobs blocks are in flight (workers wait when writing is
slower), results travel as numpy arrays. RAM stays ~constant, whatever the number of events.
    --blockSize N   events per block (default 200; smaller = less RAM, a bit slower)
    --inFlight  N   blocks simulated / waiting at the same time (default 2 x jobs)
    --recycle   N   every worker is replaced by a fresh process after N blocks (default 25), so memory
                    fragmentation in long-lived workers cannot pile up; 0 = never

Needs numpy, uproot (>= 5.7: older versions crash on files > ~2 GB), pandas and Functions.py (= lab5015model.py) in the same folder.
"""
import os
import sys
import numpy as np
import Functions as M          # = lab5015model.py, renamed


DEFAULT_DLED_NS = 1. / (277. * 2. * 3.14159) * 1e3     # 0.5746 ns = the 277 MHz Elmore pole of the MTD model
_KCACHE = {}


def _frontend_kernel_fft(sipmPars, BW2fix, delayNs):
    """rfft of everything after the p.e. list: SiPM x HP(BW2fix) x LP 20 MHz x HP 30 MHz x LP 1.19 GHz x DLED x norm.
    The chain is linear, so w = hLYSO (x) K; K is cached per (SiPM, BW2fix, delay)."""
    key = (sipmPars["name"], round(BW2fix, 6), delayNs)
    K = _KCACHE.get(key)
    if K is not None:
        return K
    with np.errstate(divide="ignore"):
        tau1 = 1. / (20. * 2. * 3.14159) * 1e3         # TOFHIR low-pass 20 MHz
        tau2 = 4 / 30.                                 # ~1.2 GHz, effect of inductance
        tau3 = 1. / (BW2fix * 2. * 3.14159) * 1e3 if BW2fix > 0 else np.inf   # high pass
        tau4 = 1. / (30. * 2. * 3.14159) * 1e3         # additional HP, 30 MHz
    tauE = delayNs if delayNs > 0 else np.inf          # DLED: delta - low-pass(tauE); delay 0 -> no DLED
    R = np.fft.rfft
    hSiPM = 0.5 * sipmPars["RF"] * M.SiPMPulseShape(M.TT[:M.NPT], sipmPars, 1., 20.)
    K = (R(hSiPM) * R(M.highpass_bins(tau3)) * R(M.lowpass_bins(tau1)) * R(M.highpass_bins(tau4))
         * R(M.lowpass_bins(tau2)) * R(M.highpass_bins(tauE)) * 1.725e06)
    if BW2fix == 0.:                                    # only the common case is worth keeping
        _KCACHE[key] = K
    return K


def myInSignal(sipmPars, peTimes, Npe, gain, delayNs=DEFAULT_DLED_NS):
    """SiPM model on an external load + TOFHIR chain (MTD constants), DLED delay configurable [ns]"""
    x = Npe * gain                     # normalized to 9500 p.e. and a gain of 3.72e5 (~3.5 Vov) in MTD
    BW2fix = 0. if x < 4.06809e+08 else 6.50098e-09 * (x - 4.06809e+08)
    # p.e. arrival times from the Geant4 optical simulation (TH1D::Fill: outside [0,tmax) -> under/overflow)
    pe = np.asarray(peTimes, dtype=float)
    pe = pe[(pe >= 0.) & (pe < M.TMAX)]
    hLYSO = np.bincount((pe / M.DT).astype(int), minlength=M.NPT)[:M.NPT].astype(float)
    return np.fft.irfft(np.fft.rfft(hLYSO) * _frontend_kernel_fft(sipmPars, BW2fix, delayNs), n=M.NPT)


def single_pe_amplitude(sipmPars, delayNs):
    """peak of the response to one p.e. (BW2fix = 0): converts thresholds to p.e. units"""
    return float(myInSignal(sipmPars, [0.], 1., 0., delayNs).max())


def slew_rate_and_amplitude(h):
    """the C++ loop: amplitude = maximum; SR from the first bin >= 4.695 and the next bin"""
    ymax = float(h.max())
    above = np.nonzero(h >= 4.695)[0]
    if len(above) == 0:
        return float("nan"), ymax                     # C++: (-1 - -1)/(-1 - -1) = nan
    b = above[0]
    x1, y1 = M.TC[b], h[b]
    if b + 1 < M.NPT:
        x2, y2 = M.TC[b + 1], h[b + 1]
    else:                                             # overflow bin in ROOT: content 0
        x2, y2 = M.TC[b] + M.DT, 0.
    return float((y2 - y1) / (x2 - x1)), ymax


# ------------------------------------------------------------------ parallel workers
_W = {}          # small, read-only settings shared with the workers (no photon data!)
_FILES = {}      # per process: open ntuple files (opened lazily inside each worker)


def _init_worker(shared):
    _W.clear()
    _W.update(shared)
    _FILES.clear()            # never reuse file handles inherited from the parent


def _read_times(e0, e1):
    """photon times of global events [e0, e1): {ch: list of arrays}, read from the ntuple(s) on demand"""
    import uproot
    out = {c: [] for c in _W["channels"]}
    needed = {fn for fn, off, n in zip(_W["files"], _W["offsets"], _W["entries"]) if off < e1 and off + n > e0}
    for fn in list(_FILES):                      # keep only the files this block needs open
        if fn not in needed:
            _FILES.pop(fn).file.close()
    for fn, off, n in zip(_W["files"], _W["offsets"], _W["entries"]):
        a0, a1 = max(e0, off), min(e1, off + n)
        if a0 >= a1:
            continue
        if fn not in _FILES:
            _FILES[fn] = uproot.open(fn, array_cache=None)["sipm"]
        arr = _FILES[fn].arrays(_W["branches"], entry_start=a0 - off, entry_stop=a1 - off, library="np")
        for c in _W["channels"]:
            out[c].extend(arr[f"sipm{c}_photonTimes_ns"])
    return out


def _simulate_block(task):
    """simulate events [e0, e1) for every SiPM model; returns {iRun: dict of numpy columns}"""
    e0, e1 = task
    chTimes, channels, runs = _read_times(e0, e1), _W["channels"], _W["runs"]
    b0, b1, keepFull = _W["b0"], _W["b1"], _W["saveHistos"]
    res = {}
    for iRun in runs:
        sp = _W["sipms"][iRun]
        gain = (sp["Cq"] + sp["Cd"]) * (sp["OV"] + 0.25) / (1.602 * 1e-19)
        evt, ch, nPhot, Npe, amp, tAmp, SR, wf, full = [], [], [], [], [], [], [], [], []
        for k, iEvent in enumerate(range(e0, e1)):
            for c in channels:
                peTimes = chTimes[c][k]
                if len(peTimes) == 0:
                    continue
                pe = np.asarray(peTimes, dtype=float)
                h = myInSignal(sp, pe, float(len(pe)), gain, _W["delayNs"])
                s, ymax = slew_rate_and_amplitude(h)
                evt.append(iEvent); ch.append(c); nPhot.append(len(pe))
                Npe.append(int(((pe >= 0.) & (pe < M.TMAX)).sum()))
                amp.append(ymax); tAmp.append(M.TC[int(np.argmax(h))]); SR.append(s)
                wf.append(h[b0:b1].astype(np.float32))
                if keepFull:
                    full.append(h)
        res[iRun] = dict(evt=np.array(evt, np.int32), ch=np.array(ch, np.int32),
                         nPhot=np.array(nPhot, np.int32), Npe=np.array(Npe, np.int32),
                         amp=np.array(amp, np.float32), tAmp=np.array(tAmp, np.float32),
                         SR=np.array(SR, np.float32),
                         wf=np.stack(wf) if wf else np.zeros((0, b1 - b0), np.float32),
                         full=full if keepFull else None)
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cfg")
    ap.add_argument("-j", "--jobs", type=int, default=None, help="parallel processes (0 = all cores)")
    ap.add_argument("--first", type=int, default=None, help="first event (for splitting into separate jobs)")
    ap.add_argument("--last", type=int, default=None, help="last event + 1 (exclusive)")
    ap.add_argument("--tag", default=None, help="added to the output name: waveforms_<label>_<tag>.root")
    ap.add_argument("--delay", type=float, default=None,
                    help="DLED delay [ns] (default cfg Output.dledDelayNs, else 0.5746 = MTD 277 MHz; 0 = no DLED)")
    ap.add_argument("--maxEvents", type=int, default=None, help="overrides Input.maxEvents (-1 = all)")
    ap.add_argument("--filePerModel", action="store_true", help="one output file per SiPM model")
    ap.add_argument("--blockSize", type=int, default=200, help="events per block (default 200)")
    ap.add_argument("--inFlight", type=int, default=None, help="max blocks in flight (default 2 x jobs)")
    ap.add_argument("--recycle", type=int, default=25,
                    help="replace each worker by a fresh one after this many blocks (frees its memory; 0 = never)")
    args = ap.parse_args()
    import uproot
    import pandas as pd
    import multiprocessing as mp

    opts = M.Cfg(args.cfg)
    label = opts.get("Input.label")
    plotFolder = opts.get("Output.plotFolder")
    os.makedirs(plotFolder, exist_ok=True)

    SiPMParamsVec, runsVec = M.GetSiPMParsFromCfg(opts)
    nRuns = len(runsVec)

    ntupleFileName = opts.get("Input.ntupleFileName", many=True)
    channels = opts.get("Input.channels", int, many=True)
    branches = [f"sipm{c}_photonTimes_ns" for c in channels]
    entries = []                                   # only the number of events - the photon times are read
    for f in ntupleFileName:                       # later by the workers, block by block
        with uproot.open(f) as fi:
            entries.append(int(fi["sipm"].num_entries))
    offsets = list(np.cumsum([0] + entries[:-1]).astype(int))
    nEvents = sum(entries)
    maxEvents = args.maxEvents if args.maxEvents is not None else (
        opts.get("Input.maxEvents", int) if opts.exists("Input.maxEvents") else -1)
    if True:
        if 0 < maxEvents < nEvents:
            nEvents = maxEvents
    first = args.first if args.first is not None else 0
    last = min(args.last, nEvents) if args.last is not None else nEvents
    print(f">>> events {first}-{last - 1} of {nEvents} in the ntuple(s), {len(channels)} channels")

    saveTree = opts.get("Output.saveTree", int) if opts.exists("Output.saveTree") else 1
    saveHistos = opts.get("Output.saveHistos", int) if opts.exists("Output.saveHistos") else 0
    wfTMin = opts.get("Output.wfTMin", float) if opts.exists("Output.wfTMin") else 15.
    wfTMax = opts.get("Output.wfTMax", float) if opts.exists("Output.wfTMax") else 65.
    delayNs = args.delay if args.delay is not None else (
        opts.get("Output.dledDelayNs", float) if opts.exists("Output.dledDelayNs") else DEFAULT_DLED_NS)
    print(f">>> DLED delay {delayNs:.3f} ns" + (" (no DLED)" if delayNs <= 0 else
          f"  (pole {1e3 / (2 * np.pi * delayNs):.0f} MHz)"))
    b0, b1 = int(wfTMin / M.DT), int(np.ceil(wfTMax / M.DT))
    nS = b1 - b0
    print(f">>> waveform window {b0 * M.DT:.2f}-{b1 * M.DT:.2f} ns: {nS} samples of {M.DT * 1000:.0f} ps "
          f"({4 * nS / 1024:.1f} kB per waveform before compression)")

    cols = ("iRun", "evt", "ch", "OV", "gain", "nPhot", "Npe", "amp", "tAmp", "SR", "wf")
    types = {"iRun": np.int32, "evt": np.int32, "ch": np.int32, "OV": np.float32, "gain": np.float32,
             "nPhot": np.int32, "Npe": np.int32, "amp": np.float32, "tAmp": np.float32, "SR": np.float32}
    edges = np.linspace(0., M.TMAX, M.NPT + 1)
    # tree name of each model = its cfg block name (only letters, digits and _)
    treeNames = ["".join(ch if ch.isalnum() or ch == "_" else "_" for ch in p["name"]) for p in SiPMParamsVec]
    gains = np.array([(p["Cq"] + p["Cd"]) * (p["OV"] + 0.25) / (1.602 * 1e-19) for p in SiPMParamsVec])
    amp1pe = np.array([single_pe_amplitude(p, delayNs) for p in SiPMParamsVec])
    filePerModel = args.filePerModel or (opts.exists("Output.filePerModel") and opts.get("Output.filePerModel", int) == 1)
    base = os.path.join(plotFolder, f"waveforms_{label}" + (f"_{args.tag}" if args.tag else ""))
    # which models go into which file
    outputs = ([(f"{base}_{treeNames[r]}.root", [r]) for r in range(nRuns)] if filePerModel
               else [(f"{base}.root", list(range(nRuns)))])
    print(">>> trees (same leaves " + ", ".join(cols) + "): " + ", ".join(treeNames)
          + ("  -> one file per model" if filePerModel else ""))

    nJobs = args.jobs if args.jobs is not None else (
        opts.get("Input.nWorkers", int) if opts.exists("Input.nWorkers") else 1)
    if nJobs <= 0:
        nJobs = os.cpu_count() or 1
    allRuns = list(range(nRuns))
    shared = dict(sipms=SiPMParamsVec, channels=channels, branches=branches, files=list(ntupleFileName),
                  offsets=offsets, entries=entries, runs=allRuns, b0=b0, b1=b1, saveHistos=saveHistos,
                  delayNs=delayNs)
    _init_worker(shared)
    block = max(1, min(args.blockSize, (last - first) // (4 * nJobs) or 1))
    tasks = [(e0, min(e0 + block, last)) for e0 in range(first, last, block)]
    inFlight = max(1, args.inFlight or 2 * nJobs)
    print(f">>> {len(tasks)} blocks of up to {block} events on {nJobs} process(es), at most {inFlight} in flight")

    run2file = {r: (outputs[r][0] if filePerModel else outputs[0][0]) for r in allRuns}
    files, trees = {}, {}
    buf = {r: [] for r in allRuns}                     # list of block column dicts per model
    nBuf = {r: 0 for r in allRuns}
    g_x = {fn: [] for fn, _ in outputs}; g_sr = {fn: [] for fn, _ in outputs}; g_am = {fn: [] for fn, _ in outputs}

    def flush(r):
        if not buf[r]:
            return
        n = nBuf[r]
        data = {"iRun": np.full(n, r, np.int32)}
        data["evt"] = np.concatenate([b["evt"] for b in buf[r]])
        data["ch"] = np.concatenate([b["ch"] for b in buf[r]])
        data["OV"] = np.full(n, SiPMParamsVec[r]["OV"], np.float32)
        data["gain"] = np.full(n, gains[r], np.float32)
        for k in ("nPhot", "Npe", "amp", "tAmp", "SR", "wf"):
            data[k] = np.concatenate([b[k] for b in buf[r]])
        if r not in trees:
            trees[r] = files[run2file[r]].mktree(
                treeNames[r], {k: (v.dtype, v.shape[1:]) if k == "wf" else v.dtype for k, v in data.items()},
                title=f"{SiPMParamsVec[r]['name']}, OV {SiPMParamsVec[r]['OV']:g} V, DLED {delayNs:.3f} ns")
        trees[r].extend(data)
        buf[r].clear(); nBuf[r] = 0

    def handle(res):
        for r, b in res.items():
            fn = run2file[r]
            x = b["nPhot"].astype(np.float64) * gains[r]
            g_x[fn].append(x); g_sr[fn].append(b["SR"]); g_am[fn].append(b["amp"])
            if saveHistos:     # the old format: one TH1D per event AND per channel
                for e, c, n, h in zip(b["evt"], b["ch"], b["nPhot"], b["full"]):
                    files[fn][f"hOuTot_Npe{float(n):.0f}__{int(e) * 100 + int(c):04d}"] = (h, edges)
            if saveTree and len(b["evt"]):
                b.pop("full", None)
                buf[r].append(b); nBuf[r] += len(b["evt"])
                if nBuf[r] >= 20000:                   # big baskets: fewer extend() calls
                    flush(r)

    pool = None
    try:
        for fn, runs in outputs:
            files[fn] = uproot.recreate(fn)
            files[fn]["axis"] = {"t0": np.array([M.TC[b0]]), "dt": np.array([M.DT]),
                                 "nSamples": np.array([nS], dtype=np.int32)}
            files[fn]["configs"] = {"iRun": np.array(runs, dtype=np.int32),
                                    "name": np.array([treeNames[r] for r in runs]),
                                    **{k: np.array([SiPMParamsVec[r][k] for r in runs]) for k in
                                       ("OV", "Rq", "Cq", "Rd", "Cd", "Nc", "Cg", "RL", "RF")},
                                    "gain": gains[runs], "delayNs": np.full(len(runs), delayNs),
                                    "amp1pe": amp1pe[runs]}
        if nJobs > 1:
            from collections import deque
            # workers come from a clean server process (not from this one, which holds the output buffers)
            meth = [m for m in ("forkserver", "spawn") if m in mp.get_all_start_methods()][0]
            ctx = mp.get_context(meth)
            pool = ctx.Pool(nJobs, initializer=_init_worker, initargs=(shared,),
                            maxtasksperchild=args.recycle if args.recycle > 0 else None)
            pending, it = deque(), iter(tasks)

            def submit():
                t = next(it, None)
                if t is not None:
                    pending.append((t, pool.apply_async(_simulate_block, (t,))))
            for _ in range(inFlight):
                submit()

            def results():
                while pending:
                    t, ar = pending.popleft()
                    out = ar.get()                       # in order: the file is the same as with -j 1
                    submit()                             # one finished -> one new (bounded memory)
                    yield t, out
            stream = results()
        else:
            stream = ((t, _simulate_block(t)) for t in tasks)
        for done, ((e0, e1), res) in enumerate(stream, 1):
            handle(res)
            print(f"\r>>> event {e1} / {last}   (block {done}/{len(tasks)})", end="", flush=True)
        print()
        if saveTree:
            for r in allRuns:
                flush(r)
        for fn, _ in outputs:
            if g_x[fn]:
                gx = np.concatenate(g_x[fn])
                files[fn]["g_SR"] = uproot.as_TGraph(pd.DataFrame({"x": gx, "y": np.concatenate(g_sr[fn])}))
                files[fn]["g_amp"] = uproot.as_TGraph(pd.DataFrame({"x": gx, "y": np.concatenate(g_am[fn])}))
    finally:
        if pool is not None:
            pool.close(); pool.join()
        for fn, fo in files.items():
            fo.close()
    for fn, _ in outputs:
        print(f">>> wrote {fn}")


if __name__ == "__main__":
    main()