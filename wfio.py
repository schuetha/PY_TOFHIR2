"""
wfio.py - read the output of createWaveformsFromNtuple.py, whatever the tree layout

  layout "perSiPM" (default now): one tree per SiPM model (name in the "configs" tree),
                                  one row per event, branches sipm<c>_wf[nS], sipm<c>_Npe, ...
  layout "perModel":              one tree per SiPM model (name in "configs"), all with the SAME leaves,
                                  one row per (event, channel): iRun, evt, ch, ..., wf[nS]
  layout "long"   (old):          tree "waveforms", one row per (model, event, channel)

iterate_long() returns the old long format from either layout, so the analysis code does not care:
    for d in wfio.iterate_long(f, ["evt", "ch", "Npe", "wf"], iRun=0):
        ...   # d["evt"], d["ch"], d["Npe"], d["wf"] (n, nS) numpy arrays
Channels without any photon are skipped (as in the old layout).
"""
import re
import numpy as np


def layout(f):
    if "waveforms" in f.keys(cycle=False):
        return "long"
    name = str(f["configs"]["name"].array(library="np")[0])
    return "perModel" if "ch" in f[name].keys() else "perSiPM"


def time_axis(f):
    ax = f["axis"].arrays(library="np")
    return float(ax["t0"][0]) + float(ax["dt"][0]) * np.arange(int(ax["nSamples"][0]))


def configs(f):
    """iRun -> dict(name, OV, delay, amp1pe, gain)"""
    c = f["configs"].arrays(library="np")
    out = {}
    for k, i in enumerate(c["iRun"]):
        out[int(i)] = dict(name=str(c["name"][k]), OV=float(c["OV"][k]),
                           delay=float(c["delayNs"][k]) if "delayNs" in c else np.nan,
                           amp1pe=float(c["amp1pe"][k]) if "amp1pe" in c else np.nan,
                           gain=float(c["gain"][k]) if "gain" in c else np.nan)
    return out


def channels(tree):
    return sorted(int(m.group(1)) for b in tree.keys() if (m := re.fullmatch(r"sipm(\d+)_wf", b)))


def iterate_long(f, columns, iRun=None, step=20000):
    """yield dicts of numpy arrays in the old long format (one entry per event x channel)"""
    columns = list(columns)
    if layout(f) == "long":
        need = list(dict.fromkeys(columns + (["iRun"] if iRun is not None else [])))
        for d in f["waveforms"].iterate(need, library="np", step_size=step):
            if iRun is not None:
                s = d["iRun"] == iRun
                d = {k: v[s] for k, v in d.items()}
            yield {k: d[k] for k in columns}
        return
    cfg = configs(f)
    runs = [r for r in ([iRun] if iRun is not None else sorted(cfg)) if r in cfg]
    if layout(f) == "perModel":
        for r in runs:
            for d in f[cfg[r]["name"]].iterate(columns, library="np", step_size=step):
                yield d
        return
    perCh = [c for c in columns if c not in ("iRun", "evt", "ch", "OV", "gain")]
    for r in runs:
        tree = f[cfg[r]["name"]]
        chans = channels(tree)
        branches = ["evt"] + [f"sipm{c}_{v}" for c in chans for v in dict.fromkeys(perCh + ["nPhot"])]
        for d in tree.iterate(branches, library="np", step_size=max(1, step // max(1, len(chans)))):
            E = len(d["evt"])
            mask = np.stack([d[f"sipm{c}_nPhot"] for c in chans], axis=1).reshape(-1) > 0
            out = {}
            for col in columns:
                if col == "evt":
                    v = np.repeat(d["evt"], len(chans))
                elif col == "ch":
                    v = np.tile(np.array(chans, dtype=np.int32), E)
                elif col == "iRun":
                    v = np.full(E * len(chans), r, dtype=np.int32)
                elif col in ("OV", "gain"):
                    v = np.full(E * len(chans), cfg[r][col], dtype=np.float32)
                else:
                    a = np.stack([d[f"sipm{c}_{col}"] for c in chans], axis=1)
                    v = a.reshape((E * len(chans),) + a.shape[2:])
                out[col] = v[mask]
            yield out