"""
lab5015model.py - Python version of the Lab5015SiPMModel library
(src/Functions.cc, src/Convolution.cc, src/FitUtils.cc and the CfgManager cfg format).

Used by createWaveformsFromNtuple.py and drawPulseShape.py.
"""
import re
import sys
import numpy as np

# ------------------------------------------------------------------ time grid (as in the main programs)
NPT  = int(2 ** 13)
TMAX = 409.6
FREQ = NPT / TMAX
DT   = TMAX / NPT
TT   = np.arange(NPT + 1) * DT          # ipt = 0 .. npt (the last one is the overflow bin, as in C++)
TC   = (np.arange(NPT) + 0.5) * DT      # bin centres


# ================================================================== CfgManager
class Cfg:
    """Reads the CfgManager format: <Block> key value(s) </Block>, '#' comments.
    Keys are addressed as in C++: opts.get('Input.label'), opts.exists('Input.maxEvents')."""

    def __init__(self, path):
        self.path = path
        self.d = {}
        block = None
        for raw in open(path):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^<\s*(/?)\s*([^>]+?)\s*>$", line)
            if m:
                block = None if m.group(1) else m.group(2)
                if block is not None:
                    self.d.setdefault(block, {})
                continue
            if block is None:
                continue
            p = line.split()
            self.d[block][p[0]] = p[1:]
        print(f"> CfgManager --- INFO: parsing {path}")

    def exists(self, key):
        b, k = key.split(".", 1)
        return b in self.d and k in self.d[b]

    def get(self, key, typ=str, many=False):
        b, k = key.split(".", 1)
        if not self.exists(key):
            sys.exit(f"> CfgManager --- ERROR: option {key} not found")
        v = self.d[b][k]
        return [typ(x) for x in v] if many else typ(v[0])


# ================================================================== SiPM parameters (GetSiPMParsFromCfg)
SIPM_KEYS = ("Npe", "OV", "Rq", "Cq", "Rd", "Cd", "Nc", "Cg", "RL", "RF", "fitYMax")


def GetSiPMParsFromCfg(opts):
    """returns (list of SiPM parameter dicts, list of 'run' strings), as GetSiPMParsFromCfg"""
    vec, runs = [], []
    for name in opts.get("Input.SiPMList", many=True):
        p = {"name": name}
        for k in SIPM_KEYS:
            v = opts.get(f"{name}.{k}", float, many=True)
            p[k] = v[0]
            if k in ("Rq", "Cq", "Rd", "Cd", "Cg", "RL", "RF"):
                p[k + "Err"] = v[1] if len(v) > 1 else 0.
        p["title"] = " ".join(opts.get(f"{name}.title", many=True))
        p["label"] = opts.get(f"{name}.label")
        vec.append(p)
        runs.append(opts.get(f"{name}.run"))
    return vec, runs


# ================================================================== fit-parameter bookkeeping (GetFitParsFromCfg)
def CountUnique(v):
    return len({x for x in v if x >= 0})


def RemapPars(params):
    ret, it = {}, 0
    for p in params:
        if p >= 0 and p not in ret:
            ret[p] = it
            it += 1
    return ret


def GetFitParsFromCfg(opts, nRuns):
    """returns parIndex {k + 7*iRun: Minuit index or -1} and the number of parameters of each type
    (amp, t0, Rq, Cd, BW, BW2, BW3), exactly as Functions.cc"""
    keys = ["paramAmp", "paramT0", "paramRq", "paramCd", "paramBW", "paramBW2", "paramBW3"]
    names = opts.get("Input.SiPMList", many=True)
    params = [[opts.get(f"{n}.{k}", int) for n in names] for k in keys]
    nPars = [CountUnique(p) for p in params]
    remap = [RemapPars(p) for p in params]
    offs = np.concatenate([[0], np.cumsum(nPars)])
    parIndex = {}
    print("-------- FIT PARAMETERS --------")
    for k, nm in enumerate(["amp", "t0", "Rq", "Cd", "BW", "BW2", "BW3"]):
        print(f"nPars_{nm}: {nPars[k]}")
    for iRun in range(nRuns):
        for k in range(7):
            v = params[k][iRun]
            parIndex[k + 7 * iRun] = int(offs[k] + remap[k][v]) if v >= 0 else -1
        print(f"iRun: {iRun}  parIndex = " + " ".join(str(parIndex[k + 7 * iRun]) for k in range(7)))
    print("--------------------------------")
    return parIndex, nPars


# ================================================================== Functions.cc
def funcLP(xx, tau):
    """exp(-x/tau)/tau ; tau = inf gives 0 (as in C++ with a zero bandwidth)"""
    xx = np.asarray(xx, dtype=float)
    if not np.isfinite(tau):
        return np.zeros_like(xx)
    return np.exp(-xx / tau) / tau


def funcScint(xx, tau_r, tau_d):
    return (np.exp(-xx / tau_d) - np.exp(-xx / tau_r)) / (tau_d - tau_r)


def SiPMPulseShape(x, p, amp, x0):
    """SiPM current on an external load (equivalent circuit: Rq, Cq, Rd, Cd, Nc, Cg, RL)"""
    x = np.asarray(x, dtype=float)
    Ge = (p["OV"] + 0.25) * (p["Cq"] + p["Cd"])
    tmr = p["Rd"] * (p["Cq"] + p["Cd"]) * 1e9
    tmd = p["Cd"] * (p["Rd"] * p["Rq"]) / (p["Rd"] + p["Rq"]) * 1e9
    t1 = p["Rq"] * (p["Cq"] + p["Cd"]) * 1e9
    t2 = p["Rq"] * p["Cq"] * 1e9
    t3 = p["RL"] * p["Nc"] * p["Cd"] * 1e9
    t4 = p["RL"] * p["Cg"] * 1e9
    s = t1 + t3 + t4
    r = np.sqrt(s * s - 4 * (t1 * t4 + t2 * t3))
    tcd1, tcd2 = 0.5 * (s + r), 0.5 * (s - r)
    a1 = (tcd1 * (tcd1 - t2)) / ((tcd1 - tcd2) * (tcd1 - tmd) * (tcd1 - tmr))
    a2 = (tcd2 * (tcd2 - t2)) / ((tcd2 - tcd1) * (tcd2 - tmd) * (tcd2 - tmr))
    a3 = (tmd * (tmd - t2)) / ((tmd - tcd1) * (tmd - tcd2) * (tmd - tmr))
    a4 = (tmr * (tmr - t2)) / ((tmr - tcd1) * (tmr - tmd) * (tmr - tcd2))
    xs = x - x0
    with np.errstate(over="ignore", invalid="ignore"):
        IL = amp * Ge * 1e9 * (a1 * np.exp(-xs / tcd1) + a2 * np.exp(-xs / tcd2)
                               + a3 * np.exp(-xs / tmd) + a4 * np.exp(-xs / tmr))
    return np.where(x >= x0, IL, 1e-15)


def lowpass_bins(tau):
    """bin contents 1/freq*(1-0.5/(freq*tau))*funcLP(tt,tau) for ipt = 0..npt-1"""
    if not np.isfinite(tau):
        return np.zeros(NPT)
    return 1. / FREQ * (1. - 0.5 / (FREQ * tau)) * funcLP(TT[:NPT], tau)


def highpass_bins(tau):
    """delta - low-pass, as the ipt == 0 / else branches"""
    k = -lowpass_bins(tau)
    k[0] += 1.
    return k


# ================================================================== Convolution.cc
def hConvol(a, b):
    """hFFT = a (x) b : circular convolution of two arrays of NPT bins (same as the ROOT FFT version)"""
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=len(a))


# ================================================================== FitUtils.cc
def FindSmallestInterval(vals, fraction=0.68):
    v = np.sort(np.asarray(vals, dtype=float))
    n = len(v)
    m = int(fraction * n)
    best = (0., 0.)
    delta = 999999.
    for i in range(n - m):
        lo = v[i]
        hi = v[i + m - 1] if i + m > 0 else 1.
        if hi - lo < delta:
            delta = hi - lo
            best = (lo, hi)
    return best


# ================================================================== ROOT helpers
def Interpolate(contents, x):
    """TH1::Interpolate on a histogram with bin centres TC"""
    return np.interp(x, TC, contents)          # constant beyond the first / last centre, as ROOT