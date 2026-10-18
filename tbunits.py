"""
tbunits.py - give simulation thresholds in the TEST-BEAM convention (TOFHIR DAC steps ith1 / ith2)

    --thrDAC 1 3 5 7 10 15      thresholds as in the logbook (ith1 values by default)
    --tbCalib tbCalib.cfg        per-SiPM-model conversion values (blocks named like the trees)

Conversion to the waveform's model units, for every SiPM model (tree) separately:

  A) no p.e. calibration:   I = ith x LSB(range)  [uA]
                            threshold = I x uaScale x attenuatorScale(attenuator)   [model units]
     The model's "uA" is the MTD normalisation (x1.725e6), not checked for SND.

  B) with peDAC:            peDAC = measured DAC steps per p.e. (dark-count staircase, noise run, ...)
                            threshold = ith / peDAC x amp1pe   [model units]
     -> "ith1 = 5" is the same number of photoelectrons in data and simulation, independent of MTD.

AUTO-DETECTION:  --tbRun /path/run0305_e.root   (the TOFHIR data file of the run to compare with)
    reads from the data file: OV values (step1), ith1/ith2/ithe used (step2), TOFHIR channels (channelID)
    finds the run's TOFHIR config next to it (config_runrun0305.ini, config_run0305.ini, ... in the same
    folder, its parent, or --tbConfigDir) and reads range_t1, range_t2, attenuator_gain, DLED code,
    then disc_calibration.tsv (path from the ini, else next to the ini / in --tbConfigDir) for the zeros.
    Without --thrDAC, the run's own ith1 (or ith2 with --branch T2) values become the thresholds.
    NOT detectable: peDAC (DAC steps per p.e.) - it must be measured; give it in --tbCalib or --peDAC.

Where the values come from, for a model whose tree is called e.g. SiPM_25um_OV4 (later wins):
    1. built-in defaults (SND TB 2026: range_t1 = 1, range_t2 = 0)
    2. auto-detected from --tbRun (data file, its TOFHIR ini, disc_calibration.tsv)
    3. block <default> of the --tbCalib file
    4. a block with the same OV as the model   (e.g. <OV4>  OV 4  ...)
    5. the block named exactly like the tree   (<SiPM_25um_OV4> ... </SiPM_25um_OV4>)
    6. options given on the command line
  inside a block, "tofhirIni <file>" reads range_t1, range_t2, attenuator_gain (and the DLED delay code)
  from the TOFHIR config the run was taken with; keys written in the block itself win over the ini.
  "discCalib <file>" + "channels ..." read the per-channel zeros: ith is clipped at 63 - zero - 1 by TOFHIR.

Calibration file format (same style as the other cfg files; '#' = comment):
    <default>
    branch T1
    rangeT1 1
    rangeT2 0
    attenuatorScale 0:1.0 2:1.0      # model units per uA for each attenuator_gain setting (fill when known)
    </default>
    <SiPM_25um_OV4>
    tofhirIni  /path/config_runrun0305.ini
    discCalib  /path/disc_calibration.tsv
    channels   128 129 130 131 132 133 134 135
    peDAC      3.0                   # DAC steps per p.e. at this OV (measured)
    </SiPM_25um_OV4>

DAC step per range (TOFHIR2, as in Lab5015 drawPulseShape):
    T1 (ith1): range 0 0.156, 1 0.313, 2 0.470, 3 0.630 uA
    T2 (ith2): range 0 0.313, 1 0.630, 2 0.940, 3 1.250 uA
"""
import os
import re

LSB_T1 = {0: 0.156, 1: 0.313, 2: 0.470, 3: 0.630}
LSB_T2 = {0: 0.313, 1: 0.630, 2: 0.940, 3: 1.250}


# ------------------------------------------------------------------ readers
def parse_cfg(path):
    """<block> key value ... </block>  ->  {block: {key: [values]}}"""
    blocks, cur = {}, None
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"<\s*(/?)\s*([^>\s]+)\s*>", line)
        if m:
            cur = None if m.group(1) else m.group(2)
            if cur is not None:
                blocks.setdefault(cur, {})
            continue
        if cur is None:
            continue
        parts = line.split()
        blocks[cur][parts[0]] = parts[1:]
    return blocks


def _int(v):
    v = v.strip().strip('"').strip("'")
    return int(v, 0) if re.match(r"^0[bxo]", v, re.I) else int(float(v))


def parse_ini(path):
    """TOFHIR config (key = value, sections ignored): returns the values the conversion needs"""
    vals = {}
    for line in open(path):
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        if line.startswith("["):
            continue
        if "=" not in line:
            continue
        k, v = (s.strip() for s in line.split("=", 1))
        k = k.lower()
        try:
            if k.endswith("range_t1"):
                vals["rangeT1"] = _int(v)
            elif k.endswith("range_t2"):
                vals["rangeT2"] = _int(v)
            elif k.endswith("attenuator_gain"):
                vals["attenuator"] = _int(v)
            elif k.endswith("dcr_delay_t"):
                vals["dledCode"] = _int(v)
            elif k.endswith("dcr_delay_t_high"):
                vals["dledHigh"] = _int(v)
            elif "disc_calibration" in k:
                vals["discTable"] = v.strip().strip('"').strip("'")
        except ValueError:
            pass
    return vals


def parse_disc(path):
    """disc_calibration.tsv: port slave chip ch baseline_T baseline_E zero_T1 zero_T2 -> {chip*32+ch: (z1, z2)}"""
    z = {}
    for line in open(path):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        p = line.split()
        try:
            port, slave, chip, ch = (int(x) for x in p[:4])
            z[chip * 32 + ch] = (float(p[6]), float(p[7]))
        except (ValueError, IndexError):
            continue
    return z


def autodetect(runFile, configDir=None, tree="data"):
    """everything that can be read from a TOFHIR run: returns {key: (value, source)}"""
    import glob
    import numpy as np
    out = {}
    m = re.search(r"run0*(\d+)", os.path.basename(runFile))
    run = int(m.group(1)) if m else None
    out["run"] = (run, os.path.basename(runFile))
    # ---- the data file
    try:
        import uproot
        import awkward as ak
        t = uproot.open(runFile)[tree]
        d = t.arrays(["step1", "step2"], library="np")
        s2 = np.round(d["step2"]).astype(int)
        out["runOV"] = (sorted(set(np.round(d["step1"], 3).tolist())), os.path.basename(runFile) + " step1")
        out["runIth1"] = (sorted(set((s2 // 10000 - 1).tolist())), os.path.basename(runFile) + " step2")
        out["runIth2"] = (sorted(set(((s2 // 100) % 100 - 1).tolist())), os.path.basename(runFile) + " step2")
        out["runIthe"] = (sorted(set((s2 % 100 - 1).tolist())), os.path.basename(runFile) + " step2")
        ch = np.unique(ak.to_numpy(ak.flatten(t["channelID"].array(entry_stop=min(t.num_entries, 200000)))))
        out["channels"] = (ch.astype(int).tolist(), os.path.basename(runFile) + " channelID")
    except Exception as e:
        print(f">>> --tbRun: could not read {runFile}: {e}")
    # ---- the TOFHIR config of the run
    dirs = [os.path.dirname(os.path.abspath(runFile))]
    dirs.append(os.path.dirname(dirs[0]))
    if configDir:
        dirs.insert(0, configDir)
    pats = ([f"config_runrun{run:04d}.ini", f"config_runrun{run}.ini", f"config_run{run:04d}.ini",
             f"config_run{run}.ini", f"*run*{run:04d}*.ini", f"*run*{run}*.ini"] if run is not None else []) + ["config*.ini"]
    ini = None
    for pat in pats:
        for d0 in dirs:
            for sub in ("", "*"):
                hits = sorted(glob.glob(os.path.join(d0, sub, pat)))
                if hits:
                    ini = hits[0]
                    break
            if ini: break
        if ini: break
    if ini:
        for k, v in parse_ini(ini).items():
            if k != "discTable":
                out[k] = (v, os.path.basename(ini))
        cands = []
        tab = parse_ini(ini).get("discTable")
        if tab:
            cands.append(tab if os.path.isabs(tab) else os.path.join(os.path.dirname(ini), tab))
        cands += [os.path.join(os.path.dirname(ini), "disc_calibration.tsv")]
        if configDir:
            cands.append(os.path.join(configDir, "disc_calibration.tsv"))
        for c in cands:
            if os.path.exists(c):
                out["discCalib"] = (c, "found via " + os.path.basename(ini))
                break
        else:
            print(f">>> --tbRun: no disc_calibration.tsv found near {ini} (no clipping at 63 - zero - 1)")
    else:
        print(f">>> --tbRun: no TOFHIR config (config_runrun{run:04d}.ini) found in {', '.join(dirs)}; "
              f"give --tbConfigDir - using default ranges")
    return out


# ------------------------------------------------------------------ options
def add_args(ap):
    g = ap.add_argument_group("test-beam threshold convention (TOFHIR DAC), see tbunits.py")
    g.add_argument("--thrDAC", type=float, nargs="*", default=None,
                   help="thresholds in DAC steps, as ith1/ith2 in the logbook (e.g. 1 3 5 7 10 15)")
    g.add_argument("--tbRun", default=None,
                   help="TOFHIR data file of the run (runXXXX_e.root): auto-detect OV, ith, ranges, attenuator, zeros")
    g.add_argument("--tbConfigDir", default=None, help="where the run's TOFHIR config / disc_calibration.tsv are")
    g.add_argument("--tbCalib", default=None,
                   help="calibration cfg with one block per SiPM model, named like the tree (and <default>)")
    g.add_argument("--branch", choices=["T1", "T2"], default=None, help="which threshold the values are (default T1 = ith1)")
    g.add_argument("--rangeT1", type=int, default=None, help="cfg_a3_range_t1 (default 1 = SND TB -> 0.313 uA/step)")
    g.add_argument("--rangeT2", type=int, default=None, help="cfg_a3_range_t2 (default 0 = SND TB -> 0.313 uA/step)")
    g.add_argument("--uaScale", type=float, default=None, help="model units per uA (default 1 = MTD normalisation)")
    g.add_argument("--peDAC", nargs="*", default=None,
                   help="measured DAC steps per p.e.: one number, or OV:value pairs (e.g. 4:3.0 2:5.6)")


class DAC:
    """converts DAC thresholds to model units, per SiPM model (cf = dict with name, OV, amp1pe)"""

    def __init__(self, a):
        self.on = a.thrDAC is not None
        self.calib = parse_cfg(a.tbCalib) if a.tbCalib else {}
        self.calibName = a.tbCalib
        if a.tbCalib:
            print(f">>> threshold calibration {a.tbCalib}: blocks {', '.join(self.calib) or '(none)'}")
        self.cli = {}
        if a.branch is not None: self.cli["branch"] = (a.branch, "command line")
        if a.rangeT1 is not None: self.cli["rangeT1"] = (a.rangeT1, "command line")
        if a.rangeT2 is not None: self.cli["rangeT2"] = (a.rangeT2, "command line")
        if a.uaScale is not None: self.cli["uaScale"] = (a.uaScale, "command line")
        if a.peDAC: self.cli["peDAC"] = (" ".join(a.peDAC), "command line")
        self.auto = autodetect(a.tbRun, a.tbConfigDir) if a.tbRun else {}
        if self.auto:
            print(f">>> auto-detected from run {self.auto['run'][0]}:")
            for k, (v, src) in self.auto.items():
                if k == "run":
                    continue
                if k in ("dledCode", "dledHigh"):
                    v = bin(v)
                if k == "channels":
                    v = f"{len(v)} channels {v[0]}-{v[-1]}"
                print(f"      {k:10s} = {v!s:28s} from {src}")
            if a.thrDAC is None:
                br = a.branch or "T1"
                key = "runIth1" if br == "T1" else "runIth2"
                if key in self.auto:
                    a.thrDAC = [float(x) for x in self.auto[key][0]]
                    print(f">>> no --thrDAC: using the run's {'ith1' if br == 'T1' else 'ith2'} values {self.auto[key][0]}")
        self.on = a.thrDAC is not None
        self._cache = {}
        self.maxThr = max(a.thrDAC) if self.on and a.thrDAC else None

    # -------- resolve the settings of one model
    def settings(self, cf):
        name = cf["name"]
        if name in self._cache:
            return self._cache[name]
        S = {"branch": ("T1", "default"), "rangeT1": (1, "default (SND TB 2026)"),
             "rangeT2": (0, "default (SND TB 2026)"), "uaScale": (1.0, "default (MTD normalisation)")}
        for k in ("rangeT1", "rangeT2", "attenuator", "dledCode", "dledHigh", "discCalib", "channels"):
            if k in self.auto:
                S[k] = self.auto[k]
        blocks = []
        if "default" in self.calib:
            blocks.append(("default", self.calib["default"]))
        for bn, b in self.calib.items():
            if bn not in ("default", name) and "OV" in b and abs(float(b["OV"][0]) - cf["OV"]) < 0.01:
                blocks.append((bn, b))
        if name in self.calib:
            blocks.append((name, self.calib[name]))
        elif self.calib:
            print(f">>> note: no block <{name}> in {self.calibName}")
        for bn, b in blocks:
            src = f"{os.path.basename(self.calibName)} <{bn}>"
            if "tofhirIni" in b:
                ini = b["tofhirIni"][0]
                for k, v in parse_ini(ini).items():
                    S[k] = (v, os.path.basename(ini))
            for k in ("branch", "rangeT1", "rangeT2", "attenuator"):
                if k in b:
                    S[k] = ((b[k][0] if k == "branch" else _int(b[k][0])), src)
            if "uaScale" in b:
                S["uaScale"] = (float(b["uaScale"][0]), src)
            if "attenuatorScale" in b:
                S["attenuatorScale"] = ({_int(x.split(":")[0]): float(x.split(":")[1]) for x in b["attenuatorScale"]}, src)
            if "peDAC" in b:
                S["peDAC"] = (" ".join(b["peDAC"]), src)
            if "discCalib" in b:
                S["discCalib"] = (b["discCalib"][0], src)
            if "channels" in b:
                S["channels"] = ([int(x) for x in b["channels"]], src)
        S.update(self.cli)                       # command line wins
        # derived values
        br = S["branch"][0]
        S["lsb"] = ((LSB_T1 if br == "T1" else LSB_T2)[S["rangeT1" if br == "T1" else "rangeT2"][0]],
                    f"DAC step of {br} range {S['rangeT1' if br == 'T1' else 'rangeT2'][0]}")
        att = S.get("attenuator", (None, ""))[0]
        if att is not None and "attenuatorScale" in S:
            sc = S["attenuatorScale"][0].get(att)
            S["attScale"] = (sc if sc is not None else 1.0,
                             S["attenuatorScale"][1] if sc is not None else f"attenuator {att} not in attenuatorScale -> 1")
        else:
            S["attScale"] = (1.0, "no attenuator scale given" if att is not None else "-")
        S["pe"] = (self._pe_value(S.get("peDAC", (None, ""))[0], cf), S.get("peDAC", (None, "-"))[1])
        if "discCalib" in S and S.get("channels"):
            z = parse_disc(S["discCalib"][0])
            k = 0 if br == "T1" else 1
            zs = [z[c][k] for c in S["channels"][0] if c in z]
            if zs:
                S["maxIth"] = (63 - max(zs) - 1, f"63 - zero - 1, zeros of {len(zs)} channels from "
                                                  f"{os.path.basename(S['discCalib'][0])}")
        if "runOV" in self.auto and not any(abs(o - cf["OV"]) < 0.01 for o in self.auto["runOV"][0]):
            print(f">>> WARNING: model {name} has OV {cf['OV']:g}, run {self.auto['run'][0]} was taken at OV "
                  f"{self.auto['runOV'][0]} - the comparison is not like for like")
        self._cache[name] = S
        return S

    @staticmethod
    def _pe_value(v, cf):
        if v is None:
            return None
        parts = v.split()
        if len(parts) == 1 and ":" not in parts[0]:
            return float(parts[0])
        for p in parts:
            ov, x = p.split(":")
            if abs(float(ov) - cf["OV"]) < 0.01:
                return float(x)
        raise SystemExit(f">>> peDAC '{v}' has no value for OV {cf['OV']:g} ({cf['name']})")

    # -------- conversion
    def name(self, cf=None):
        return "ith1" if (self.settings(cf)["branch"][0] if cf else "T1") == "T1" else "ith2"

    @property
    def unit(self):
        return "DAC (ith)"

    def clip(self, ith, cf):
        m = self.settings(cf).get("maxIth")
        return min(ith, m[0]) if m else ith

    def to_ua(self, ith, cf):
        S = self.settings(cf)
        return self.clip(ith, cf) * S["lsb"][0]

    def to_model(self, ith, cf):
        S = self.settings(cf)
        if S["pe"][0] is not None:
            return self.clip(ith, cf) / S["pe"][0] * cf["amp1pe"]
        return self.to_ua(ith, cf) * S["uaScale"][0] * S["attScale"][0]

    def method(self, cf=None):
        if cf is None:
            return "per SiPM model, see the tables below"
        S = self.settings(cf)
        if S["pe"][0] is not None:
            return f"calibrated: {S['pe'][0]:g} DAC steps per p.e. ({S['pe'][1]})"
        return (f"uncalibrated: ith x {S['lsb'][0]} uA x uaScale {S['uaScale'][0]:g} x attenuator scale "
                f"{S['attScale'][0]:g}  (MTD normalisation)")

    def table(self, values, cf):
        S = self.settings(cf)
        out = [f"conversion: {self.method(cf)}"]
        for k in ("branch", "rangeT1", "rangeT2", "lsb", "attenuator", "attScale", "uaScale", "maxIth",
                  "dledCode", "dledHigh"):
            if k in S and (k != "rangeT2" or S["branch"][0] == "T2") and (k != "rangeT1" or S["branch"][0] == "T1"):
                v = S[k][0]
                v = bin(v) if k in ("dledCode", "dledHigh") else v
                unused = "   (not used: peDAC given)" if S["pe"][0] is not None and k in ("attScale", "uaScale") else ""
                out.append(f"  {k:10s} = {v!s:8s} from {S[k][1]}{unused}")
        nm = self.name(cf)
        for v in values:
            c = self.clip(v, cf)
            x = self.to_model(v, cf)
            out.append(f"  {nm}={v:g}" + (f" (clipped to {c:g})" if c != v else "") +
                       f" -> {self.to_ua(v, cf):.3f} uA -> {x:.3f} model u. = {x / cf['amp1pe']:.1f} p.e.")
        return out