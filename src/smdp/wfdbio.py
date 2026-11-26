from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import wfdb
    HAVE_WFDB = True
except Exception:
    HAVE_WFDB = False


@dataclass
class Record:
    signal: np.ndarray
    fs: float
    leads: list[str]
    comments: list[str]


def _parse_header(path: Path):
    lines = [l.strip() for l in path.with_suffix(".hea").read_text().splitlines()]
    comments = [l[1:].strip() for l in lines if l.startswith("#")]
    lines = [l for l in lines if l and not l.startswith("#")]
    head = lines[0].split()
    nsig = int(head[1])
    fs = float(head[2].split("/")[0]) if len(head) > 2 else 250.0
    nsamp = int(head[3]) if len(head) > 3 else None
    sigs = []
    for l in lines[1:1 + nsig]:
        p = l.split()
        fmt = int(p[1].split("x")[0].split(":")[0].split("+")[0])
        g = p[2] if len(p) > 2 else "200"
        gain = float(g.split("(")[0].split("/")[0]) if g else 200.0
        baseline = 0
        if "(" in g:
            baseline = int(g.split("(")[1].split(")")[0])
        adczero = int(p[4]) if len(p) > 4 else 0
        if "(" not in g:
            baseline = adczero
        name = p[8] if len(p) > 8 else f"ch{len(sigs)}"
        sigs.append(dict(file=p[0], fmt=fmt, gain=gain if gain != 0 else 200.0,
                         baseline=baseline, name=name))
    return fs, nsamp, sigs, comments


def _read_dat(datfile: Path, fmt: int, nsig: int, sampfrom: int, sampto: int | None) -> np.ndarray:
    if fmt == 16:
        bytes_per = 2 * nsig
        with open(datfile, "rb") as fh:
            fh.seek(sampfrom * bytes_per)
            count = -1 if sampto is None else (sampto - sampfrom) * nsig
            raw = np.fromfile(fh, dtype="<i2", count=count)
        return raw[: (len(raw) // nsig) * nsig].reshape(-1, nsig).T.astype(np.float64)
    if fmt == 212:
        total = None if sampto is None else sampto * nsig
        nbytes = -1 if total is None else int(np.ceil(total * 1.5)) + 3
        raw = np.fromfile(datfile, dtype=np.uint8, count=nbytes).astype(np.int32)
        raw = raw[: (len(raw) // 3) * 3].reshape(-1, 3)
        s1 = raw[:, 0] + ((raw[:, 1] & 0x0F) << 8)
        s2 = raw[:, 2] + ((raw[:, 1] & 0xF0) << 4)
        s = np.empty(2 * len(raw), dtype=np.int32)
        s[0::2], s[1::2] = s1, s2
        s[s > 2047] -= 4096
        s = s[: (len(s) // nsig) * nsig].reshape(-1, nsig).T.astype(np.float64)
        return s[:, sampfrom:sampto]
    raise NotImplementedError(f"WFDB format {fmt} not supported by fallback reader; pip install wfdb")


def read_record(path: str | Path, sampfrom: int = 0, sampto: int | None = None,
                channels: list[str] | None = None) -> Record:
    path = Path(str(path).removesuffix(".hea").removesuffix(".dat"))
    if HAVE_WFDB:
        r = wfdb.rdrecord(str(path), sampfrom=sampfrom, sampto=sampto)
        sig = r.p_signal.T
        leads = list(r.sig_name)
        rec = Record(sig, float(r.fs), leads, list(r.comments or []))
    else:
        fs, nsamp, sigs, comments = _parse_header(path)
        files = {}
        for i, s in enumerate(sigs):
            files.setdefault((s["file"], s["fmt"]), []).append(i)
        out = np.zeros((len(sigs), 0))
        parts = {}
        for (fname, fmt), idxs in files.items():
            dat = _read_dat(path.parent / fname, fmt, len(idxs), sampfrom, sampto)
            for j, i in enumerate(idxs):
                parts[i] = dat[j]
        n = min(len(v) for v in parts.values())
        out = np.stack([parts[i][:n] for i in range(len(sigs))])
        for i, s in enumerate(sigs):
            out[i] = (out[i] - s["baseline"]) / s["gain"]
        rec = Record(out, fs, [s["name"] for s in sigs], comments)
    if channels is not None:
        rec = select_leads(rec, channels)
    return rec


_ALIASES = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF", "MLII": "II", "ML2": "II"}


def canon(name: str) -> str:
    n = name.strip()
    return _ALIASES.get(n.upper(), n if n in ("aVR", "aVL", "aVF") else n.upper())


def select_leads(rec: Record, wanted: list[str]) -> Record:
    names = [canon(n) for n in rec.leads]
    idx = []
    for w in wanted:
        cw = canon(w)
        if cw not in names:
            raise KeyError(f"lead {w} not in record leads {rec.leads}")
        idx.append(names.index(cw))
    return Record(rec.signal[idx], rec.fs, [rec.leads[i] for i in idx], rec.comments)


def write_record_fmt16(path: str | Path, signal_mv: np.ndarray, fs: float, leads: list[str],
                       gain: float = 1000.0, comments: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dig = np.clip(np.round(signal_mv * gain), -32767, 32767).astype("<i2")
    dig.T.tofile(path.with_suffix(".dat"))
    lines = [f"{path.name} {len(leads)} {fs:g} {signal_mv.shape[1]}"]
    for i, l in enumerate(leads):
        lines.append(f"{path.name}.dat 16 {gain:g}(0)/mV 16 0 {int(dig[i, 0])} 0 0 {l}")
    for c in comments or []:
        lines.append(f"# {c}")
    path.with_suffix(".hea").write_text("\n".join(lines) + "\n")
