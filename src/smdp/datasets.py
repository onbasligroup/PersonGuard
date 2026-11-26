from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .wfdbio import Record, read_record, select_leads


@dataclass
class RecordSpec:
    path: str
    kind: str = "wfdb"
    max_seconds: float | None = None


@dataclass
class Person:
    dataset: str
    person_id: str
    site: str
    records: list[RecordSpec]
    meta: dict = field(default_factory=dict)
    single_lead: str = "II"
    person_id_reliable: bool = True


def iter_ptbxl(root: str | Path) -> Iterator[Person]:
    root = Path(root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1]["diagnostic_class"].to_dict()
    for pid, g in db.groupby("patient_id"):
        g = g.sort_values("recording_date") if "recording_date" in g else g
        classes = set()
        for codes in g["scp_codes"]:
            for c in ast.literal_eval(codes):
                if c in diag and isinstance(diag[c], str):
                    classes.add(diag[c])
        first = g.iloc[0]
        meta = {
            "age": float(first.get("age", np.nan)),
            "sex": "F" if int(first.get("sex", 0)) == 1 else "M",
            "labels": "|".join(sorted(classes)),
            "strat_fold": int(first.get("strat_fold", -1)),
            "device": str(first.get("device", "")),
            "n_records": len(g),
        }
        site = f"ptbxl_site{int(first['site'])}" if "site" in g and pd.notna(first["site"]) else "ptbxl"
        recs = [RecordSpec(str(root / f)) for f in g["filename_hr"]]
        yield Person("ptbxl", f"ptbxl_{int(pid)}", site, recs, meta)


def iter_mimic_ecg(root: str | Path, labels_csv: str | None = None,
                   max_records_per_subject: int = 8, top_dirs: list[str] | None = None) -> Iterator[Person]:
    root = Path(root)
    rl = pd.read_csv(root / "record_list.csv")
    if top_dirs:
        rl = rl[rl["path"].str.split("/").str[1].isin(top_dirs)]
    lab = None
    if labels_csv:
        lab = pd.read_csv(labels_csv).set_index("subject_id")
    for sid, g in rl.groupby("subject_id"):
        g = g.sort_values("ecg_time").head(max_records_per_subject)
        meta = {"n_records": len(g)}
        if lab is not None and sid in lab.index:
            row = lab.loc[sid]
            meta.update({"age": float(row.get("age", np.nan)), "sex": row.get("sex", ""),
                         "labels": row.get("labels", "") if isinstance(row.get("labels", ""), str) else ""})
        recs = [RecordSpec(str(root / p)) for p in g["path"]]
        yield Person("mimic_ecg", f"mimic_{int(sid)}", "mimic_bidmc", recs, meta)


def iter_icentia11k(root: str | Path, max_segments: int = 16, seconds_per_segment: float = 120.0,
                    top_dirs: list[str] | None = None) -> Iterator[Person]:
    root = Path(root)
    for pdir in sorted(root.glob("p*/p*")):
        if top_dirs and pdir.parent.name not in top_dirs:
            continue
        if not pdir.is_dir():
            continue
        segs = sorted(pdir.glob(f"{pdir.name}_s*.hea"))
        if not segs:
            continue
        idx = np.linspace(0, len(segs) - 1, min(max_segments, len(segs))).round().astype(int)
        recs = [RecordSpec(str(segs[i].with_suffix("")), max_seconds=seconds_per_segment)
                for i in sorted(set(idx))]
        yield Person("icentia11k", f"icentia_{pdir.name}", "icentia_patch", recs,
                     {"n_records": len(segs)}, single_lead="first")


_SNOMED_GROUPS = {
    "426783006": "NORM",
    "164865005": "MI", "164861001": "MI", "57054005": "MI", "164917005": "MI",
    "164934002": "STTC", "59931005": "STTC", "429622005": "STTC", "164931005": "STTC",
    "413444003": "STTC", "164930006": "STTC",
    "59118001": "CD", "164909002": "CD", "713427006": "CD", "713426002": "CD", "270492004": "CD",
    "445118002": "CD", "445211001": "CD", "251146004": "CD", "195042002": "CD", "698252002": "CD",
    "164873001": "HYP", "89792004": "HYP", "67741000119109": "HYP", "446358003": "HYP",
    "164889003": "AF", "164890007": "AF",
}


def _parse_cinc_style_header(hea: Path) -> dict:
    txt = hea.read_text(errors="ignore")
    age = re.search(r"#\s*Age:\s*([\d.]+)", txt)
    sex = re.search(r"#\s*Sex:\s*(\w+)", txt)
    dx = re.search(r"#\s*Dx:\s*([\d,]+)", txt)
    codes = dx.group(1).split(",") if dx else []
    return {"age": float(age.group(1)) if age else np.nan,
            "sex": (sex.group(1)[0].upper() if sex else ""),
            "labels": "|".join(sorted({_SNOMED_GROUPS[c] for c in codes if c in _SNOMED_GROUPS})),
            "snomed": "|".join(codes)}


CINC_UNIQUE_SOURCES = ("georgia", "cpsc_2018", "cpsc_2018_extra", "st_petersburg_incart")


def iter_cinc2021(root: str | Path, sources: list[str] | None = list(CINC_UNIQUE_SOURCES)) -> Iterator[Person]:
    root = Path(root)
    for hea in sorted(root.rglob("*.hea")):
        source = hea.parent.parent.name if hea.parent.name.startswith("g") else hea.parent.name
        if sources is not None and source not in sources:
            continue
        yield Person("cinc2021", f"cinc_{source}_{hea.stem}", f"cinc_{source}",
                     [RecordSpec(str(hea.with_suffix("")), kind="cinc_mat")],
                     _parse_cinc_style_header(hea), person_id_reliable=False)


def iter_ecg_arrhythmia(root: str | Path) -> Iterator[Person]:
    root = Path(root)
    for hea in sorted(root.rglob("JS*.hea")):
        yield Person("ecg_arrhythmia", f"ecgarr_{hea.stem}", "chapman_ningbo",
                     [RecordSpec(str(hea.with_suffix("")), kind="cinc_mat")],
                     _parse_cinc_style_header(hea), person_id_reliable=True)


MITBIH_SAME_SUBJECT = {"202": "201"}


def iter_mitbih(root: str | Path) -> Iterator[Person]:
    root = Path(root)
    by_subject: dict[str, list[str]] = {}
    for hea in sorted(root.glob("*.hea")):
        if not hea.stem.isdigit():
            continue
        subj = MITBIH_SAME_SUBJECT.get(hea.stem, hea.stem)
        by_subject.setdefault(subj, []).append(str(hea.with_suffix("")))
    for subj, recs in by_subject.items():
        yield Person("mitbih", f"mitbih_{subj}", "mitbih", [RecordSpec(r) for r in recs],
                     {"n_records": len(recs)}, single_lead="II")


def iter_shhs(root: str | Path, max_seconds: float = 3600.0) -> Iterator[Person]:
    root = Path(root)
    for edf in sorted(root.rglob("shhs1-*.edf")):
        nsrrid = edf.stem.split("-")[-1]
        yield Person("shhs", f"shhs_{nsrrid}", "shhs1",
                     [RecordSpec(str(edf), kind="edf", max_seconds=max_seconds)], {},
                     single_lead="ECG")


def iter_manifest(root: str | Path) -> Iterator[Person]:
    root = Path(root)
    df = pd.read_csv(root / "manifest.csv")
    for _, row in df.iterrows():
        meta = {k: row[k] for k in df.columns if k not in ("person_id", "site", "record")}
        recs = [RecordSpec(str(root / r)) for r in str(row["record"]).split(";")]
        yield Person(root.name, str(row["person_id"]), str(row["site"]), recs, meta)


ITERATORS = {
    "ptbxl": iter_ptbxl, "mimic_ecg": iter_mimic_ecg, "icentia11k": iter_icentia11k,
    "cinc2021": iter_cinc2021, "ecg_arrhythmia": iter_ecg_arrhythmia, "mitbih": iter_mitbih, "shhs": iter_shhs, "manifest": iter_manifest,
}


def load_record(spec: RecordSpec) -> Record:
    if spec.kind == "wfdb":
        sampto = None
        if spec.max_seconds is not None:
            from .wfdbio import _parse_header
            fs = _parse_header(Path(spec.path))[0]
            sampto = int(spec.max_seconds * fs)
        try:
            return read_record(spec.path, sampto=sampto)
        except ValueError:
            return read_record(spec.path)
    if spec.kind == "cinc_mat":
        from scipy.io import loadmat
        from .wfdbio import _parse_header
        fs, _, sigs, comments = _parse_header(Path(spec.path))
        val = loadmat(spec.path + ".mat")["val"].astype(np.float64)
        for i, s in enumerate(sigs):
            val[i] = (val[i] - s["baseline"]) / s["gain"]
        return Record(val, fs, [s["name"] for s in sigs], comments)
    if spec.kind == "edf":
        import pyedflib
        with pyedflib.EdfReader(spec.path) as f:
            labels = f.getSignalLabels()
            i = [j for j, l in enumerate(labels) if l.upper().startswith("ECG")][0]
            fs = f.getSampleFrequency(i)
            n = f.getNSamples()[i]
            if spec.max_seconds:
                n = min(n, int(spec.max_seconds * fs))
            x = f.readSignal(i, 0, n)
            unit = f.getPhysicalDimension(i).lower()
        if unit in ("v",):
            x = x * 1000.0
        elif unit in ("uv",):
            x = x / 1000.0
        return Record(x[None, :], fs, ["ECG"], [])
    raise ValueError(spec.kind)


def pick_single_lead(rec: Record, lead: str) -> np.ndarray:
    if lead == "first":
        return rec.signal[0]
    return select_leads(rec, [lead]).signal[0]
