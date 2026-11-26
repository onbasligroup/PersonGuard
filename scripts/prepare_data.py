import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PN = "https://physionet.org/files"
S3 = {
    "mitbih": ("mitdb/1.0.0", "mitbih"),
    "ptbxl": ("ptb-xl/1.0.3", "ptbxl"),
    "ecg_arrhythmia": ("ecg-arrhythmia/1.0.0", "ecg_arrhythmia"),
    "icentia11k": ("icentia11k-continuous-ecg/1.0", "icentia11k"),
}
CINC_DIRS = r"training/(georgia|cpsc_2018|cpsc_2018_extra|st_petersburg_incart)/"
ARIA = ["aria2c", "-c", "--auto-file-renaming=false", "--max-tries=0", "--retry-wait=15", "--timeout=60",
        "--connect-timeout=60", "--console-log-level=warn", "--summary-interval=300"]

REPORT_RULES = [
    ("MI", r"infarct|myocardial infarction|\bmi\b|q waves? (?:present|in)|pathologic q"),
    ("STTC", r"st (?:segment )?(?:elevation|depression)|st-t|st & t|t wave (?:abnormal|inversion|changes)|"
             r"nonspecific t|repolarization|ischemi|st changes"),
    ("CD", r"bundle branch block|\brbbb\b|\blbbb\b|fascicular block|av block|a-v block|"
           r"\d(?:st|nd|rd)? degree (?:a-?v )?block|intraventricular conduction|ivcd|wpw|pre-?excitation|"
           r"prolonged pr|bifascicular|trifascicular"),
    ("HYP", r"ventricular hypertrophy|\blvh\b|\brvh\b|atrial enlargement|atrial abnormality|"
            r"increased voltage|high voltage"),
    ("AF", r"atrial fibrillation|atrial flutter"),
]
NORMAL = r"normal ecg|within normal limits"
ICD_MAP = {
    "I21": "MI", "I22": "MI", "I252": "MI", "410": "MI", "412": "MI",
    "I48": "AF", "42731": "AF",
    "I50": "HF", "428": "HF",
    "I44": "CD", "I45": "CD", "4260": "CD", "4261": "CD", "4263": "CD", "4264": "CD", "4265": "CD",
    "I11": "HYP", "I13": "HYP", "I42": "HYP", "402": "HYP", "404": "HYP", "425": "HYP",
}


def run(cmd, cwd=None):
    print(" ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), cwd=cwd).returncode == 0


def aws():
    return ["aws"] if shutil.which("aws") else [sys.executable, "-m", "awscli"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def verify(folder, physionet_path, pattern=None):
    folder = Path(folder)
    sums = folder / "SHA256SUMS.txt"
    if not sums.exists():
        urllib.request.urlretrieve(f"{PN}/{physionet_path}/SHA256SUMS.txt", sums)
    bad = 0
    for line in sums.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, name = parts[0], parts[1].lstrip("*").removeprefix("./")
        if pattern and not re.search(pattern, name):
            continue
        p = folder / name
        if not p.exists() or sha256(p) != digest:
            print(f"FAILED {p}", file=sys.stderr)
            bad += 1
    if bad:
        print(f"### {folder}: {bad} files failed, run the command again", file=sys.stderr)
        return False
    print(f"### {folder}: checksums OK")
    return True


def copy_missing(src, dst):
    for p in Path(src).rglob("*"):
        q = Path(dst) / p.relative_to(src)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif not q.exists():
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)


def download(dest, datasets):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ok = True
    for ds in datasets:
        print(f"=================== {ds} ===================", flush=True)
        if ds in S3:
            path, folder = S3[ds]
            sync = aws() + ["s3", "sync", "--no-sign-request", "--region", "us-east-1", "--only-show-errors",
                            f"s3://physionet-open/{path}/", folder]
            ok &= run(sync, cwd=dest) and verify(dest / folder, path)
        elif ds == "mimic_ecg_meta":
            (dest / "mimic_ecg").mkdir(exist_ok=True)
            ok &= run(ARIA + ["-x", "4", "-d", "mimic_ecg", "-o", "record_list.csv",
                              f"{PN}/mimic-iv-ecg/1.0/record_list.csv"], cwd=dest)
            ok &= run(ARIA + ["-x", "8", "-d", "mimic_ecg", "-o", "machine_measurements.csv",
                              f"{PN}/mimic-iv-ecg/1.0/machine_measurements.csv"], cwd=dest)
        elif ds == "mimic_ecg":
            if not run(ARIA + ["-x", "16", "-s", "16", "-k", "10M", "-o", "mimic-iv-ecg-1.0.zip",
                               "https://physionet.org/content/mimic-iv-ecg/get-zip/1.0/"], cwd=dest):
                ok = False
                continue
            marker = dest / "mimic_ecg" / "files" / ".unzipped"
            if not marker.exists():
                tmp = dest / "mimic_tmp"
                with zipfile.ZipFile(dest / "mimic-iv-ecg-1.0.zip") as z:
                    z.extractall(tmp)
                for top in tmp.iterdir():
                    copy_missing(top, dest / "mimic_ecg")
                shutil.rmtree(tmp)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
            if verify(dest / "mimic_ecg", "mimic-iv-ecg/1.0"):
                print("mimic-iv-ecg-1.0.zip can now be deleted")
            else:
                ok = False
        elif ds == "cinc2021":
            folder = dest / "cinc2021"
            folder.mkdir(exist_ok=True)
            sums = folder / "SHA256SUMS.txt"
            if not sums.exists():
                urllib.request.urlretrieve(f"{PN}/challenge-2021/1.0.3/SHA256SUMS.txt", sums)
            lines = []
            for line in sums.read_text().splitlines():
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    name = parts[1].lstrip("*").removeprefix("./")
                    if re.match(CINC_DIRS, name):
                        lines.append(f"{PN}/challenge-2021/1.0.3/{name}\n  out={name}")
            (folder / "cinc_list.txt").write_text("\n".join(lines) + "\n")
            print(f"files to fetch: {len(lines)}")
            run(ARIA + ["-i", "cinc_list.txt", "-j", "16"], cwd=folder)
            ok &= verify(folder, "challenge-2021/1.0.3", CINC_DIRS)
        else:
            print(f"unknown dataset {ds}")
    return ok


def classify_report(statements):
    text = " | ".join(s.lower() for s in statements if isinstance(s, str))
    found = {c for c, rx in REPORT_RULES if re.search(rx, text)}
    if re.search(r"(?:no|not) (?:evidence of )?(?:acute )?infarct", text):
        found.discard("MI")
    if not found and re.search(NORMAL, text):
        found = {"NORM"}
    return found


def labels(ecg, out, max_records):
    import pandas as pd
    root = Path(ecg)
    rl = pd.read_csv(root / "record_list.csv", usecols=["subject_id", "study_id", "ecg_time"])
    mm = pd.read_csv(root / "machine_measurements.csv")
    rep_cols = [c for c in mm.columns if c.startswith("report_")]
    mm["classes"] = mm[rep_cols].apply(lambda r: classify_report(r.tolist()), axis=1)
    df = rl.merge(mm[["study_id", "classes"]], on="study_id", how="left")
    df = df.sort_values(["subject_id", "ecg_time"]).groupby("subject_id").head(max_records)
    lab = df.groupby("subject_id")["classes"].apply(lambda s: set().union(*[x for x in s if isinstance(x, set)]))
    res = pd.DataFrame({"subject_id": lab.index,
                        "labels": [("NORM" if c == {"NORM"} else "|".join(sorted(c - {"NORM"}))) for c in lab]})
    res.to_csv(out, index=False)
    print(res.labels.str.split("|").explode().value_counts())
    print(f"-> {out}  ({len(res)} subjects)")


def icd_class(code):
    for k in sorted(ICD_MAP, key=len, reverse=True):
        if code.startswith(k):
            return ICD_MAP[k]
    return None


def labels_icd(hosp, ecg, out, window_days):
    import pandas as pd
    h = Path(hosp)
    pat = pd.read_csv(h / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    dx = pd.read_csv(h / "diagnoses_icd.csv.gz", usecols=["subject_id", "hadm_id", "icd_code"], dtype={"icd_code": str})
    if window_days is not None:
        adm = pd.read_csv(h / "admissions.csv.gz", usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
                          parse_dates=["admittime", "dischtime"])
        rec = pd.read_csv(Path(ecg) / "record_list.csv", usecols=["subject_id", "ecg_time"], parse_dates=["ecg_time"])
        adm = adm.join(rec.groupby("subject_id").ecg_time.min().rename("t0"), on="subject_id")
        w = pd.Timedelta(days=window_days)
        adm = adm[(adm.admittime <= adm.t0 + w) & (adm.dischtime >= adm.t0 - w)]
        dx = dx[dx.hadm_id.isin(adm.hadm_id)]
    dx["cls"] = dx.icd_code.map(icd_class)
    lab = dx.dropna(subset=["cls"]).groupby("subject_id").cls.apply(lambda s: "|".join(sorted(set(s))))
    subjects = pd.read_csv(Path(ecg) / "record_list.csv", usecols=["subject_id"]).subject_id.unique()
    res = pat[pat.subject_id.isin(subjects)].rename(columns={"gender": "sex", "anchor_age": "age"})
    res["labels"] = res.subject_id.map(lab).fillna("")
    res.to_csv(out, index=False)
    print(res.labels.str.split("|").explode().value_counts())


def featurize(a):
    from smdp.features import merge_shards, run_shard
    if a.merge:
        merge_shards(Path(a.out) / (a.name or a.dataset), allow_read_errors=a.allow_read_errors)
    else:
        run_shard(a.dataset, a.root, a.out, name=a.name, shard=a.shard, num_shards=a.num_shards, workers=a.workers,
                  limit=a.limit, want_csd=False, iterator_kwargs=json.loads(a.iter_kwargs), tag=a.tag)


def main():
    ap = argparse.ArgumentParser(description="Download ECG collections, build MIMIC labels and extract features")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download and verify collections")
    d.add_argument("dest")
    d.add_argument("datasets", nargs="+",
                   help="mitbih ptbxl ecg_arrhythmia icentia11k cinc2021 mimic_ecg mimic_ecg_meta")

    lb = sub.add_parser("labels", help="MIMIC-IV-ECG labels from machine reports")
    lb.add_argument("--ecg", required=True)
    lb.add_argument("--out", default="mimic_labels.csv")
    lb.add_argument("--max-records", type=int, default=8)

    li = sub.add_parser("labels-icd", help="MIMIC-IV-ECG labels from MIMIC-IV hospital diagnoses")
    li.add_argument("--hosp", required=True)
    li.add_argument("--ecg", required=True)
    li.add_argument("--out", default="mimic_labels.csv")
    li.add_argument("--window-days", type=float, default=None)

    f = sub.add_parser("featurize", help="extract spectral features")
    f.add_argument("--dataset", required=True)
    f.add_argument("--root", default=None)
    f.add_argument("--out", default="features")
    f.add_argument("--name", default=None)
    f.add_argument("--shard", type=int, default=0)
    f.add_argument("--num-shards", type=int, default=1)
    f.add_argument("--workers", type=int, default=None)
    f.add_argument("--limit", type=int, default=None)
    f.add_argument("--iter-kwargs", default="{}")
    f.add_argument("--tag", default=None)
    f.add_argument("--merge", action="store_true")
    f.add_argument("--allow-read-errors", action="store_true")

    a = ap.parse_args()
    if a.cmd == "download":
        sys.exit(0 if download(a.dest, a.datasets) else 1)
    if a.cmd == "labels":
        labels(a.ecg, a.out, a.max_records)
    elif a.cmd == "labels-icd":
        labels_icd(a.hosp, a.ecg, a.out, a.window_days)
    else:
        featurize(a)


if __name__ == "__main__":
    main()
