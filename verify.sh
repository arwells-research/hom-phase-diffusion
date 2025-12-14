#!/usr/bin/env bash
set -euo pipefail

echo "================ VERIFY: HOM Phase Diffusion Generator ================"

# ----------------------------------------------------------------------
# 0) Sanity: repo status, python, executability, CRLF
# ----------------------------------------------------------------------
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status --porcelain
else
  echo "[verify] NOTE: not a git repo; skipping git checks"
fi

python3 --version
file scripts/*.py scripts/*.sh | sed -n '1,120p'
chmod +x scripts/*.py scripts/*.sh

# Fix CRLF (prevents /usr/bin/env: ‘python3\r’)
sed -i 's/\r$//' scripts/*.py scripts/*.sh
head -n 1 scripts/export_paper_numbers.py
head -n 1 scripts/dft_phase_diffusion.py

# ----------------------------------------------------------------------
# 1) Smoke test: tiny dataset
# ----------------------------------------------------------------------
rm -f data/generated/_sanity.csv
python3 scripts/dft_phase_diffusion.py \
  --max-delay-steps 10 \
  --pairs 20000 \
  --omega-sigma 0.10 \
  --out-csv data/generated/_sanity.csv \
  --no-plot

head -n 2 data/generated/_sanity.csv
tail -n 2 data/generated/_sanity.csv

# ----------------------------------------------------------------------
# 2) Analysis smoke
# ----------------------------------------------------------------------
python3 scripts/run_diffusion_analysis.py \
  --c-min 2e-2 --max-N 80 \
  data/generated/_sanity.csv | sed -n '1,160p'

# ----------------------------------------------------------------------
# 3) Full suite run (baseline)
# ----------------------------------------------------------------------
rm -rf outputs/logs outputs/paper
mkdir -p outputs/logs outputs/paper

C_MIN=2e-2 MAX_N=80 PAIRS=200000 PYTHON=python3 \
  scripts/run_diffusion_suite.sh | tee outputs/logs/_suite_stdout.log

# ----------------------------------------------------------------------
# 4) Exporter
# ----------------------------------------------------------------------
python3 scripts/export_paper_numbers.py \
  --c-min 2e-2 --max-N 80 | tee outputs/logs/_export_stdout.log

ls -lh outputs/paper/
sed -n '1,160p' outputs/paper/paper_table_diffusion.tex

# ----------------------------------------------------------------------
# 5) Internal consistency: JSON ↔ table
# ----------------------------------------------------------------------
python3 - <<'PY'
import json, re
from pathlib import Path

d=json.load(open("outputs/paper/paper_numbers.json"))
keys=sorted(d["diffusion"].keys())
print("n_keys =", len(keys))

bad=[k for k in keys if not re.match(r"^(omega|pure_diff)_sigma_(NA|\d+\.\d+)_N\d+(_dense)?$", k)]
print("bad_keys =", bad)
if bad:
    raise SystemExit("FAIL: bad JSON keys")

tex=Path("outputs/paper/paper_table_diffusion.tex").read_text().splitlines()

row_rx = re.compile(
    r"""^
        (?:\\texttt\{)?                      # optional \texttt{
        (omega|pure\\?_diff)                 # omega | pure_diff | pure\_diff
        (?:\})?                              # optional closing }
        \s*&                                 # first column terminator
    """,
    re.VERBOSE,
)

rows=[]
for ln in tex:
    s=ln.strip()
    if s.endswith(r"\\") and row_rx.match(s):
        rows.append(s)

print("table_rows =", len(rows))
if len(rows) != len(keys):
    raise SystemExit(f"FAIL: table_rows({len(rows)}) != n_keys({len(keys)})")
print("OK: JSON ↔ table consistency")
PY

# ----------------------------------------------------------------------
# 6) Determinism check (hashes)
# ----------------------------------------------------------------------
ROOT_DIR="$(pwd)"
GEN_DIR="$ROOT_DIR/data/generated"

echo "[verify] phase_diff CSV count before hashing:"
n_csv=$(ls -1 "$GEN_DIR"/phase_diff_*.csv | wc -l)
echo "$n_csv"
test "$n_csv" -eq 8 || { echo "FAIL: expected 8 CSVs"; exit 1; }

python3 - <<'PY' "$GEN_DIR" > outputs/logs/_hashes_run1.txt
import hashlib, glob, os, sys
gen=sys.argv[1]
paths=sorted(glob.glob(os.path.join(gen,"phase_diff_*.csv")))
print("csvs =", len(paths))
def h(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()
for p in paths:
    print(h(p), os.path.basename(p))
PY

C_MIN=2e-2 MAX_N=80 PAIRS=200000 PYTHON=python3 \
  scripts/run_diffusion_suite.sh >/dev/null

python3 - <<'PY' "$GEN_DIR" > outputs/logs/_hashes_run2.txt
import hashlib, glob, os, sys
gen=sys.argv[1]
paths=sorted(glob.glob(os.path.join(gen,"phase_diff_*.csv")))
print("csvs =", len(paths))
def h(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()
for p in paths:
    print(h(p), os.path.basename(p))
PY

diff -u outputs/logs/_hashes_run1.txt outputs/logs/_hashes_run2.txt \
  && echo "OK: deterministic CSV outputs"

# ----------------------------------------------------------------------
# 7) Physics sanity checks (fit-based, paper-aligned)
# ----------------------------------------------------------------------
python3 - <<'PY'
import json, math, numpy as np, pandas as pd

# ---- A) intrinsic diffusion slope ----
df=pd.read_csv("data/generated/phase_diff_pure_diff_N200.csv")
N=df["N_delay_steps"].to_numpy()
var=df["var_dtheta"].to_numpy()
mask=(N>=20)&(N<=120)
D_mid=np.polyfit(N[mask],var[mask],1)[0]
print("pure_diff var slope (20..120) =", D_mid)
if not (0.95 <= D_mid <= 1.02):
    raise SystemExit("FAIL: intrinsic diffusion slope out of range")

# ---- B) fitted coherence monotonicity ----
d=json.load(open("outputs/paper/paper_numbers.json"))
k="pure_diff_sigma_NA_N200" if "pure_diff_sigma_NA_N200" in d["diffusion"] else "pure_diff_sigma_NA_N50"
mix=d["diffusion"][k]["analysis"]["coherence"]["mixed"]
A=float(mix.get("A",1.0))
alpha=float(mix["alpha"])
beta=float(mix["beta"])
Ns=np.arange(0,51)
cfit=A*np.exp(-alpha*Ns-beta*Ns*Ns)
if not np.all(np.diff(cfit)<=1e-15):
    raise SystemExit("FAIL: fitted coherence not monotone")
print("OK: fitted coherence monotone")
print("cfit[0],cfit[10],cfit[50] =",cfit[0],cfit[10],cfit[50])
PY

# 7b) generator sanity: intrinsic diffusion D matches paper value
python3 - <<'PY'
import numpy as np
import pandas as pd

# Load pure diffusion data
df = pd.read_csv("data/generated/phase_diff_pure_diff_N200.csv")
N = df["N_delay_steps"].to_numpy()
var = df["var_dtheta"].to_numpy()

# Fit over the same midrange used elsewhere
mask = (N >= 20) & (N <= 120)
D_est = np.polyfit(N[mask], var[mask], 1)[0]

# Paper value
D_paper = 0.985
tol = 0.015   # conservative finite-N tolerance

print(f"Intrinsic diffusion check:")
print(f"  D_est   = {D_est:.6f}")
print(f"  D_paper= {D_paper:.6f}")
print(f"  |Δ|     = {abs(D_est - D_paper):.6f}")

if abs(D_est - D_paper) > tol:
    raise SystemExit("FAIL: intrinsic diffusion D deviates from paper value")

print("OK: intrinsic diffusion coefficient consistent with paper")
PY

# ----------------------------------------------------------------------
# 8) Exporter echo (paper traceability)
# ----------------------------------------------------------------------
python3 - <<'PY'
import json
d=json.load(open("outputs/paper/paper_numbers.json"))
for k in sorted(d["diffusion"]):
    out=d["diffusion"][k]["analysis"]
    b=out["variance"]["mixed"]["b"]
    m=out["coherence"]["mixed"]
    if m:
        print(k,"b=",b,"alpha=",m["alpha"],"beta=",m["beta"],"N_used=",m["N_used"])
    else:
        print(k,"b=",b,"coherence=None")
PY

echo "================ VERIFY COMPLETE: PASS ================="