#!/usr/bin/env bash
set -euo pipefail

# paper/make_paper.sh
# Build pipeline for the DFT HOM paper:
#  - generate figures (python)
#  - build fig5_beamsplitter_geometry.pdf from make_fig5.tex
#  - export paper numbers / table tex
#  - build main PDF with latexmk (+ bibtex)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python3}"
LATEXMK="${LATEXMK:-latexmk}"

TEX_MAIN="${TEX_MAIN:-dft_hom_phase_geometry_v1.tex}"   # main TeX file lives in paper/
FIG5_TEX="${FIG5_TEX:-make_fig5.tex}"                  # fig5 tex lives in paper/

# Where your scripts likely live:
SCRIPTS_DIR="${REPO_ROOT}/scripts"

# If you want to force-copy generated PDFs into paper/, set COPY_FIGS=1 (default).
COPY_FIGS="${COPY_FIGS:-1}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  all         Build everything (figs + fig5 + numbers + main pdf)
  pdf         Build main PDF for \$TEX_MAIN (default: $TEX_MAIN)
  figs        Run python figure generators (make_fig*.py) if present
  fig5        Build fig5_beamsplitter_geometry.pdf from $FIG5_TEX
  numbers     Run scripts/export_paper_numbers.py
  clean       Remove LaTeX artifacts in paper/ (keeps PDFs by default)
  dist        Create ../dist bundle (pdf + figs + paper_numbers.json)
  distclean   Deeper clean (also removes generated PDFs in paper/)
  help        Show this help

Env overrides:
  PYTHON=python3.13
  TEX_MAIN=yourpaper.tex
  FIG5_TEX=make_fig5.tex
  COPY_FIGS=0|1       (default 1) copy generated fig PDFs into paper/
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

run_fig_generators() {
  echo "[paper] Running figure generators..."

  # 1) Prefer scripts/make_fig*.py (your repo convention)
  local any=0
  if [[ -d "$SCRIPTS_DIR" ]]; then
    shopt -s nullglob
    local gen
    for gen in "$SCRIPTS_DIR"/make_fig*.py; do
      any=1
      echo "  -> $PYTHON $(realpath --relative-to="$REPO_ROOT" "$gen")"
      "$PYTHON" "$gen"
    done
    shopt -u nullglob
  fi

  # 2) Fallback: any make_fig*.py anywhere in repo (excluding venv/.git)
  if [[ $any -eq 0 ]]; then
    echo "  (no scripts/make_fig*.py found; scanning repo for make_fig*.py)"
    mapfile -t gens < <(find "$REPO_ROOT" \
      -path "$REPO_ROOT/.git" -prune -o \
      -path "$REPO_ROOT/.venv" -prune -o \
      -path "$REPO_ROOT/venv" -prune -o \
      -name 'make_fig*.py' -type f -print)

    if [[ ${#gens[@]} -eq 0 ]]; then
      echo "  (no make_fig*.py generators found; skipping)"
    else
      local g
      for g in "${gens[@]}"; do
        echo "  -> $PYTHON $(realpath --relative-to="$REPO_ROOT" "$g")"
        "$PYTHON" "$g"
      done
    fi
  fi

  if [[ "$COPY_FIGS" == "1" ]]; then
    echo "[paper] Ensuring expected figure PDFs are available in paper/ ..."

    # List the filenames your TeX is including (add more if needed).
    local needed=(
      "fig1_var_vs_N_pure_diffusion.pdf"
      "fig2_b_vs_sigma2.pdf"
      "fig2_var_vs_N_bandwidths.pdf"
      "fig3_coherence_envelope.pdf"
      "fig4_hom_comparison.pdf"
      "fig5_beamsplitter_geometry.pdf"
    )

    local f src
    for f in "${needed[@]}"; do
      if [[ -f "${SCRIPT_DIR}/${f}" ]]; then
        continue
      fi

      # Find the most recent matching file anywhere under repo (common outputs/ locations)
      src="$(find "$REPO_ROOT" -name "$f" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')"
      if [[ -n "$src" && -f "$src" ]]; then
        echo "  -> copy $(realpath --relative-to="$REPO_ROOT" "$src")  => paper/$f"
        cp -f "$src" "${SCRIPT_DIR}/${f}"
      else
        echo "  !! missing: $f (not found anywhere under repo yet)"
      fi
    done
  fi
}

build_fig5() {
  local tex="${SCRIPT_DIR}/${FIG5_TEX}"
  [[ -f "$tex" ]] || die "Missing fig5 tex: $tex"

  echo "[paper] Building fig5 from $(basename "$tex")"
  ( cd "$SCRIPT_DIR" && \
    "$LATEXMK" -pdf -interaction=nonstopmode -halt-on-error "$(basename "$tex")" )
}

export_numbers() {
  local exp="${SCRIPTS_DIR}/export_paper_numbers.py"
  [[ -f "$exp" ]] || die "Missing exporter: $exp"

  echo "[paper] Exporting paper numbers / table tex"
  ( cd "$REPO_ROOT" && "$PYTHON" "$exp" )
}

build_pdf() {
  local main="${SCRIPT_DIR}/${TEX_MAIN}"
  [[ -f "$main" ]] || die "Missing main TeX: $main"

  echo "[paper] Building $(basename "$main")"
  ( cd "$SCRIPT_DIR" && \
    "$LATEXMK" -pdf -bibtex -interaction=nonstopmode -halt-on-error "$(basename "$main")" )
}

clean() {
  echo "[paper] Cleaning LaTeX artifacts (keeping PDFs)"
  ( cd "$SCRIPT_DIR" && \
    "$LATEXMK" -c "$TEX_MAIN" >/dev/null 2>&1 || true ; \
    "$LATEXMK" -c "$FIG5_TEX" >/dev/null 2>&1 || true )

  rm -f "$SCRIPT_DIR"/*.aux "$SCRIPT_DIR"/*.bbl "$SCRIPT_DIR"/*.blg \
        "$SCRIPT_DIR"/*.fdb_latexmk "$SCRIPT_DIR"/*.fls "$SCRIPT_DIR"/*.log \
        "$SCRIPT_DIR"/*.out "$SCRIPT_DIR"/*.toc "$SCRIPT_DIR"/*.lof "$SCRIPT_DIR"/*.lot \
        "$SCRIPT_DIR"/*.synctex.gz || true
}

distclean() {
  clean
  echo "[paper] Removing generated PDFs in paper/"
  rm -f "$SCRIPT_DIR"/fig*.pdf || true
  rm -f "$SCRIPT_DIR"/"$(basename "${TEX_MAIN%.tex}.pdf")" || true
}

cmd="${1:-help}"
case "$cmd" in
  all)
    have "$PYTHON"  || die "PYTHON not found: $PYTHON"
    have "$LATEXMK" || die "latexmk not found: $LATEXMK"
    run_fig_generators
    build_fig5
    export_numbers
    build_pdf
    ;;
  pdf)      build_pdf ;;
  figs)     run_fig_generators ;;
  fig5)     build_fig5 ;;
  numbers)  export_numbers ;;
  clean)    clean ;;
  distclean) distclean ;;
  dist)
    # Where are we?
    PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$PAPER_DIR/.." && pwd)"
    DIST_DIR="$REPO_ROOT/dist"
    OUTP_DIR="$REPO_ROOT/outputs/paper"

    TEX_MAIN="${TEX_MAIN:-dft_hom_phase_geometry_v1.tex}"
    BASENAME="${TEX_MAIN%.tex}"

    mkdir -p "$DIST_DIR"

    echo "[dist] Building full paper first..."
    "$0" all

    echo "[dist] Copying artifacts to $DIST_DIR ..."
    # Core paper
    cp -f "$PAPER_DIR/${BASENAME}.pdf" "$DIST_DIR/"
    cp -f "$PAPER_DIR/${BASENAME}.tex" "$DIST_DIR/"
    cp -f "$PAPER_DIR/references.bib"  "$DIST_DIR/"

    # Figures (generated in paper/)
    cp -f "$PAPER_DIR"/fig{1,2,3,4,5}_*.pdf "$DIST_DIR/" 2>/dev/null || true

    # If you want the standalone fig5 build too:
    cp -f "$PAPER_DIR/make_fig5.pdf" "$DIST_DIR/" 2>/dev/null || true

    # Paper-number single source of truth exports
    if [ -f "$OUTP_DIR/paper_numbers.json" ]; then
      cp -f "$OUTP_DIR/paper_numbers.json" "$DIST_DIR/"
    fi
    if [ -f "$OUTP_DIR/paper_numbers_full.json" ]; then
      cp -f "$OUTP_DIR/paper_numbers_full.json" "$DIST_DIR/"
    fi
    if [ -f "$OUTP_DIR/paper_table_diffusion.tex" ]; then
      cp -f "$OUTP_DIR/paper_table_diffusion.tex" "$DIST_DIR/"
    fi

    # Optional: include a build stamp for traceability
    {
      echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "tex_main=$TEX_MAIN"
      (cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/') || true
      (cd "$REPO_ROOT" && git status --porcelain 2>/dev/null | wc -l | awk '{print "git_dirty_files="$1}') || true
    } > "$DIST_DIR/build_info.txt"

    echo "[dist] Done. Contents:"
    ls -lh "$DIST_DIR"
    ;;
  help|-h|--help) usage ;;
  *)
    die "Unknown command: $cmd"
    ;;
esac