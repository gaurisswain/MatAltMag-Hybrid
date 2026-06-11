#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib}"
"${PYTHON:-python3}" - <<'PY'
from pathlib import Path
from mataltmag_hybrid.analysis.ablations import write_candidate_error_analysis, write_surrogate_gnn_explanation
from mataltmag_hybrid.analysis.embedding_plots import write_embedding_pca_plots
from mataltmag_hybrid.analysis.report_tables import write_reproducibility_checklist
from mataltmag_hybrid.analysis.shap_analysis import write_hybrid_feature_importance

write_reproducibility_checklist(Path("report/reproducibility_checklist.md"))
write_embedding_pca_plots()
write_surrogate_gnn_explanation()
write_candidate_error_analysis()
write_hybrid_feature_importance()
PY
