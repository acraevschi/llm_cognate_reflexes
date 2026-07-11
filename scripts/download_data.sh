#!/usr/bin/env bash
# Download Lexibank datasets and Glottolog for cognate_reflexes.
#
# Usage:
#   ./scripts/download_data.sh [DATA_DIR]
#
# DATA_DIR defaults to ./data.  Lexibank repos go under DATA_DIR/lexibank/
# and Glottolog under DATA_DIR/glottolog/.
set -euo pipefail

DATA_DIR="${1:-./data}"
LEXIBANK_DIR="${DATA_DIR}/lexibank"
GLOTTOLOG_DIR="${DATA_DIR}/glottolog"

# Colours for output (if terminal supports them).
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'  # No colour

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ======================================================================
# Glottolog
# ======================================================================

mkdir -p "${DATA_DIR}"

if [ -d "${GLOTTOLOG_DIR}" ]; then
    log_info "Glottolog already cloned at ${GLOTTOLOG_DIR} — skipping."
else
    log_info "Cloning Glottolog into ${GLOTTOLOG_DIR}…"
    git clone --depth 1 https://github.com/glottolog/glottolog.git "${GLOTTOLOG_DIR}"
    log_info "Glottolog cloned."
fi

# ======================================================================
# Lexibank
# ======================================================================

mkdir -p "${LEXIBANK_DIR}"

CLONED=0
SKIPPED=0
FAILED=0
PAGE=1
PER_PAGE=100

log_info "Fetching Lexibank repository list from GitHub…"

while true; do
    # Fetch a page of repos.
    RESPONSE=$(curl -sS \
        -H "Accept: application/vnd.github+json" \
        "https://api.github.com/orgs/lexibank/repos?per_page=${PER_PAGE}&page=${PAGE}&type=public")

    # Check for empty page (end of pagination).
    NUM_REPOS=$(echo "${RESPONSE}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if isinstance(data, list):
    print(len(data))
else:
    print(0)
")

    if [ "${NUM_REPOS}" -eq 0 ]; then
        break
    fi

    # Extract repo names and clone URLs.
    echo "${RESPONSE}" | python3 -c "
import sys, json
repos = json.load(sys.stdin)
for repo in repos:
    print(repo['name'], repo['clone_url'])
" | while read -r REPO_NAME CLONE_URL; do
        if [ "${REPO_NAME}" = "glottolog" ]; then
            log_info "Skipping redundant ${REPO_NAME} repo."
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        TARGET="${LEXIBANK_DIR}/${REPO_NAME}"
        if [ -d "${TARGET}" ]; then
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        log_info "Cloning ${REPO_NAME}…"
        if git clone --depth 1 "${CLONE_URL}" "${TARGET}" 2>/dev/null; then
            CLONED=$((CLONED + 1))
        else
            log_warn "Failed to clone ${REPO_NAME}"
            FAILED=$((FAILED + 1))
        fi
    done

    PAGE=$((PAGE + 1))
done

log_info "Done."
log_info "Summary: cloned=${CLONED}, skipped (already present)=${SKIPPED}, failed=${FAILED}"
