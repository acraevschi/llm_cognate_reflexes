# --- Configuration -----------------------------------------------------------
BASE_DIR      := ./data
DATA_DIR      := $(BASE_DIR)/lexibank
GLOTTOLOG_DIR := $(BASE_DIR)/glottolog
PYTHON        := python
PYTEST        := pytest

# --- Targets -----------------------------------------------------------------
.PHONY: env download-lexibank download-glottolog download-all \
        generate-reflexes generate-reconstruction test stats help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

env:  ## Update conda environment from environment.yml
	conda env update -f environment.yml --prune

download-lexibank:  ## Download Lexibank datasets
	bash scripts/download_data.sh $(BASE_DIR)

download-glottolog:  ## Clone the Glottolog repository
	@if [ ! -d "$(GLOTTOLOG_DIR)" ]; then \
		git clone https://github.com/glottolog/glottolog.git $(GLOTTOLOG_DIR); \
	else \
		echo "Glottolog already present at $(GLOTTOLOG_DIR); pulling latest…"; \
		git -C $(GLOTTOLOG_DIR) pull; \
	fi

download-all: download-lexibank download-glottolog  ## Download Lexibank + Glottolog

generate-reflexes:  ## Generate cognate-reflex triplets
	$(PYTHON) scripts/generate_triplets.py \
		--task cognate_reflex \
		--data-dir $(DATA_DIR) \
		--glottolog-dir $(GLOTTOLOG_DIR)

generate-reconstruction:  ## Generate reconstruction triplets
	$(PYTHON) scripts/generate_triplets.py \
		--task reconstruction \
		--data-dir $(DATA_DIR) \
		--glottolog-dir $(GLOTTOLOG_DIR)

test:  ## Run the test suite
	$(PYTEST) tests/

stats:  ## Print dataset statistics without generating triplets
	$(PYTHON) scripts/generate_triplets.py \
		--stats-only \
		--data-dir $(DATA_DIR) \
		--glottolog-dir $(GLOTTOLOG_DIR)
