.PHONY: help clean check autoformat setup
.DEFAULT: help

# Generates a useful overview/help message for various make features - add to this as necessary!
help:
	@echo "make setup"
	@echo "    Download required llama2 config/tokenizer files from HuggingFace"
	@echo "make clean"
	@echo "    Remove all temporary pyc/pycache files"
	@echo "make check"
	@echo "    Run code style and linting (black, ruff) *without* changing files!"
	@echo "make autoformat"
	@echo "    Run code styling (black, ruff) and update in place - committing with pre-commit also does this."

# === Model Config Download ===
LLAMA2_DIR := prismatic/models/backbones/llm/llama2_config
LLAMA2_REPO := NousResearch/Llama-2-7b-hf
LLAMA2_FILES := config.json special_tokens_map.json tokenizer_config.json tokenizer.json tokenizer.model

setup:
	@mkdir -p $(LLAMA2_DIR)
	@for f in $(LLAMA2_FILES); do \
		if [ ! -f "$(LLAMA2_DIR)/$$f" ]; then \
			echo "Downloading $$f from $(LLAMA2_REPO)..."; \
			huggingface-cli download $(LLAMA2_REPO) $$f --local-dir $(LLAMA2_DIR); \
		else \
			echo "$$f already exists, skipping."; \
		fi; \
	done
	@echo "Done! All llama2 config files are ready."

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

check:
	black --check .
	ruff check --show-source .

autoformat:
	black .
	ruff check --fix --show-fixes .
