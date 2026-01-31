.PHONY: help clean setup install run run-all list results

# Colors for output
GREEN := \033[92m
YELLOW := \033[93m
RED := \033[91m
BLUE := \033[94m
RESET := \033[0m

help:
	@echo "$(BLUE)Upsonic Benchmarks - Available Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(RESET)"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo "  make install      - Install/update dependencies in existing venv"
	@echo ""
	@echo "$(GREEN)Benchmark Commands:$(RESET)"
	@echo "  make run          - Run default benchmark (Simple Text Query)"
	@echo "  make run-all      - Run all test cases"
	@echo "  make run-compare  - Compare OpenAI vs Anthropic models"
	@echo "  make list         - List available test cases"
	@echo ""
	@echo "$(GREEN)Custom Runs:$(RESET)"
	@echo "  make run-math     - Run Math Problem test case"
	@echo "  make run-structured - Run Simple Structured Output test case"
	@echo "  make run-analysis - Run Text Analysis test case"
	@echo "  make run-iterations N=10 - Run with custom iterations"
	@echo ""
	@echo "$(GREEN)Maintenance Commands:$(RESET)"
	@echo "  make clean        - Clean up cache and temporary files"
	@echo "  make clean-all    - Clean everything including venv and results"
	@echo "  make results      - Show recent benchmark results"
	@echo ""

clean:
	@echo "$(YELLOW)Cleaning up cache and temporary files...$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned up cache files$(RESET)"

clean-all: clean
	@echo "$(YELLOW)Cleaning virtual environment and results...$(RESET)"
	@if [ -d ".venv" ]; then \
		rm -rf .venv && \
		echo "$(GREEN)✓ Removed virtual environment$(RESET)"; \
	else \
		echo "$(YELLOW)No virtual environment to remove$(RESET)"; \
	fi
	@if [ -d "overhead_analysis/results" ]; then \
		find overhead_analysis/results -name "*.json" -delete 2>/dev/null || true && \
		echo "$(GREEN)✓ Cleaned benchmark results$(RESET)"; \
	fi

setup:
	@echo "$(BLUE)Setting up benchmark environment...$(RESET)"
	@if [ ! -d ".venv" ]; then \
		echo "$(YELLOW)Creating virtual environment...$(RESET)" && \
		uv venv && \
		echo "$(GREEN)✓ Virtual environment created$(RESET)"; \
	else \
		echo "$(YELLOW)Virtual environment already exists$(RESET)"; \
	fi
	@echo "$(YELLOW)Installing dependencies...$(RESET)"
	@. .venv/bin/activate && \
		uv pip install pydantic python-dotenv pympler && \
		uv pip install -e .. && \
		echo "$(GREEN)✓ Dependencies installed$(RESET)"
	@echo ""
	@echo "$(BLUE)Setup complete!$(RESET)"
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  1. Create .env file in parent directory with: OPENAI_API_KEY=your-key"
	@echo "  2. Run: make run"

install:
	@if [ ! -d ".venv" ]; then \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Installing/updating dependencies...$(RESET)"
	@. .venv/bin/activate && \
		uv pip install --upgrade pydantic python-dotenv pympler && \
		uv pip install -e .. && \
		echo "$(GREEN)✓ Dependencies updated$(RESET)"

run:
	@if [ ! -d ".venv" ]; then \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if [ ! -f "../.env" ]; then \
		echo "$(RED)✗ .env file not found in parent directory$(RESET)"; \
		echo "$(YELLOW)Create .env with: OPENAI_API_KEY=your-key$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running benchmark (Simple Text Query)...$(RESET)"
	@echo ""
	@. .venv/bin/activate && python -m overhead_analysis.benchmark

run-all:
	@if [ ! -d ".venv" ]; then \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if [ ! -f "../.env" ]; then \
		echo "$(RED)✗ .env file not found in parent directory$(RESET)"; \
		echo "$(YELLOW)Create .env with: OPENAI_API_KEY=your-key$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running all benchmark test cases...$(RESET)"
	@echo "$(YELLOW)This will take several minutes and consume API credits$(RESET)"
	@echo ""
	@. .venv/bin/activate && python -m overhead_analysis.benchmark --all-tests

list:
	@if [ ! -d ".venv" ]; then \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Available Test Cases:$(RESET)"
	@echo ""
	@. .venv/bin/activate && python -m overhead_analysis.benchmark --list-tests

results:
	@echo "$(BLUE)Recent Benchmark Results:$(RESET)"
	@echo ""
	@if [ -d "overhead_analysis/results" ]; then \
		ls -lht overhead_analysis/results/*.json 2>/dev/null | head -10 || \
		echo "$(YELLOW)No results found yet. Run 'make run' first.$(RESET)"; \
	else \
		echo "$(YELLOW)Results directory not found$(RESET)"; \
	fi

# Custom benchmark runs
run-math:
	@. .venv/bin/activate && python -m overhead_analysis.benchmark --test-case "Math Problem"

run-structured:
	@. .venv/bin/activate && python -m overhead_analysis.benchmark --test-case "Simple Structured Output"

run-analysis:
	@. .venv/bin/activate && python -m overhead_analysis.benchmark --test-case "Text Analysis"

run-iterations:
	@if [ -z "$(N)" ]; then \
		echo "$(RED)✗ Please specify iterations: make run-iterations N=10$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running benchmark with $(N) iterations...$(RESET)"
	@. .venv/bin/activate && python -m overhead_analysis.benchmark --iterations $(N)

run-compare:
	@if [ ! -d ".venv" ]; then \
		echo "$(RED)✗ Virtual environment not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if [ ! -f "../.env" ]; then \
		echo "$(RED)✗ .env file not found in parent directory$(RESET)"; \
		echo "$(YELLOW)Create .env with: OPENAI_API_KEY=xxx and ANTHROPIC_API_KEY=xxx$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Running benchmark with OpenAI + Anthropic models...$(RESET)"
	@echo "$(YELLOW)This will test both providers and compare results$(RESET)"
	@echo ""
	@. .venv/bin/activate && python -m overhead_analysis.benchmark --model "openai/gpt-4o-mini,anthropic/claude-3-5-haiku-20241022"

# Development helpers
dev:
	@echo "$(BLUE)Activating virtual environment...$(RESET)"
	@echo "$(YELLOW)Run: source .venv/bin/activate$(RESET)"

test-env:
	@echo "$(BLUE)Testing environment setup...$(RESET)"
	@if [ -d ".venv" ]; then \
		echo "$(GREEN)✓ Virtual environment exists$(RESET)"; \
	else \
		echo "$(RED)✗ Virtual environment not found$(RESET)"; \
	fi
	@if [ -f "../.env" ]; then \
		echo "$(GREEN)✓ .env file exists$(RESET)"; \
	else \
		echo "$(RED)✗ .env file not found$(RESET)"; \
	fi
	@if [ -d ".venv" ]; then \
		. .venv/bin/activate && \
		if python -c "import upsonic" 2>/dev/null; then \
			echo "$(GREEN)✓ Upsonic installed$(RESET)"; \
		else \
			echo "$(RED)✗ Upsonic not installed$(RESET)"; \
		fi; \
	fi

