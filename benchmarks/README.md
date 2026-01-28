# Upsonic Framework Benchmarks

This directory contains benchmark projects for measuring the performance of various Upsonic Framework components.

## 🚀 Setup

### ⚡ Quick Start (with Makefile - Recommended)

```bash
cd benchmarks
make setup              # Virtual env + dependencies
cd .. && echo "OPENAI_API_KEY=sk-xxx" > .env && cd benchmarks
make run                # Run first benchmark
```

**For all Makefile commands:** `make help`  
**Detailed quick start:** [QUICKSTART.md](QUICKSTART.md)

### 📋 Manual Setup

```bash
cd benchmarks
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install pydantic python-dotenv pympler
uv pip install -e ..

# Create environment file
cd .. && echo "OPENAI_API_KEY=sk-xxx" > .env && cd benchmarks

# Run benchmark
python -m overhead_analysis.benchmark
```

---

## 📁 Available Benchmark Projects

### 1. Overhead Analysis (`overhead_analysis/`)

Three-way comparison: Direct LLM Call (minimal overhead) vs Agent (no prompt) vs Agent (with prompt).

**Measured Metrics:**
- Memory usage (object size in bytes)
- Execution speed (mean, median, stdev, min, max)
- Cost metrics (per iteration and total, including tokens)
- Sample outputs (actual responses from each approach)

**Run:**
```bash
python -m benchmarks.overhead_analysis.benchmark
python -m benchmarks.overhead_analysis.benchmark --all-tests
```

**Details:** [overhead_analysis/README.md](overhead_analysis/README.md)

---

## 🛠️ Common Utilities

### `utils.py` - Shared Profiling Tools

Common tools used by all benchmark projects:

- **MemoryProfiler**: Object size measurement, peak memory tracking
- **PerformanceProfiler**: Timing measurement, statistical analysis
- **BenchmarkReporter**: Report generation, JSON export, console output

**Usage:**
```python
from benchmarks.utils import MemoryProfiler, PerformanceProfiler

# Memory profiling
profiler = MemoryProfiler()
profiler.start_tracking()
# ... your code ...
current_mb, peak_mb = profiler.stop_tracking()

# Performance profiling
result, elapsed_ms = PerformanceProfiler.measure_time(operation)

# Multiple runs with statistics
metrics = PerformanceProfiler.measure_multiple_runs(operation, iterations=10)
```

---

## 🚀 Creating a New Benchmark Project

### 1. Directory Structure

```
benchmarks/
├── your_benchmark_name/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── test_cases.py (optional)
│   ├── README.md
│   └── results/
│       └── .gitkeep
```

### 2. Example Template

```python
# your_benchmark_name/benchmark.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.utils import (
    MemoryProfiler,
    PerformanceProfiler,
    BenchmarkReporter
)

def run_benchmark():
    # Your benchmark logic here
    pass

if __name__ == "__main__":
    run_benchmark()
```

### 3. Best Practices

✅ **Use common utilities**: Use tools from `benchmarks.utils` module  
✅ **Save results**: Save in JSON format to `results/` directory  
✅ **Add README**: Explain how to run and what is measured  
✅ **Define test cases**: Create repeatable tests  
✅ **Statistical analysis**: Multiple runs, mean, median, stdev  

---

## 📊 Environment Setup

A `.env` file is required for all benchmarks:

```bash
# In main directory (Upsonic/.env)
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here  # Optional
```

---

## 🎯 Future Benchmark Projects (Suggestions)

### Model Comparison (`model_comparison/`)
Performance comparison of different LLM providers:
- OpenAI vs Anthropic vs Google
- Speed, cost, quality metrics

### Tool Performance (`tool_performance/`)
Tool execution overhead analysis:
- Native functions vs MCP tools
- Tool orchestration performance

### Memory Backend (`memory_backend/`)
Performance comparison of different memory backends:
- SQLite vs PostgreSQL vs Redis
- Read/write speed, storage overhead

### Batch Processing (`batch_processing/`)
Parallel vs sequential execution analysis:
- Throughput metrics
- Resource utilization

### Streaming Performance (`streaming_performance/`)
Stream vs non-stream execution comparison:
- First token latency
- Total completion time

---

## 📈 Standard Metrics

Standard metrics used across all benchmark projects:

### Memory Metrics
- `shallow_size_bytes`: Size of the object itself
- `deep_size_bytes`: Total size including all references
- `peak_memory_mb`: Maximum memory usage
- `current_memory_mb`: Current memory usage

### Performance Metrics
- `init_time_ms`: Initialization time
- `execution_time_ms`: Execution time
- `total_time_ms`: Total time
- `mean_time_ms`: Average (multiple runs)
- `median_time_ms`: Median
- `stdev_time_ms`: Standard deviation
- `min_time_ms` / `max_time_ms`: Min/Max

### Cost Metrics
- `total_cost`: Total API cost in USD
- `total_input_token`: Total input tokens
- `total_output_token`: Total output tokens
- `cost_per_1k_tokens`: Cost per 1,000 tokens

---

## 🤝 Contributing

For new benchmark projects or improvements:

1. Create a new subdirectory
2. Use common utilities (`benchmarks.utils`)
3. Add README and documentation
4. Submit a pull request

---

## 📝 Notes

- **First run**: May be slower due to model loading
- **Network connection**: Required for LLM API calls
- **API costs**: Consider test iterations
- **Statistical significance**: At least 5 iterations recommended

---

## 📚 Resources

- [Upsonic Documentation](https://docs.upsonic.ai)
- [Direct LLM Call Guide](https://docs.upsonic.ai/concepts/direct-llm-call/overview)
- [Agent Guide](https://docs.upsonic.ai/concepts/agent)
