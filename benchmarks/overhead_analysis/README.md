# Overhead Analysis: Direct vs Agent

This benchmark measures the performance differences between Upsonic Framework's **Direct LLM Call** (minimal overhead) and **Agent** (full-featured) approaches.

## 🚀 Setup

### 1. Virtual Environment (in benchmarks directory)

```bash
# Navigate to benchmarks/ directory
cd ../..  # If you're in this directory
cd benchmarks  # If you're in main directory

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
# Core packages
uv pip install pydantic python-dotenv pympler anthropic

# Upsonic framework (editable mode)
uv pip install -e ..
```

### 3. Environment File

Create a `.env` file in the main directory (Upsonic/):

```bash
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here  # Optional
```

### 4. Run Benchmark

```bash
# In benchmarks/ directory
python -m overhead_analysis.benchmark
```

---

## 🎯 Purpose

Direct LLM Call is designed to provide maximum speed with minimum overhead. This benchmark proves how much lighter and faster Direct is compared to Agent.

### Measured Metrics

- **Memory Usage**: Object size, peak memory usage
- **Execution Speed**: Initialization time, task completion time
- **Cost Metrics**: API costs, tokens, cost per 1K tokens
- **Comparison**: Direct vs Agent ratios

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file in the main directory:

```bash
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here  # Optional
```

### 2. Simple Benchmark

```bash
python -m benchmarks.overhead_analysis.benchmark
```

### 3. Specific Test Case

```bash
python -m benchmarks.overhead_analysis.benchmark --test-case "Math Problem"
```

### 4. All Test Cases

```bash
python -m benchmarks.overhead_analysis.benchmark --all-tests
```

### 5. Multiple Models (Comma-separated)

```bash
python -m benchmarks.overhead_analysis.benchmark --model "openai/gpt-4o-mini,anthropic/claude-3-5-haiku-20241022"
```

### 6. Different Model

```bash
python -m benchmarks.overhead_analysis.benchmark --model "openai/gpt-4o"
```

### 7. More Iterations

```bash
python -m benchmarks.overhead_analysis.benchmark --iterations 10 --all-tests
```

## 📊 Test Scenarios

1. **Simple Text Query** - Simple text prompt (minimum complexity)
2. **Simple Structured Output** - Structured output with Pydantic model
3. **Math Problem** - Mathematical reasoning
4. **Text Analysis** - Complex text analysis
5. **Context-based Query** - Query with context information

## 📈 Example Results

### Console Output
```
================================================================================
BENCHMARK RESULTS: Simple Text Query (gpt-4o-mini)
================================================================================
Timestamp: 2026-01-28T15:29:15.562264
Python: 3.12.8

TASK:
  Description: What is 2 + 2?
  Response Format: str

DIRECT:
  Memory:
    Object Size: 38,432 bytes
  Performance:
    Mean Time: 1257.84 ms
    Median Time: 1277.25 ms
    Std Dev: 88.24 ms
  Cost:
    Total Cost: $0.000028
    Total Tokens: 92

AGENT (NO PROMPT):
  Memory:
    Object Size: 47,320 bytes
  Performance:
    Mean Time: 1237.53 ms
    Median Time: 1229.80 ms
    Std Dev: 35.74 ms
  Cost:
    Total Cost: $0.000028
    Total Tokens: 92

AGENT (WITH PROMPT):
  Memory:
    Object Size: 46,696 bytes
  Performance:
    Mean Time: 1205.27 ms
    Median Time: 1146.21 ms
    Std Dev: 106.43 ms
  Cost:
    Total Cost: $0.000056
    Total Tokens: 284
================================================================================
```

### Markdown Report Format
The generated Markdown reports include:

**Task Information Section:**
- Task description and parameters
- Response format
- Context information (if any)

**Detailed Comparison Table:**
- Speed metrics (Mean, Stdev, Median, Min, Max Time)
- Memory usage (bytes)
- Cost metrics (Mean cost per iteration, Total cost)
- Token usage (Mean and Total for Input/Output/Total tokens)
- Side-by-side comparison of Direct, Agent (No Prompt), and Agent (With Prompt)

**Sample Outputs:**
- Actual responses from each approach
- Shows how Direct, Agent (no prompt), and Agent (with prompt) answered the task

## 📁 Results

Benchmark results are saved in the `results/` directory in both JSON and Markdown formats:

```
results/
├── Simple_Text_Query_(gpt-4o-mini)_20260127_103045.json
├── Simple_Text_Query_(gpt-4o-mini)_20260127_103045.md
├── Math_Problem_(gpt-4o-mini)_20260127_103145.json
├── Math_Problem_(gpt-4o-mini)_20260127_103145.md
└── ...
```

### Markdown Reports Include:

- **Task Information**: Full description, response format, context
- **Detailed Comparison Table**: 
  - Speed metrics (Mean, Stdev, Median, Min, Max)
  - Memory usage (bytes)
  - Cost metrics (per iteration and total)
  - Token usage (mean and total)
  - Three-way comparison: Direct vs Agent (no prompt) vs Agent (with prompt)
- **Sample Outputs**: Actual responses from each approach showing quality differences

## 🔧 Command Line Options

```bash
--test-case TEXT         Run specific test case
--model TEXT            Model identifier or comma-separated list
                        Examples:
                          --model "openai/gpt-4o-mini"
                          --model "openai/gpt-4o"
                          --model "anthropic/claude-3-5-sonnet-20241022"
                          --model "openai/gpt-4o-mini,anthropic/claude-3-5-haiku-20241022"
--iterations INT        Number of iterations (default: 5)
--all-tests             Run all test cases
--list-tests            List available test cases
```

## 🤖 Supported Models

The benchmark supports any model available through Upsonic's providers:

### OpenAI Models
- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- `openai/gpt-4-turbo`
- `openai/gpt-3.5-turbo`

### Anthropic Models
- `anthropic/claude-3-5-sonnet-20241022`
- `anthropic/claude-3-5-haiku-20241022`
- `anthropic/claude-3-opus-20240229`

### Model Validation
The benchmark automatically validates model names before running. If an invalid model is provided, you'll see:
```bash
❌ Invalid model 'invalid/model': Unknown provider: invalid

💡 Examples of valid model formats:
   openai/gpt-4o-mini
   anthropic/claude-3-5-haiku-20241022
```

**Note:** Make sure you have the required API keys in your `.env` file for the providers you want to use.

## 📝 Adding New Test Cases

Add a new method to `test_cases.py`:

```python
@staticmethod
def get_my_custom_test() -> Dict[str, Any]:
    return {
        "name": "My Custom Test",
        "description": "Your prompt here",
        "response_format": str,
        "attachments": None,
        "context": None
    }
```

## 🎓 Key Learnings

What you can learn from this benchmark:

- **When to use Direct**: Simple query/response scenarios, document processing, speed-critical tasks
- **When to use Agent**: Tool orchestration, memory management, complex workflows
- **Performance trade-offs**: Speed vs features balance
- **Cost implications**: Token usage differences between approaches

## 🤝 Contributing

Submit pull requests for new test cases or improvements!

## 📚 Resources

- [Upsonic Documentation](https://docs.upsonic.ai)
- [Direct LLM Call Guide](https://docs.upsonic.ai/concepts/direct-llm-call/overview)
- [Agent Guide](https://docs.upsonic.ai/concepts/agent)
