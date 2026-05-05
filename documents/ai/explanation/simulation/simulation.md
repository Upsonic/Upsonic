---
name: llm-time-series-simulation
description: Use when working with Upsonic's LLM-powered time-series forecasting framework under src/upsonic/simulation/. Use when a user asks to build, customize, or run scenario-based forecasts, define a custom BaseSimulationObject, configure step-by-step LLM predictions, generate summary/detailed/visual/statistical reports, or export simulation results to JSON/CSV/PDF/HTML. Trigger when the user mentions Simulation, BaseSimulationObject, SimulationResult, SimulationConfig, SimulationStepOutput, TimeStep, TimeStepManager, MerchantRevenueForecastSimulation, StockPriceForecastSimulation, UserGrowthSimulation, scenario forecasting, time-step simulation, MRR forecast, stock price forecast, user growth forecast, metrics_to_track, build_step_prompt, get_step_output_schema, ReportsCollection, SummaryReport, DetailedReport, VisualReport, StatisticalReport, Chart.js reports, or reportlab PDF export.
---

# `src/upsonic/simulation/` — LLM-Powered Time-Series Simulation Framework

## 1. What this folder is

`src/upsonic/simulation/` is Upsonic's **LLM-powered, agent-driven time-series simulation engine**. It is *not* a Monte-Carlo numerical simulator, *not* an agent–environment reinforcement-learning sandbox, and *not* a multi-agent dialogue replay. Instead, it treats a Large Language Model as a **stochastic forecaster** that, given a structured business/financial scenario and the previous step's state, predicts the next step's metrics.

In other words:

> The framework iterates through `N` discrete time steps. At each step it builds a structured prompt describing the scenario plus the prior state, calls a Pydantic-typed `Direct` agent task, parses the structured response, extracts a configurable set of metrics, and appends them to a state series. The completed series is wrapped in a `SimulationResult` that exposes four chainable report types (summary / detailed / visual / statistical), each exportable to JSON, CSV, PDF, HTML, or rendered live in Jupyter via matplotlib / Chart.js.

Concretely the framework provides:

| Concern                          | Implementation                                                                                       |
| -------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Orchestration loop               | `Simulation` (`simulation.py`)                                                                       |
| Scenario contract                | `BaseSimulationObject` ABC + `SimulationConfig` + `SimulationStepOutput` (`base.py`)                 |
| Time granularity / calendar math | `TimeStep` enum + `TimeStepManager` (`time_step.py`)                                                 |
| Per-step record + result + reports | `SimulationStepRecord`, `SimulationResult`, `BaseReport` + 4 subclasses, `ReportsCollection` (`result.py`) |
| Pre-built scenarios              | `MerchantRevenueForecastSimulation`, `StockPriceForecastSimulation`, `UserGrowthSimulation` (`scenarios/`) |

The example given at the top of `__init__.py` is a 100-day daily forecast of monthly recurring revenue:

```python
from upsonic.simulation import Simulation
from upsonic.simulation.scenarios import MerchantRevenueForecastSimulation

simulation = Simulation(
    MerchantRevenueForecastSimulation(
        merchant_name="TechCo",
        shareholders=["Alice", "Bob"],
        sector="E-commerce",
        location="San Francisco",
        current_monthly_revenue_usd=50000,
    ),
    model="openai/gpt-4o",
    time_step="daily",
    simulation_duration=100,
    metrics_to_track=["monthly recurring revenue"],
)

result = simulation.run()
result.report("summary").to_pdf("summary.pdf")
result.report("visual").show()
```

So this is a **scenario-forecasting framework**: the LLM acts as the "physics engine" of a business/asset/user-base, and the framework is the time-loop and bookkeeping layer.

## 2. Folder layout

```
src/upsonic/simulation/
├── __init__.py            # Lazy-loading public surface (Simulation, BaseSimulationObject, ...)
├── base.py                # ABC + Pydantic base models + dataclass config
├── simulation.py          # Simulation orchestrator (sync + async run, retry, progress)
├── time_step.py           # TimeStep enum + TimeStepManager (timestamp/calendar math)
├── result.py              # Step record, SimulationResult, 4 reports, ReportsCollection
└── scenarios/
    ├── __init__.py        # Lazy exports of pre-built scenarios
    ├── merchant_revenue.py# MerchantRevenueForecastSimulation + step output schema
    ├── stock_price.py     # StockPriceForecastSimulation + step output schema
    └── user_growth.py     # UserGrowthSimulation + step output schema
```

Total: 9 Python modules, all read end-to-end for this document.

## 3. Top-level files

### 3.1 `__init__.py` — public, lazy-loaded surface

A defensive lazy-import shim. Heavy modules (`simulation.py` indirectly imports `Direct` and the entire agent stack) are deferred until first attribute access via `__getattr__`.

| Public name             | Source                                  |
| ----------------------- | --------------------------------------- |
| `Simulation`            | `upsonic.simulation.simulation`         |
| `BaseSimulationObject`  | `upsonic.simulation.base`               |
| `SimulationResult`      | `upsonic.simulation.result`             |
| `TimeStep`              | `upsonic.simulation.time_step`          |

`__all__` exposes exactly these four. Any other attribute raises `AttributeError`.

### 3.2 `base.py` — abstract scenario contract

Three Pydantic / ABC primitives:

| Symbol                      | Kind          | Role                                                                                                       |
| --------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| `SimulationState`           | `BaseModel`   | `step`, `timestamp`, `metrics: dict`. Currently unused at runtime — reserved for typed state extensions.   |
| `SimulationStepOutput`      | `BaseModel`   | Default LLM-output shape: `step`, `reasoning`, `confidence`, `metrics`. Scenarios subclass this.           |
| `BaseSimulationObject`      | `ABC`         | Defines the contract every scenario must fulfill.                                                          |
| `SimulationConfig`          | `@dataclass`  | Validated parameter bundle: model, time_step (string), duration, temperature, retry policy, progress flag. |

`BaseSimulationObject` requires four abstracts and offers three overridable hooks:

```text
abstract  name                       -> str
abstract  description                -> str
abstract  get_initial_state          -> Dict[str, Any]                # state at step 0
abstract  build_step_prompt(step, prev_state, metrics_to_track,
                            time_step_unit) -> str
abstract  get_step_output_schema     -> Type[BaseModel]               # response_format

overridable  extract_metrics(step_output, metrics_to_track) -> Dict
overridable  validate_metrics(metrics, step)                -> Dict
overridable  get_context_for_step(step)                     -> Optional[str]
helper       to_dict()                                      -> Dict   # serialization
```

`extract_metrics` does normalized fuzzy-matching on metric names (`"monthly recurring revenue"` → `monthly_recurring_revenue`), so users can list metrics in plain English.

`SimulationConfig.__post_init__` validates `time_step ∈ {hourly, daily, weekly, monthly, quarterly, yearly}`, `simulation_duration > 0`, `0.0 ≤ temperature ≤ 2.0`, `max_retries ≥ 0`.

### 3.3 `simulation.py` — orchestrator

Houses the `Simulation` class. Initialization params:

| Param                 | Default          | Notes                                              |
| --------------------- | ---------------- | -------------------------------------------------- |
| `simulation_object`   | required         | Any `BaseSimulationObject` subclass                |
| `model`               | `"openai/gpt-4o"`| Forwarded to `Direct(model=...)`                   |
| `time_step`           | `"daily"`        | Parsed via `TimeStep.from_string`                  |
| `simulation_duration` | `100`            | Number of LLM-driven steps after step 0            |
| `metrics_to_track`    | `[]`             | Names extracted from each step's structured output |
| `temperature`         | `0.7`            | Stored on `SimulationConfig` (validation only)     |
| `retry_on_error`      | `True`           | Per-step retry loop                                |
| `max_retries`         | `3`              | Exponential backoff: `0.5 * attempt` seconds       |
| `show_progress`       | `True`           | Rich-based progress lines                          |
| `start_date`          | `datetime.now()` | Anchor for `TimeStepManager`                       |

A unique `simulation_id = uuid4()` is assigned. The class exposes properties (`simulation_id`, `simulation_object`, `duration`, `time_step`, `metrics_to_track`, `is_running`) and three execution entry points:

| Method  | Behavior                                                                                                   |
| ------- | ---------------------------------------------------------------------------------------------------------- |
| `run()` | Sync wrapper: if a loop is already running, runs `_run_simulation_async` in a `ThreadPoolExecutor.submit(asyncio.run, ...)`; otherwise plain `asyncio.run`. |
| `arun()`| Pure async passthrough to `_run_simulation_async`.                                                         |
| `_execute_step(step, prev_state)` | Builds prompt, calls `Direct.do_async(Task(..., response_format=schema))`, retries with backoff, extracts and validates metrics, returns a `SimulationStepRecord`. |

The internal step-loop logic is:

```text
init Direct(model=...) and prepare model
state ← simulation_object.get_initial_state()
emit step-0 record with state as metrics
for step in 1..duration:
    prompt ← simulation_object.build_step_prompt(step, state, metrics_to_track, time_unit)
            + time_manager.get_time_context(step)['timestamp']
            + simulation_object.get_context_for_step(step) (optional)
            + "Please predict the following metrics: ..."
    schema ← simulation_object.get_step_output_schema()
    task   ← Task(description=prompt, response_format=schema)
    result ← await direct.do_async(task)        # with retry-on-error
    metrics← simulation_object.extract_metrics(result, metrics_to_track)
    metrics← simulation_object.validate_metrics(metrics, step)
    state.update(metrics)                       # carries forward
    append SimulationStepRecord
return SimulationResult(...)
```

Failure path: when `retry_on_error=True` and a step still fails after all retries, an `error_record` is appended with `success=False`, `error=str(e)`, and the loop continues with the *unchanged* state. When `retry_on_error=False`, the exception is re-raised.

Progress UI is fully Rich-based: a panel at start (`🔮 Simulation`), a one-line live counter per step, a green completion line.

### 3.4 `time_step.py` — calendar primitives

```python
class TimeStep(Enum):
    HOURLY    = "hourly"
    DAILY     = "daily"
    WEEKLY    = "weekly"
    MONTHLY   = "monthly"
    QUARTERLY = "quarterly"
    YEARLY    = "yearly"
```

Each variant exposes `.singular_unit` ("hour"), `.plural_unit` ("hours"), `.display_name` ("Hourly"), and `.get_timedelta(steps)`. Monthly = 30 days, quarterly = 91 days, yearly = 365 days — explicitly approximations.

`TimeStepManager(time_step, start_date)` resolves step → datetime and produces:

* `get_timestamp_for_step(step)` → `datetime`
* `format_timestamp(step)` → human string per granularity (e.g. `"Q3 2026"`, `"Week of 2026-04-28"`, `"2026-04-28 14:00"`)
* `get_step_description(step)` → `"Daily step 5 (2026-05-03)"`
* `get_time_context(step)` → dict with `step`, `timestamp`, `datetime`, `year`, `month`, `month_name`, `day`, `day_of_week`, `quarter`, plus boolean flags `is_weekend`, `is_month_start`, `is_month_end`, `is_year_start`, `is_year_end`. These rich flags are what get injected into prompts to give the LLM seasonality awareness.
* `generate_timeline(total_steps)` → bulk timeline for visualizations.

### 3.5 `result.py` — records, reports, exports

This is the biggest single module (≈1150 lines). Architecture:

```
SimulationStepRecord      ← @dataclass; per-step row
SimulationResult          ← top-level handle returned by Simulation.run()
ReportsCollection         ← lazy bag of 4 reports keyed by string
BaseReport                ← export plumbing (json/csv/pdf/html/show)
├── SummaryReport         ← totals, durations, metric deltas, % change
├── DetailedReport        ← step-by-step table with metric columns
├── VisualReport          ← Chart.js HTML + matplotlib `.show()` fallback
└── StatisticalReport     ← min/max/mean/median/stdev/variance/q1/q3/range/trend slope
```

Key design points:

* **Lazy report instantiation.** `SimulationResult.report(...)` and `SimulationResult.reports()` only build `ReportsCollection` on first call.
* **Chainable exports.** Every `to_json/csv/pdf/html` returns `self`, enabling `result.report("summary").to_pdf("a.pdf").to_csv("a.csv")`.
* **Format probing.**
  * `to_pdf` lazily imports `reportlab` and raises a friendly `ImportError` if missing.
  * `show()` tries `IPython.display`, falls back to `rich.console`. `VisualReport.show()` additionally tries `matplotlib`.
* **Trend detection** in `StatisticalReport` does a manual linear regression slope across steps and emits `trend_direction ∈ {up, down, flat}`.
* **Visualization** uses Chart.js loaded from `cdn.jsdelivr.net` and a hard-coded 6-color palette; multi-metric series share the same X axis (timestamps).
* `ReportsCollection.save_all(directory, format)` mass-exports every report in the chosen format to a directory.

`SimulationStepRecord.to_dict()` carefully detects `parsed_response.model_dump()` to handle Pydantic instances.

`SimulationResult.to_dict()` and `to_json()` support full reproducible serialization (raw prompts, raw responses, parsed responses, metrics, errors).

## 4. Subfolders

### 4.1 `scenarios/`

Pre-built `BaseSimulationObject` subclasses. Each defines:

1. A Pydantic step-output schema (`*StepOutput`) that the LLM is forced to populate.
2. A scenario class with rich `__init__` parameters, descriptive `name` / `description`, a typed `get_initial_state`, a templated `build_step_prompt` (using f-strings with markdown headings), `get_step_output_schema`, and overridden `extract_metrics` / `validate_metrics`.

| Scenario                              | Step-output schema fields                                                                                                                                                                                      | Key validation                                                       |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `MerchantRevenueForecastSimulation`   | `monthly_recurring_revenue`, `daily_revenue`, `customer_count`, `average_order_value`, `churn_rate`, `growth_rate`, `market_sentiment`, `key_factors`, `risks`                                                  | revenue ≥ 0, customers ≥ 0 int, churn ∈ [0,1], growth ∈ [-0.5, 1.0]  |
| `StockPriceForecastSimulation`        | `stock_price`, `price_change`, `percent_change`, `trading_volume`, `market_sentiment`, `volatility_index`, `support_level`, `resistance_level`, `key_catalysts`, `risks`                                       | price ≥ 0.01, volume ≥ 0 int                                         |
| `UserGrowthSimulation`                | `total_users`, `active_users`, `new_signups`, `churned_users`, `retention_rate`, `activation_rate`, `viral_coefficient`, `engagement_score`, `growth_rate`, `acquisition_channel`, `key_factors`               | counts ≥ 0 int, rates ∈ [0,1], engagement ∈ [0,100], active ≤ total  |

`MerchantRevenueForecastSimulation` is the only scenario that overrides `get_context_for_step` — it injects monthly-billing-cycle and start-of-week context every 30 / 7 steps, demonstrating the extension hook.

The prompts contain explicit "Guidelines" and "Patterns by Stage" hints (e.g. `Daily growth should typically be between -5% and +5%`) so that the LLM produces realistic — not pathologically optimistic — trajectories.

`scenarios/__init__.py` is again a lazy `__getattr__` shim, exposing only:

```
MerchantRevenueForecastSimulation
StockPriceForecastSimulation
UserGrowthSimulation
```

## 5. Cross-file relationships

```
                ┌────────────────────┐
                │   user code        │
                └─────────┬──────────┘
                          │ from upsonic.simulation import Simulation
                          │ from upsonic.simulation.scenarios import ...
                          ▼
        ┌──────────────────────────────────┐
        │       __init__.py (lazy)         │
        └────┬───────────────┬─────────────┘
             │               │
             ▼               ▼
   ┌────────────────┐  ┌──────────────────────┐
   │ simulation.py  │  │ scenarios/*          │
   │  Simulation    │  │  *Simulation classes │
   └────┬───────────┘  └──────────┬───────────┘
        │ uses                    │ extends
        │                         ▼
        │              ┌────────────────────┐
        │              │ base.py            │
        │              │ BaseSimulationObj. │
        │              │ SimulationConfig   │
        │              │ SimulationStepOut. │
        │              └─────────┬──────────┘
        │                        │ types referenced
        ▼                        │
   ┌────────────────┐            │
   │ time_step.py   │            │
   │ TimeStep       │            │
   │ TimeStepManager│            │
   └────────────────┘            │
                                 │
        Simulation ─────────────►│
                                 ▼
                       ┌─────────────────┐
                       │ result.py       │
                       │ Step record     │
                       │ SimulationResult│
                       │ Reports x4      │
                       └─────────────────┘
```

Edge details:

* `simulation.py` imports `BaseSimulationObject`, `SimulationConfig`, `SimulationStepOutput` from `base`; `TimeStep`, `TimeStepManager` from `time_step`; `SimulationResult`, `SimulationStepRecord` from `result`.
* `simulation.py` performs *runtime* imports of `upsonic.direct.Direct`, `upsonic.tasks.tasks.Task`, and `rich.console`/`rich.panel` to keep import-time light.
* `result.py` uses `TYPE_CHECKING` to avoid circular references with `base` and `time_step`.
* `scenarios/*` only depends on `base` and `pydantic` — they can be unit-tested without the orchestrator.

## 6. Public API

The intended user-facing API is:

| Symbol                               | Where                                       | Purpose                                                   |
| ------------------------------------ | ------------------------------------------- | --------------------------------------------------------- |
| `Simulation`                         | `upsonic.simulation`                        | Orchestrator                                              |
| `Simulation.run() / arun()`          | `upsonic.simulation.simulation`             | Execute synchronously / asynchronously                    |
| `BaseSimulationObject`               | `upsonic.simulation`                        | Base class for custom scenarios                           |
| `SimulationStepOutput`               | `upsonic.simulation.base`                   | Base Pydantic schema for LLM output (subclass when custom)|
| `SimulationConfig`                   | `upsonic.simulation.base`                   | Internally-built validated config; also serializable      |
| `SimulationResult`                   | `upsonic.simulation`                        | Final result handle                                       |
| `SimulationResult.report(name)`      | `upsonic.simulation.result`                 | One-of `summary | detailed | visual | statistical`        |
| `SimulationResult.reports()`         | `upsonic.simulation.result`                 | `ReportsCollection` for `save_all`                        |
| `SimulationResult.get_metric_series` | `upsonic.simulation.result`                 | Time series for a single metric                           |
| `SimulationResult.to_json(path)`     | `upsonic.simulation.result`                 | Full result dump                                          |
| `TimeStep`                           | `upsonic.simulation`                        | Granularity enum (HOURLY / DAILY / ... / YEARLY)          |
| `MerchantRevenueForecastSimulation`  | `upsonic.simulation.scenarios`              | E-commerce MRR forecast                                   |
| `StockPriceForecastSimulation`       | `upsonic.simulation.scenarios`              | Stock/asset price forecast                                |
| `UserGrowthSimulation`               | `upsonic.simulation.scenarios`              | Digital-product user growth forecast                      |

Report export methods (all chainable):

| Method                     | Output                                                      |
| -------------------------- | ----------------------------------------------------------- |
| `BaseReport.to_dict()`     | Raw dict (per-report shape)                                 |
| `BaseReport.to_json(path)` | JSON file                                                   |
| `BaseReport.to_csv(path)`  | CSV file                                                    |
| `BaseReport.to_pdf(path)`  | PDF file (via `reportlab`)                                  |
| `BaseReport.to_html(path)` | Self-contained HTML, with Chart.js inlined for `VisualReport` |
| `BaseReport.show()`        | Inline render (Jupyter HTML / matplotlib / Rich console)    |

A typical creation of a custom scenario looks like:

```python
from typing import Dict, Any, List, Type
from pydantic import BaseModel, Field
from upsonic.simulation.base import BaseSimulationObject, SimulationStepOutput

class TrafficStepOutput(SimulationStepOutput):
    visitors: int = Field(..., description="Predicted visitors for this step")

class TrafficSimulation(BaseSimulationObject):
    def __init__(self, site: str, baseline: int):
        self.site, self.baseline = site, baseline

    @property
    def name(self) -> str:        return "TrafficForecast"
    @property
    def description(self) -> str: return f"Traffic forecast for {self.site}"

    def get_initial_state(self) -> Dict[str, Any]:
        return {"visitors": self.baseline}

    def build_step_prompt(self, step, previous_state, metrics_to_track, time_step_unit):
        return (
            f"Site: {self.site}\n"
            f"Previous visitors: {previous_state['visitors']:,}\n"
            f"Predict {time_step_unit} {step}'s visitors."
        )

    def get_step_output_schema(self) -> Type[BaseModel]:
        return TrafficStepOutput
```

That is the full surface needed to plug into the orchestrator.

## 7. Integration with rest of Upsonic

The simulation framework leans on the agent stack rather than embedding its own LLM client.

| Upsonic component             | Used for                                                                                          | Symbol                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `upsonic.direct.Direct`       | Runtime LLM caller (`do_async`); model preparation                                                | `Simulation._initialize_model`, `Simulation._execute_step`   |
| `upsonic.tasks.tasks.Task`    | Wraps each step's prompt + `response_format` (Pydantic class) — so structured output is enforced  | `Simulation._execute_step`                                   |
| `pydantic.BaseModel`          | Step-output schemas, validation                                                                   | All `*StepOutput`                                            |
| `rich.console / rich.panel / rich.table` | Live progress, console fallback rendering                                              | `Simulation._print_progress_*`, `BaseReport._print_to_console` |
| `IPython.display`             | Inline rendering inside notebooks                                                                 | `BaseReport.show`                                            |
| `matplotlib`                  | Inline static plotting fallback                                                                   | `VisualReport.show`                                          |
| `reportlab` (optional)        | PDF rendering                                                                                     | `BaseReport.to_pdf`                                          |

Notably the simulation **does not** integrate with:

* `upsonic.team` (no multi-agent delegation) — the LLM is a single forecaster, not a team.
* `upsonic.reliability_layer` — reliability here is just a per-step retry loop with backoff; it does not invoke verifier/editor agents.
* `upsonic.knowledge_base` / RAG — the prompt is purely scenario-derived; no document retrieval.
* `upsonic.storage` — results live in memory; persistence is the caller's job (via `to_json` / `to_csv` / `to_pdf`).
* `upsonic.safety_engine` — outputs are not policy-filtered; consumers can layer this themselves on the structured response.

Because every step is just a `Task` going through `Direct`, *any* model provider Upsonic supports (`openai/...`, `anthropic/...`, Azure, Bedrock) can drive a simulation — the simulator is provider-agnostic.

## 8. End-to-end flow

Below is a step-by-step trace of `simulation.run()` for a 3-day MRR forecast, with the actual call sites labeled.

```
USER
  │
  ▼
Simulation(...)                                  # simulation.py:62
  ├─ TimeStep.from_string("daily")               # time_step.py:28
  ├─ TimeStepManager(DAILY, start_date)          # time_step.py:115
  ├─ SimulationConfig(...).__post_init__         # base.py:296  (validates time_step/duration/temp/retries)
  └─ self._simulation_id = uuid4()
  │
  ▼
.run()                                           # simulation.py:370
  └─ asyncio.run(_run_simulation_async())
       │
       ├─ _initialize_model()                    # simulation.py:157
       │     └─ Direct(model="openai/gpt-4o")._prepare_model()  # upsonic.direct
       │
       ├─ state = simulation_object.get_initial_state()
       │   {"monthly recurring revenue": 50000.0, ...}
       │
       ├─ append step-0 record (no LLM call)
       │
       ├─ for step in 1..3:
       │     │
       │     ├─ prompt = _build_step_prompt(step, state)        # simulation.py:164
       │     │       ├─ simulation_object.build_step_prompt(...)
       │     │       │       (formatted markdown company profile + previous metrics)
       │     │       ├─ time_manager.get_time_context(step)
       │     │       │       {"timestamp": "2026-04-29", "day_of_week": "Wednesday", ...}
       │     │       └─ simulation_object.get_context_for_step(step)
       │     │             "End of month - expect higher activity..."  (every 30 steps)
       │     │
       │     ├─ schema = simulation_object.get_step_output_schema()
       │     │       MerchantRevenueStepOutput
       │     │
       │     ├─ task = Task(description=prompt, response_format=schema)   # tasks.tasks
       │     │
       │     ├─ retry-loop:
       │     │     for attempt in 0..max_retries:
       │     │         try:
       │     │             result = await direct.do_async(task)            # LLM call
       │     │             break
       │     │         except Exception:
       │     │             await asyncio.sleep(0.5 * attempt)
       │     │
       │     ├─ metrics = simulation_object.extract_metrics(result, ...)  # base.py:176 + scenario override
       │     │       fuzzy-match "monthly recurring revenue" -> field
       │     │
       │     ├─ metrics = simulation_object.validate_metrics(metrics, step) # bounds: ≥0, ≤1, etc.
       │     │
       │     ├─ state.update(metrics)            # carry forward
       │     │
       │     └─ append SimulationStepRecord(step, timestamp, prompt,
       │                                    raw_response, parsed_response,
       │                                    metrics, exec_time, success=True)
       │
       └─ return SimulationResult(simulation_id, sim_object, config,
                                  steps, start_time, end_time,
                                  time_manager, metrics_to_track)
  │
  ▼
USER
  result.report("summary").to_pdf("summary.pdf")
       └─ ReportsCollection lazily built
            └─ SummaryReport.to_dict()
                 ├─ counts successful/failed steps
                 ├─ totals execution time
                 ├─ initial vs final metrics, change, % change
                 └─ → reportlab SimpleDocTemplate → PDF

  result.report("visual").show()
       ├─ VisualReport.to_dict()        # zip timestamps + per-metric series
       └─ matplotlib.pyplot.plot()      # inline in Jupyter

  result.reports().save_all("./reports", format="html")
       └─ for each of summary/detailed/visual/statistical:
              report.to_html("./reports/<name>_report.html")
```

A few interesting properties of this flow:

1. **State is plain `dict[str, Any]`.** The carry-forward state is whatever the previous step's `extract_metrics + validate_metrics` produced — no Pydantic enforcement on state itself. Scenarios deliberately store both `"monthly recurring revenue"` and `"monthly_recurring_revenue"` keys so prompts can read by either name.
2. **Step 0 is real but synthetic.** It has empty `prompt` / `raw_response`, `execution_time=0`, and `metrics=initial_state.copy()`. `SimulationResult.total_steps` excludes it.
3. **Retries do not re-randomize.** The same prompt is sent each retry; only `temperature` (validated but not currently passed into the `Task` in the code) controls randomness. *Note*: although `Simulation` validates `temperature`, the value is not forwarded to `Task` — it lives only inside `SimulationConfig` for serialization.
4. **Failure is non-fatal by default.** Failed steps land as `success=False` records and the loop continues with stale state, which produces an honest gap in the output series rather than aborting a 100-step run.
5. **`run()` is reentrant inside Jupyter.** The `try: asyncio.get_running_loop()` branch spawns a `ThreadPoolExecutor` and submits `asyncio.run(_run_simulation_async())` — this avoids the "asyncio.run() cannot be called from a running event loop" error that would otherwise prevent notebook usage.
6. **All four reports are derived from the same `steps` list.** Switching from a summary view to a visual chart does not re-run the simulation; reports are pure post-processing over the recorded `SimulationStepRecord` series.

Together these properties make `src/upsonic/simulation/` a self-contained, scenario-pluggable forecasting layer that re-uses Upsonic's structured-output `Direct` / `Task` machinery for each tick of the simulated clock.
