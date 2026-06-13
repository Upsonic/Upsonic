"""
Visual before/after test for ContextManagementMiddleware with assertions.

Run with:
    uv run pytest tests/smoke_tests/agent/test_context_management_visual.py -s -v

Three test cases covering all three stages of the apply() pipeline:
  1. Tool pruning alone resolves the overflow
  2. Tool pruning + LLM summarization resolves the overflow
  3. Context still full after all strategies (context_full=True)

All three use the SAME chat history with different safety_margin_ratio
values to trigger different resolution stages.
"""

from typing import Any, List

import pytest

from upsonic.agent.context_managers.context_management_middleware import (
    ContextManagementMiddleware,
)
from upsonic.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from upsonic.models import infer_model
from tests.smoke_tests._model_selection import without_model_override
from upsonic.usage import RequestUsage

SYSTEM_PROMPT_CONTENT: str = (
    "You are a senior data analyst assistant with access to database_query, "
    "chart_generator, report_writer, email_sender, scheduler, forecaster, "
    "anomaly_detector, and data_exporter tools. Always provide thorough analysis."
)

TOOL_ROUNDS: List[tuple[str, str, str, str]] = [
    ("database_query", "tc_q1",
     '{"sql": "SELECT region, category, SUM(revenue) FROM sales WHERE year=2025 AND quarter=1 GROUP BY region, category ORDER BY revenue DESC"}',
     "Q1 2025: NA-Enterprise=$8.2M, NA-SMB=$3.1M, NA-Services=$1.2M, EU-Enterprise=$5.1M, EU-SMB=$2.0M, "
     "APAC-Enterprise=$4.2M, APAC-SMB=$1.8M, LATAM-Enterprise=$2.0M, MEA-Enterprise=$0.9M. Total=$32.0M"),
    ("database_query", "tc_q2",
     '{"sql": "SELECT region, category, SUM(revenue) FROM sales WHERE year=2025 AND quarter=2 GROUP BY region, category ORDER BY revenue DESC"}',
     "Q2 2025: NA-Enterprise=$9.5M, NA-SMB=$3.4M, NA-Services=$1.3M, EU-Enterprise=$5.6M, EU-SMB=$2.2M, "
     "APAC-Enterprise=$5.0M, APAC-SMB=$2.1M, LATAM-Enterprise=$2.3M, MEA-Enterprise=$1.1M. Total=$36.2M (+13% QoQ)"),
    ("database_query", "tc_q3",
     '{"sql": "SELECT region, category, SUM(revenue) FROM sales WHERE year=2025 AND quarter=3 GROUP BY region, category ORDER BY revenue DESC"}',
     "Q3 2025: NA-Enterprise=$10.1M, NA-SMB=$3.6M, NA-Services=$1.4M, EU-Enterprise=$5.9M, EU-SMB=$2.3M, "
     "APAC-Enterprise=$5.5M, APAC-SMB=$2.3M, LATAM-Enterprise=$2.5M, MEA-Enterprise=$1.2M. Total=$39.1M (+8% QoQ)"),
    ("database_query", "tc_q4",
     '{"sql": "SELECT region, category, SUM(revenue) FROM sales WHERE year=2025 AND quarter=4 GROUP BY region, category ORDER BY revenue DESC"}',
     "Q4 2025: NA-Enterprise=$11.3M, NA-SMB=$4.0M, NA-Services=$1.6M, EU-Enterprise=$6.4M, EU-SMB=$2.5M, "
     "APAC-Enterprise=$6.1M, APAC-SMB=$2.6M, LATAM-Enterprise=$2.8M, MEA-Enterprise=$1.3M. Total=$43.0M (+10% QoQ)"),
    ("anomaly_detector", "tc_anomaly",
     '{"dataset": "revenue_2025_all_regions", "sensitivity": "high", "compare_baseline": "2024"}',
     "Anomalies detected: 1) APAC-SMB Q3 spike +28% (new partnership with Rakuten), "
     "2) MEA-Enterprise Q2 dip -12% (regulatory delay in UAE), 3) LATAM-Services Q4 surge +45% "
     "(Brazil tax incentive program). Confidence: 94%. All flagged for review."),
    ("chart_generator", "tc_charts",
     '{"charts": ["revenue_by_region_stacked_bar", "yoy_growth_line", "category_pie_q4", "anomaly_scatter"]}',
     "Generated 4 charts: 1) Stacked bar (revenue by region per quarter), 2) Line chart (YoY growth trends), "
     "3) Pie chart (Q4 category breakdown), 4) Scatter plot (anomaly visualization). All saved to /reports/charts/"),
    ("report_writer", "tc_report",
     '{"template": "executive_summary", "include_charts": true, "format": "pdf"}',
     "Executive report generated: 24 pages, includes all 4 charts, executive summary, regional deep-dives, "
     "anomaly analysis, and strategic recommendations. Saved to /reports/FY2025_Revenue_Analysis.pdf"),
    ("scheduler", "tc_schedule",
     '{"event": "Board Revenue Review", "date": "2026-03-15", "attendees": ["ceo@company.com", "cfo@company.com"]}',
     "Meeting scheduled: 'Board Revenue Review' on March 15, 2026 at 10:00 AM EST. "
     "Calendar invites sent to CEO and CFO. Report attached to invite."),
]


def _get_model() -> Any:
    with without_model_override():  # pin known 200k window past the gpt-5 override
        return infer_model("anthropic/claude-sonnet-4-5")


def _build_rich_chat_history() -> List[Any]:
    """Build a realistic 22-message chat history with all part types."""
    msgs: List[Any] = []

    msgs.append(ModelRequest(parts=[
        SystemPromptPart(content=SYSTEM_PROMPT_CONTENT),
        UserPromptPart(content=(
            "I need a comprehensive analysis of our company's Q1-Q4 2025 revenue "
            "performance across all regions (North America, Europe, Asia-Pacific, "
            "Latin America, Middle East & Africa). Include year-over-year comparisons, "
            "identify trends, and flag any anomalies."
        )),
    ]))

    msgs.append(ModelResponse(
        parts=[TextPart(content=(
            "I'll conduct a comprehensive revenue analysis for FY2025 across all "
            "five regions. My plan: 1) Query quarterly revenue data, 2) Run YoY "
            "comparisons, 3) Detect anomalies, 4) Generate visualizations, "
            "5) Build a forecast, 6) Compile the final report, 7) Export data, "
            "8) Schedule the board presentation. Let me start with the database queries."
        ))],
        model_name="claude-sonnet-4-5-20250929",
        usage=RequestUsage(input_tokens=15000, output_tokens=5000),
        finish_reason="stop",
    ))

    msgs.append(ModelRequest(parts=[
        UserPromptPart(content=(
            "Yes, please run all the queries. Also compare with 2024 numbers and "
            "include product category breakdowns (Enterprise SaaS, SMB SaaS, "
            "Professional Services, Training & Certification)."
        )),
    ]))

    for tool_name, tc_id, args, result in TOOL_ROUNDS:
        msgs.append(ModelResponse(
            parts=[ToolCallPart(tool_name=tool_name, tool_call_id=tc_id, args=args)],
            model_name="claude-sonnet-4-5-20250929",
            usage=RequestUsage(input_tokens=30000, output_tokens=5000),
            finish_reason="tool_call",
        ))
        msgs.append(ModelRequest(parts=[ToolReturnPart(
            tool_name=tool_name, tool_call_id=tc_id, content=result,
        )]))

    msgs.append(ModelResponse(
        parts=[TextPart(content=(
            "Here's the complete FY2025 Revenue Analysis:\n\n"
            "TOTAL REVENUE: $150.3M (up 22% YoY from $123.2M in 2024)\n\n"
            "Regional Performance:\n"
            "- North America: $52.7M (35% of total) -- strongest growth in Enterprise SaaS\n"
            "- Europe: $24.0M (16%) -- steady growth, EU-SMB accelerating\n"
            "- Asia-Pacific: $22.6M (15%) -- fastest growing region (+28% YoY)\n"
            "- Latin America: $9.6M (6%) -- Services segment surging\n"
            "- Middle East & Africa: $4.5M (3%) -- recovered from Q2 regulatory dip\n\n"
            "Key Findings: Enterprise SaaS drives 62% of revenue. APAC is the growth engine. "
            "Three anomalies flagged and explained. Board presentation scheduled for March 15."
        ))],
        model_name="claude-sonnet-4-5-20250929",
        usage=RequestUsage(input_tokens=15000, output_tokens=5000),
        finish_reason="stop",
    ))

    msgs.append(ModelRequest(parts=[
        UserPromptPart(content=(
            "Excellent work! One last thing -- what are your top 3 strategic "
            "recommendations for Q1 2026 based on this analysis?"
        )),
    ]))

    msgs.append(ModelResponse(
        parts=[TextPart(content=(
            "Based on the FY2025 analysis, my top 3 strategic recommendations for Q1 2026:\n\n"
            "1. DOUBLE DOWN ON APAC: Invest in the Rakuten partnership and expand Enterprise "
            "SaaS presence in Japan, South Korea, and Australia. APAC showed +28% YoY growth -- "
            "the highest of any region.\n\n"
            "2. ACCELERATE SMB SELF-SERVE: SMB SaaS grew 18% across all regions with minimal "
            "sales investment. Launch a self-serve onboarding flow to reduce CAC by ~30% and "
            "capture the long tail of smaller accounts.\n\n"
            "3. RESOLVE MEA REGULATORY BLOCKERS: The Q2 UAE regulatory delay cost ~$1.3M. "
            "Engage local legal counsel proactively for 2026 compliance in UAE, Saudi Arabia, "
            "and Egypt to prevent recurrence."
        ))],
        model_name="claude-sonnet-4-5-20250929",
        usage=RequestUsage(input_tokens=15000, output_tokens=5000),
        finish_reason="stop",
    ))

    return msgs


def _print_full_message_detail(label: str, messages: List[Any]) -> None:
    print(f"\n{'~'*80}")
    print(f"  {label}  ({len(messages)} messages)")
    print(f"{'~'*80}")

    total_input: int = 0
    total_output: int = 0

    for i, msg in enumerate(messages):
        print(f"\n  [{i:2d}] {type(msg).__name__}")

        if isinstance(msg, ModelRequest):
            print(f"       timestamp={getattr(msg, 'timestamp', None)}")
            for j, p in enumerate(msg.parts):
                ptype: str = type(p).__name__
                if isinstance(p, SystemPromptPart):
                    print(f"       part[{j}] {ptype}:")
                    print(f"         content={p.content!r}")
                elif isinstance(p, UserPromptPart):
                    print(f"       part[{j}] {ptype}:")
                    print(f"         content={p.content!r}")
                    print(f"         timestamp={getattr(p, 'timestamp', None)}")
                elif isinstance(p, ToolReturnPart):
                    print(f"       part[{j}] {ptype}:")
                    print(f"         tool_name={p.tool_name!r}")
                    print(f"         tool_call_id={p.tool_call_id!r}")
                    print(f"         content={p.content!r}")
                    print(f"         timestamp={getattr(p, 'timestamp', None)}")
                else:
                    print(f"       part[{j}] {ptype}: {p!r}")

        elif isinstance(msg, ModelResponse):
            print(f"       model_name={msg.model_name!r}")
            print(f"       timestamp={msg.timestamp!r}")
            print(f"       finish_reason={msg.finish_reason!r}")
            usage = msg.usage
            print(f"       usage.input_tokens={usage.input_tokens}")
            print(f"       usage.output_tokens={usage.output_tokens}")
            if usage.input_tokens > 0 or usage.output_tokens > 0:
                total_input += usage.input_tokens
                total_output += usage.output_tokens
            for j, p in enumerate(msg.parts):
                ptype_r: str = type(p).__name__
                if isinstance(p, TextPart):
                    print(f"       part[{j}] {ptype_r}:")
                    print(f"         content={p.content!r}")
                elif isinstance(p, ToolCallPart):
                    print(f"       part[{j}] {ptype_r}:")
                    print(f"         tool_name={p.tool_name!r}")
                    print(f"         tool_call_id={p.tool_call_id!r}")
                    print(f"         args={p.args!r}")
                else:
                    print(f"       part[{j}] {ptype_r}: {p!r}")

    print(f"\n  TOKEN TOTALS: input={total_input:,}  output={total_output:,}  sum={total_input + total_output:,}")
    print(f"{'~'*80}\n")


def _assert_message_is_request(msg: Any, idx: int) -> None:
    print(f"  ASSERT output[{idx}] is ModelRequest ... ", end="")
    assert isinstance(msg, ModelRequest), f"output[{idx}] expected ModelRequest, got {type(msg).__name__}"
    print("OK")


def _assert_message_is_response(msg: Any, idx: int) -> None:
    print(f"  ASSERT output[{idx}] is ModelResponse ... ", end="")
    assert isinstance(msg, ModelResponse), f"output[{idx}] expected ModelResponse, got {type(msg).__name__}"
    print("OK")


def _assert_part_type(msg: Any, part_idx: int, expected_type: type, msg_idx: int) -> None:
    part = msg.parts[part_idx]
    label: str = expected_type.__name__
    print(f"  ASSERT output[{msg_idx}].parts[{part_idx}] is {label} ... ", end="")
    assert isinstance(part, expected_type), (
        f"output[{msg_idx}].parts[{part_idx}] expected {label}, got {type(part).__name__}"
    )
    print("OK")


def _assert_part_count(msg: Any, expected: int, msg_idx: int) -> None:
    print(f"  ASSERT output[{msg_idx}] has {expected} part(s) ... ", end="")
    assert len(msg.parts) == expected, (
        f"output[{msg_idx}] expected {expected} parts, got {len(msg.parts)}"
    )
    print("OK")


def _assert_identical_content(actual: str, expected: str, label: str) -> None:
    print(f"  ASSERT {label} content identical ... ", end="")
    assert actual == expected, f"{label} content differs:\n  actual={actual!r}\n  expected={expected!r}"
    print("OK")


def _assert_field_equal(actual: Any, expected: Any, label: str) -> None:
    print(f"  ASSERT {label} == {expected!r} ... ", end="")
    assert actual == expected, f"{label}: expected {expected!r}, got {actual!r}"
    print("OK")


def _assert_field_not_none(value: Any, label: str) -> None:
    print(f"  ASSERT {label} is not None ... ", end="")
    assert value is not None, f"{label} is None"
    print("OK")


def _assert_content_shorter(summarized: str, original: str, label: str) -> None:
    print(f"  ASSERT {label} is shorter ({len(summarized)} < {len(original)}) ... ", end="")
    assert len(summarized) < len(original), (
        f"{label}: summarized ({len(summarized)} chars) not shorter than original ({len(original)} chars)"
    )
    print("OK")


def _assert_preserved_response(
    out_msg: Any,
    in_msg: Any,
    out_idx: int,
    in_idx: int,
) -> None:
    """Assert a ModelResponse was preserved byte-identical from input."""
    print(f"\n  -- output[{out_idx}] should be PRESERVED from input[{in_idx}] --")
    _assert_message_is_response(out_msg, out_idx)
    _assert_part_count(out_msg, len(in_msg.parts), out_idx)

    for j, (op, ip) in enumerate(zip(out_msg.parts, in_msg.parts)):
        _assert_part_type(out_msg, j, type(ip), out_idx)
        if isinstance(ip, TextPart):
            _assert_identical_content(op.content, ip.content, f"output[{out_idx}].parts[{j}]")
        elif isinstance(ip, ToolCallPart):
            _assert_field_equal(op.tool_name, ip.tool_name, f"output[{out_idx}].parts[{j}].tool_name")
            _assert_field_equal(op.tool_call_id, ip.tool_call_id, f"output[{out_idx}].parts[{j}].tool_call_id")
            _assert_field_equal(op.args, ip.args, f"output[{out_idx}].parts[{j}].args")

    _assert_field_equal(out_msg.usage.input_tokens, in_msg.usage.input_tokens, f"output[{out_idx}].usage.input_tokens")
    _assert_field_equal(out_msg.usage.output_tokens, in_msg.usage.output_tokens, f"output[{out_idx}].usage.output_tokens")
    _assert_field_equal(out_msg.finish_reason, in_msg.finish_reason, f"output[{out_idx}].finish_reason")
    _assert_field_equal(out_msg.model_name, in_msg.model_name, f"output[{out_idx}].model_name")
    _assert_field_equal(out_msg.timestamp, in_msg.timestamp, f"output[{out_idx}].timestamp")


def _assert_preserved_request(
    out_msg: Any,
    in_msg: Any,
    out_idx: int,
    in_idx: int,
) -> None:
    """Assert a ModelRequest was preserved byte-identical from input."""
    print(f"\n  -- output[{out_idx}] should be PRESERVED from input[{in_idx}] --")
    _assert_message_is_request(out_msg, out_idx)
    _assert_part_count(out_msg, len(in_msg.parts), out_idx)

    for j, (op, ip) in enumerate(zip(out_msg.parts, in_msg.parts)):
        _assert_part_type(out_msg, j, type(ip), out_idx)
        if isinstance(ip, UserPromptPart):
            _assert_identical_content(op.content, ip.content, f"output[{out_idx}].parts[{j}]")
        elif isinstance(ip, ToolReturnPart):
            _assert_field_equal(op.tool_name, ip.tool_name, f"output[{out_idx}].parts[{j}].tool_name")
            _assert_field_equal(op.tool_call_id, ip.tool_call_id, f"output[{out_idx}].parts[{j}].tool_call_id")
            _assert_identical_content(str(op.content), str(ip.content), f"output[{out_idx}].parts[{j}].content")


def _assert_summarized_structure(
    result: List[Any],
    input_msgs: List[Any],
    expected_count: int,
    summarized_range: tuple[int, int],
) -> None:
    """Assert that summarized messages preserve structure (types, part counts)."""
    start, end = summarized_range
    print(f"\n  -- Checking summarized messages output[{start}:{end}] structure --")

    for i in range(start, end):
        out_msg = result[i]
        if isinstance(out_msg, ModelRequest):
            _assert_message_is_request(out_msg, i)
            for j, p in enumerate(out_msg.parts):
                if isinstance(p, SystemPromptPart):
                    _assert_identical_content(
                        p.content, SYSTEM_PROMPT_CONTENT,
                        f"output[{i}].parts[{j}] SystemPrompt",
                    )
                    print(f"  ASSERT output[{i}].parts[{j}] SystemPrompt INTACT ... OK")
                elif isinstance(p, UserPromptPart):
                    _assert_field_not_none(p.content, f"output[{i}].parts[{j}].content")
                    print(f"  ASSERT output[{i}].parts[{j}] UserPrompt has content ... OK")
                elif isinstance(p, ToolReturnPart):
                    _assert_field_not_none(p.tool_name, f"output[{i}].parts[{j}].tool_name")
                    _assert_field_not_none(p.tool_call_id, f"output[{i}].parts[{j}].tool_call_id")
                    _assert_field_not_none(p.content, f"output[{i}].parts[{j}].content")
                    print(f"  ASSERT output[{i}].parts[{j}] ToolReturn({p.tool_name}, {p.tool_call_id}) preserved ... OK")

        elif isinstance(out_msg, ModelResponse):
            _assert_message_is_response(out_msg, i)
            _assert_field_not_none(out_msg.model_name, f"output[{i}].model_name")
            _assert_field_not_none(out_msg.timestamp, f"output[{i}].timestamp")
            _assert_field_not_none(out_msg.finish_reason, f"output[{i}].finish_reason")

            has_tool_call: bool = any(isinstance(p, ToolCallPart) for p in out_msg.parts)
            expected_fr: str = "tool_call" if has_tool_call else "stop"
            _assert_field_equal(
                out_msg.finish_reason, expected_fr,
                f"output[{i}].finish_reason",
            )

            for j, p in enumerate(out_msg.parts):
                if isinstance(p, TextPart):
                    _assert_field_not_none(p.content, f"output[{i}].parts[{j}].content")
                    print(f"  ASSERT output[{i}].parts[{j}] TextPart has content ... OK")
                elif isinstance(p, ToolCallPart):
                    _assert_field_not_none(p.tool_name, f"output[{i}].parts[{j}].tool_name")
                    _assert_field_not_none(p.tool_call_id, f"output[{i}].parts[{j}].tool_call_id")
                    _assert_field_not_none(p.args, f"output[{i}].parts[{j}].args")
                    print(f"  ASSERT output[{i}].parts[{j}] ToolCall({p.tool_name}, {p.tool_call_id}) preserved ... OK")


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Tool pruning ALONE resolves
# ═══════════════════════════════════════════════════════════════════════

class TestStage1ToolPruningResolves:
    @pytest.mark.asyncio
    async def test_tool_pruning_resolves(self) -> None:
        print("\n\n" + "=" * 80)
        print("  TEST 1: Stage 1 -- Tool Pruning Resolves")
        print("  safety_margin_ratio=0.75 -> limit=150,000")
        print("=" * 80)

        model = _get_model()
        msgs: List[Any] = _build_rich_chat_history()

        print("\n  >>> BEFORE apply():")
        _print_full_message_detail("INPUT", msgs)

        mw = ContextManagementMiddleware(
            model=model, keep_recent_count=2, safety_margin_ratio=0.75,
        )
        result, ctx_full = await mw.apply(msgs)

        print("\n  >>> AFTER apply():")
        _print_full_message_detail("OUTPUT", result)

        # -- context_full --
        print(f"\n  ASSERT context_full == False ... ", end="")
        assert ctx_full is False
        print("OK")

        # -- message count: 22 -> 10 (12 pruned = 6 old rounds x 2 messages each) --
        print(f"  ASSERT message count == 10 ... ", end="")
        assert len(result) == 10, f"Expected 10, got {len(result)}"
        print("OK")

        # -- msg[0]: original SystemPrompt + UserPrompt preserved --
        print("\n  -- output[0]: SystemPrompt + UserPrompt --")
        _assert_message_is_request(result[0], 0)
        _assert_part_count(result[0], 2, 0)
        _assert_part_type(result[0], 0, SystemPromptPart, 0)
        _assert_part_type(result[0], 1, UserPromptPart, 0)
        _assert_identical_content(result[0].parts[0].content, msgs[0].parts[0].content, "output[0].parts[0] SystemPrompt")
        _assert_identical_content(result[0].parts[1].content, msgs[0].parts[1].content, "output[0].parts[1] UserPrompt")

        # -- msg[1]: original TextPart response preserved --
        _assert_preserved_response(result[1], msgs[1], 1, 1)

        # -- msg[2]: original UserPrompt preserved --
        _assert_preserved_request(result[2], msgs[2], 2, 2)

        # -- msg[3-4]: report_writer tool round (was input[15-16]) --
        _assert_preserved_response(result[3], msgs[15], 3, 15)
        _assert_preserved_request(result[4], msgs[16], 4, 16)

        # -- msg[5-6]: scheduler tool round (was input[17-18]) --
        _assert_preserved_response(result[5], msgs[17], 5, 17)
        _assert_preserved_request(result[6], msgs[18], 6, 18)

        # -- msg[7]: post-tool text response (was input[19]) --
        _assert_preserved_response(result[7], msgs[19], 7, 19)

        # -- msg[8]: final user question (was input[20]) --
        _assert_preserved_request(result[8], msgs[20], 8, 20)

        # -- msg[9]: final text response (was input[21]) --
        _assert_preserved_response(result[9], msgs[21], 9, 21)

        print("\n" + "=" * 80)
        print("  TEST 1 PASSED -- all assertions OK")
        print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Tool pruning + Summarization resolves
# ═══════════════════════════════════════════════════════════════════════

class TestStage2SummarizationResolves:
    @pytest.mark.asyncio
    async def test_summarization_resolves(self) -> None:
        print("\n\n" + "=" * 80)
        print("  TEST 2: Stage 2 -- Summarization Resolves")
        print("  safety_margin_ratio=0.40 -> limit=80,000")
        print("=" * 80)

        model = _get_model()
        msgs: List[Any] = _build_rich_chat_history()

        print("\n  >>> BEFORE apply():")
        _print_full_message_detail("INPUT", msgs)

        mw = ContextManagementMiddleware(
            model=model, keep_recent_count=2, safety_margin_ratio=0.40,
        )
        result, ctx_full = await mw.apply(msgs)

        print("\n  >>> AFTER apply():")
        _print_full_message_detail("OUTPUT", result)

        # -- context_full --
        print(f"\n  ASSERT context_full == False ... ", end="")
        assert ctx_full is False
        print("OK")

        # -- message count preserved: 10 in (after pruning) -> 10 out --
        print(f"  ASSERT message count == 10 ... ", end="")
        assert len(result) == 10, f"Expected 10, got {len(result)}"
        print("OK")

        # -- output[0]: SystemPrompt (intact) + UserPrompt (summarized) --
        print("\n  -- output[0]: SystemPrompt + UserPrompt --")
        _assert_message_is_request(result[0], 0)
        _assert_part_count(result[0], 2, 0)
        _assert_part_type(result[0], 0, SystemPromptPart, 0)
        _assert_part_type(result[0], 1, UserPromptPart, 0)
        _assert_identical_content(
            result[0].parts[0].content, SYSTEM_PROMPT_CONTENT,
            "output[0].parts[0] SystemPrompt INTACT",
        )
        _assert_content_shorter(
            result[0].parts[1].content, msgs[0].parts[1].content,
            "output[0].parts[1] UserPrompt condensed",
        )

        # -- output[1]: TextPart response (summarized) --
        print("\n  -- output[1]: Summarized TextPart response --")
        _assert_message_is_response(result[1], 1)
        _assert_part_count(result[1], 1, 1)
        _assert_part_type(result[1], 0, TextPart, 1)
        _assert_field_equal(result[1].finish_reason, "stop", "output[1].finish_reason")
        _assert_field_not_none(result[1].model_name, "output[1].model_name")
        _assert_field_not_none(result[1].timestamp, "output[1].timestamp")
        _assert_content_shorter(
            result[1].parts[0].content, msgs[1].parts[0].content,
            "output[1].parts[0] TextPart condensed",
        )

        # -- output[2]: UserPrompt (summarized) --
        print("\n  -- output[2]: Summarized UserPrompt --")
        _assert_message_is_request(result[2], 2)
        _assert_part_count(result[2], 1, 2)
        _assert_part_type(result[2], 0, UserPromptPart, 2)
        _assert_content_shorter(
            result[2].parts[0].content, msgs[2].parts[0].content,
            "output[2].parts[0] UserPrompt condensed",
        )

        # -- output[3]: ToolCall report_writer (summarized but structure preserved) --
        print("\n  -- output[3]: Summarized ToolCall(report_writer) --")
        _assert_message_is_response(result[3], 3)
        _assert_part_count(result[3], 1, 3)
        _assert_part_type(result[3], 0, ToolCallPart, 3)
        _assert_field_equal(result[3].parts[0].tool_name, "report_writer", "output[3].parts[0].tool_name")
        _assert_field_equal(result[3].parts[0].tool_call_id, "tc_report", "output[3].parts[0].tool_call_id")
        _assert_field_equal(
            result[3].parts[0].args,
            '{"template": "executive_summary", "include_charts": true, "format": "pdf"}',
            "output[3].parts[0].args",
        )
        _assert_field_equal(result[3].finish_reason, "tool_call", "output[3].finish_reason")

        # -- output[4]: ToolReturn report_writer (summarized) --
        print("\n  -- output[4]: Summarized ToolReturn(report_writer) --")
        _assert_message_is_request(result[4], 4)
        _assert_part_count(result[4], 1, 4)
        _assert_part_type(result[4], 0, ToolReturnPart, 4)
        _assert_field_equal(result[4].parts[0].tool_name, "report_writer", "output[4].parts[0].tool_name")
        _assert_field_equal(result[4].parts[0].tool_call_id, "tc_report", "output[4].parts[0].tool_call_id")
        _assert_field_not_none(result[4].parts[0].content, "output[4].parts[0].content")

        # -- output[5]: ToolCall scheduler (summarized but structure preserved) --
        print("\n  -- output[5]: Summarized ToolCall(scheduler) --")
        _assert_message_is_response(result[5], 5)
        _assert_part_count(result[5], 1, 5)
        _assert_part_type(result[5], 0, ToolCallPart, 5)
        _assert_field_equal(result[5].parts[0].tool_name, "scheduler", "output[5].parts[0].tool_name")
        _assert_field_equal(result[5].parts[0].tool_call_id, "tc_schedule", "output[5].parts[0].tool_call_id")
        _assert_field_equal(
            result[5].parts[0].args,
            '{"event": "Board Revenue Review", "date": "2026-03-15", "attendees": ["ceo@company.com", "cfo@company.com"]}',
            "output[5].parts[0].args",
        )
        _assert_field_equal(result[5].finish_reason, "tool_call", "output[5].finish_reason")

        # -- output[6-9]: Recent 2 pairs preserved verbatim --
        _assert_preserved_request(result[6], msgs[18], 6, 18)
        _assert_preserved_response(result[7], msgs[19], 7, 19)
        _assert_preserved_request(result[8], msgs[20], 8, 20)
        _assert_preserved_response(result[9], msgs[21], 9, 21)

        print("\n" + "=" * 80)
        print("  TEST 2 PASSED -- all assertions OK")
        print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: Context full after all strategies
# ═══════════════════════════════════════════════════════════════════════

class TestStage3ContextFull:
    @pytest.mark.asyncio
    async def test_context_full(self) -> None:
        print("\n\n" + "=" * 80)
        print("  TEST 3: Stage 3 -- Context Full (all strategies exhausted)")
        print("  safety_margin_ratio=0.15 -> limit=30,000")
        print("=" * 80)

        model = _get_model()
        msgs: List[Any] = _build_rich_chat_history()

        print("\n  >>> BEFORE apply():")
        _print_full_message_detail("INPUT", msgs)

        mw = ContextManagementMiddleware(
            model=model, keep_recent_count=2, safety_margin_ratio=0.15,
        )
        result, ctx_full = await mw.apply(msgs)

        print("\n  >>> AFTER apply():")
        _print_full_message_detail("OUTPUT", result)

        # -- context_full --
        print(f"\n  ASSERT context_full == True ... ", end="")
        assert ctx_full is True
        print("OK")

        # -- message count preserved: 10 in -> 10 out (summarization preserves structure) --
        print(f"  ASSERT message count == 10 ... ", end="")
        assert len(result) == 10, f"Expected 10, got {len(result)}"
        print("OK")

        # -- output[0]: SystemPrompt intact + UserPrompt summarized --
        print("\n  -- output[0]: SystemPrompt + UserPrompt --")
        _assert_message_is_request(result[0], 0)
        _assert_part_count(result[0], 2, 0)
        _assert_part_type(result[0], 0, SystemPromptPart, 0)
        _assert_part_type(result[0], 1, UserPromptPart, 0)
        _assert_identical_content(
            result[0].parts[0].content, SYSTEM_PROMPT_CONTENT,
            "output[0].parts[0] SystemPrompt INTACT",
        )

        # -- Summarized section: check structure of output[0..5] --
        _assert_summarized_structure(result, msgs, 10, (0, 6))

        # -- Recent 2 pairs preserved verbatim (output[6..9]) --
        _assert_preserved_request(result[6], msgs[18], 6, 18)
        _assert_preserved_response(result[7], msgs[19], 7, 19)
        _assert_preserved_request(result[8], msgs[20], 8, 20)
        _assert_preserved_response(result[9], msgs[21], 9, 21)

        # -- Token estimation still exceeds limit (that's why context_full=True) --
        tokens_after: int = mw._estimate_message_tokens(result)
        limit: int = int(200000 * 0.15)
        print(f"\n  ASSERT tokens ({tokens_after:,}) > limit ({limit:,}) ... ", end="")
        assert tokens_after > limit, f"Expected tokens > {limit}, got {tokens_after}"
        print("OK")

        print("\n" + "=" * 80)
        print("  TEST 3 PASSED -- all assertions OK")
        print("=" * 80)
