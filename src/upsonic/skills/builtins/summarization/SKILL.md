---
name: summarization
description: Summarize documents, articles, conversations, code, and technical content into concise, accurate summaries. Use when user asks to summarize, condense, create a TL;DR, write an executive summary, extract key points, or distill content. Trigger when user says things like "summarize this", "give me the key points", "TL;DR", "what are the main takeaways", "condense this", "brief me on this", or provides long content asking for a shorter version. Also trigger for meeting notes summaries, research paper abstracts, and changelog summaries. Do NOT trigger for rewriting or paraphrasing at similar length, translation, or content generation from scratch.
metadata:
  version: "2.0.0"
  author: Upsonic
  tags: [writing, analysis, content, documents]
---

# Summarization

Produce concise, accurate summaries that capture what matters. A good summary saves the reader time while preserving the information they need to make decisions or take action.

## Before You Summarize

Understand the context:

1. **What type of content is this?** (Business doc, technical spec, conversation, research paper, code)
2. **Who is the audience?** (Executive, engineer, general reader)
3. **What's the purpose?** (Decision-making, catch-up, reference, sharing)
4. **How long should the summary be?** (If not specified, scale to content complexity — not just length)

These determine which approach to use and what to emphasize.

## Reference Materials

- Load `summary-templates.md` for ready-to-use templates for each summary type (Executive, Technical, Meeting, Research, Changelog). Use these as starting structures and adapt to the specific content.

## Summarization Approaches

Choose the approach that matches the content type:

### Executive Summary
For business documents, reports, proposals, and strategy decks.

**Structure:**
1. **Bottom line first**: Lead with the key decision, recommendation, or conclusion
2. **Critical metrics**: Include the 2-4 numbers that matter most (revenue, timeline, cost, impact)
3. **Key findings**: 3-5 bullet points covering what was discovered or decided
4. **Risks and concerns**: What could go wrong, what's uncertain
5. **Next steps**: Who does what by when

**Example pattern:**
```
**Recommendation:** [One sentence with the core decision]

**Key metrics:** [2-4 data points]

**Findings:**
- [Most important finding]
- [Second most important]
- [Third]

**Risks:** [1-2 key risks]

**Next steps:** [Action items with owners and deadlines]
```

**Guidelines for executive summaries:**
- Use business language, not technical jargon
- Every sentence should help the reader make a decision
- If you can't state the bottom line in one sentence, the source material may be unclear — flag this
- Include specific numbers, not vague qualifiers ("revenue grew 23%" not "revenue grew significantly")

### Technical Summary
For technical documents, architecture docs, RFCs, code reviews, and documentation.

**Structure:**
1. **Purpose**: What problem does this solve? Why does it exist?
2. **Approach**: How does it work at a high level?
3. **Key decisions**: What technical choices were made and why?
4. **Dependencies**: What does this rely on? What relies on it?
5. **Trade-offs**: What was gained and what was sacrificed?
6. **Limitations**: Known issues, constraints, or gaps
7. **Open questions**: Unresolved decisions or areas needing further work

**Guidelines for technical summaries:**
- Preserve technical precision — don't simplify terms that have specific meanings
- Include architecture decisions and their rationale
- Note API contracts, data formats, and integration points
- Mention performance characteristics if discussed in the source
- Flag breaking changes or migration requirements

### Research/Academic Summary
For research papers, studies, whitepapers, and analytical reports.

**Structure:**
1. **Research question**: What was being investigated?
2. **Methodology**: How was the study conducted? (brief)
3. **Key findings**: The 3-5 most important results
4. **Significance**: Why do these findings matter?
5. **Limitations**: What the study doesn't cover or can't prove
6. **Implications**: What should change based on these findings?

**Guidelines for research summaries:**
- Distinguish between correlation and causation
- Include sample sizes and confidence levels when available
- Note if results are statistically significant vs practically significant
- Preserve nuance — don't overstate findings
- Flag if the methodology has notable limitations

### Conversation/Meeting Summary
For meeting notes, chat logs, email threads, and discussions.

**Structure:**
1. **Decisions made**: What was agreed upon (be specific)
2. **Action items**: Who is doing what by when
3. **Key discussion points**: The main topics debated
4. **Disagreements**: Where people differed and how it was resolved (or wasn't)
5. **Open questions**: What still needs to be decided
6. **Parking lot**: Topics raised but deferred

**Guidelines for conversation summaries:**
- Attribute decisions and action items to specific people
- Capture the "why" behind decisions, not just the "what"
- Note when consensus was reached vs when a decision was imposed
- Include deadlines and commitments
- Flag anything that seemed unresolved or contentious

### Code/Changelog Summary
For code diffs, pull requests, release notes, and changelogs.

**Structure:**
1. **What changed**: High-level description of the changes
2. **Why**: The motivation (bug fix, feature, refactor, performance)
3. **Impact**: What users/developers will notice
4. **Breaking changes**: Anything that requires migration or adaptation
5. **Notable details**: Interesting implementation choices or caveats

**Guidelines for code summaries:**
- Lead with user-facing impact, not implementation details
- Group related changes together
- Distinguish between bug fixes, features, and internal changes
- Highlight breaking changes prominently
- Note if tests were added or updated

## Core Principles

These apply regardless of which approach you use:

### Accuracy Above All
- Never introduce information that isn't in the source material
- If something is ambiguous in the source, say so — don't guess
- Use the same terminology as the source (don't substitute synonyms that shift meaning)
- If quoting, be exact. If paraphrasing, preserve the original intent

### Structure Matters
- Use bullet points, numbered lists, and headers for scannability
- Put the most important information first (inverted pyramid)
- Group related points together
- Use consistent formatting throughout

### Right-Size the Summary
- A 2-page memo and a 200-page report don't get the same treatment
- Scale to complexity of content, not just its length
- A simple topic covered in 50 pages might need only 3 bullets
- A dense technical spec of 5 pages might need a full page summary

### Preserve Quantitative Data
- Always include specific numbers, dates, and metrics from the source
- "Revenue was $4.2M, up 23% YoY" not "revenue increased significantly"
- Include dates, deadlines, and timelines
- Preserve units and precision

### Handle Uncertainty Honestly
- If the source is unclear, say "The document is unclear on..." rather than guessing
- If information seems contradictory, note the contradiction
- If key information is missing, flag what's absent
- Distinguish between what the source states as fact vs opinion vs speculation

## Multi-Document Summarization

When summarizing multiple documents together:

1. **Read all documents first** before starting the summary
2. **Identify themes** that appear across multiple documents
3. **Note contradictions** between documents and flag them
4. **Attribute information** to specific documents when sources differ
5. **Synthesize, don't just concatenate** — find the narrative across documents

## Quality Checklist

Before delivering a summary, verify:

- [ ] Could someone make a decision based on this summary alone?
- [ ] Are all key numbers and dates preserved?
- [ ] Is anything in the summary not actually in the source?
- [ ] Would the original author agree this is a fair representation?
- [ ] Is the most important information at the top?
- [ ] Is it the right length for the content and audience?
