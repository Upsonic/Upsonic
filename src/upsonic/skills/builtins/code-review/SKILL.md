---
name: code-review
description: Perform structured code reviews with actionable feedback. Use when a user asks to review code, check code quality, find bugs, audit security, improve performance, or assess maintainability. Trigger when user says things like "review this code", "check for bugs", "is this code secure", "any issues with this", "code quality check", or pastes code asking for feedback. Also trigger for pull request reviews and pre-merge code checks. Do NOT trigger for writing new code from scratch, refactoring requests without review context, or general programming questions.
metadata:
  version: "2.0.0"
  author: Upsonic
  tags: [development, quality, review, security, bugs]
---

# Code Review

Perform a structured, multi-dimensional code review. Act as a senior engineer reviewing a colleague's work — be thorough but constructive.

## When to Offer This Workflow

**Trigger conditions:**
- User pastes code and asks for review or feedback
- User mentions "review", "check", "audit", "any issues", "bugs"
- User shares a pull request or diff
- User asks about code quality, security, or performance

**Initial approach:**
Before diving in, understand the context:
1. What language/framework is this?
2. What does this code do? (Read it first, don't ask unless unclear)
3. Is this a snippet or full module?
4. Any specific concerns the user mentioned?

## Review Process

### Phase 1: Understand Before Judging

Read the entire code before making any comments. Understand:
- The purpose and intent of the code
- The architecture and design patterns being used
- The constraints the author may be working under
- Whether this is production code, a prototype, or a learning exercise

This context matters. A quick prototype doesn't need the same scrutiny as a payment processing module.

### Phase 2: Systematic Review

Review across five dimensions, in priority order:

#### 1. Correctness (Highest Priority)
Does the code do what it's supposed to?

Check for:
- Logic errors and off-by-one mistakes
- Null/undefined/nil handling — what happens when data is missing?
- Edge cases: empty inputs, zero values, maximum values, concurrent access
- Error handling: are exceptions caught appropriately? Are errors swallowed silently?
- Race conditions in concurrent code
- Resource leaks (unclosed files, connections, streams)
- Type mismatches or implicit conversions that could cause bugs

#### 2. Security
Could this code be exploited?

Check for:
- **Injection**: SQL injection, XSS, command injection, path traversal
- **Authentication/Authorization**: Missing auth checks, privilege escalation
- **Data exposure**: Secrets in code, verbose error messages leaking internals, PII in logs
- **Input validation**: Trusting user input without sanitization
- **Cryptography**: Weak algorithms, hardcoded keys, improper random number generation
- **Dependencies**: Known vulnerable libraries, outdated packages
- **OWASP Top 10**: Systematically check against common vulnerability classes

#### 3. Performance
Will this code perform well under load?

Check for:
- **Algorithmic complexity**: O(n^2) where O(n) is possible, unnecessary nested loops
- **Database**: N+1 queries, missing indexes, full table scans, unbounded result sets
- **Memory**: Unnecessary allocations, loading entire files into memory, unbounded caches
- **I/O**: Synchronous calls that should be async, missing connection pooling
- **Caching**: Repeated expensive computations that could be cached
- **Batching**: Individual operations that could be batched

#### 4. Maintainability
Can another developer understand and modify this code?

Check for:
- **Naming**: Are variables, functions, and classes named clearly? Could a reader guess what they do?
- **Size**: Functions over 30 lines or classes over 300 lines usually need splitting
- **Single Responsibility**: Does each function/class do one thing well?
- **DRY**: Is logic duplicated? Could shared utilities reduce repetition?
- **Error messages**: Are they actionable? Do they help debugging?
- **Documentation**: Are complex algorithms or business rules explained?
- **Testability**: Could this code be unit tested easily? Are dependencies injectable?

#### 5. Style and Conventions (Lowest Priority)
Does the code follow project and language conventions?

Check for:
- Consistent formatting (indentation, spacing, line length)
- Idiomatic language usage (e.g., list comprehensions in Python, streams in Java)
- Naming conventions (camelCase vs snake_case per language norms)
- Import organization
- Consistent error handling patterns

### Phase 3: Prioritize and Report

## Reference Materials

- Load `severity-guide.md` to understand how to classify issue severity (Critical / Warning / Suggestion) with examples for each level
- Load `owasp-top-10.md` when doing security-focused reviews for a comprehensive checklist of vulnerability categories

## Output Format

Structure your review as follows:

### Critical Issues
Issues that must be fixed — bugs, security vulnerabilities, data loss risks.

For each issue:
- **Location**: File and line number(s)
- **What's wrong**: Clear explanation of the problem
- **Why it matters**: The real-world impact (data loss, security breach, crash)
- **Fix**: Concrete code suggestion

### Warnings
Issues that should be fixed — performance problems, potential bugs, poor error handling.

Same format as critical issues.

### Suggestions
Nice-to-have improvements — readability, style, minor refactors.

Keep these brief. Don't nitpick.

### What's Done Well
Acknowledge good patterns, clean abstractions, thorough error handling, or clever solutions. This matters — it reinforces good practices and shows you read the code carefully.

## Guidelines

- **Lead with the most important issues.** If there's a SQL injection vulnerability, that matters more than variable naming.
- **Be specific.** "This could be improved" is useless. "Line 42: this SQL query concatenates user input directly — use parameterized queries to prevent injection" is actionable.
- **Show, don't just tell.** When suggesting a fix, include a code snippet showing the improvement.
- **Explain the why.** Don't just say "use a Set instead of Array" — explain that membership checks are O(1) vs O(n), and it matters here because the array is checked inside a loop.
- **Consider the author's intent.** If code looks intentionally structured a certain way, ask about it before suggesting changes.
- **Scale your review.** A 10-line utility function doesn't need 50 comments. A payment processing module deserves deep scrutiny.
- **Don't flag things that are clearly intentional.** If someone uses `# type: ignore`, they probably have a reason.
- **Group related issues.** If the same pattern appears multiple times, mention it once with all locations rather than repeating yourself.

## Common Patterns to Watch For

### Python
- Mutable default arguments (`def foo(items=[])`)
- Bare `except:` catching everything including KeyboardInterrupt
- `==` vs `is` for None/True/False comparisons
- Missing `__init__` in packages
- String formatting with `%` or `.format()` where f-strings are clearer

### JavaScript/TypeScript
- `==` vs `===` (prefer strict equality)
- Missing `await` on async functions
- Callback hell that should use async/await
- `var` usage (prefer `const`/`let`)
- Missing error boundaries in React components

### General
- TODO/FIXME/HACK comments that have been there too long
- Dead code (unreachable branches, unused functions)
- Magic numbers without named constants
- Overly complex conditionals that need extraction or truth tables
- Missing input validation at system boundaries

## Handling Specific Review Types

### Pull Request Reviews
When reviewing a PR or diff:
- Focus on the changes, not pre-existing code (unless changes make existing issues worse)
- Check that tests cover the new behavior
- Verify the PR description matches what the code does
- Look for incomplete migrations or partial refactors

### Security Audits
When the user specifically asks about security:
- Be more thorough on the security dimension
- Check for OWASP Top 10 systematically — load `owasp-top-10.md` from references for the full checklist
- Review authentication and authorization flows end-to-end
- Check for information disclosure in error messages and logs
- Review dependency versions against known CVE databases

### Performance Reviews
When the user specifically asks about performance:
- Focus on algorithmic complexity
- Look for database query patterns
- Check for unnecessary I/O
- Suggest profiling if the bottleneck isn't obvious from code inspection
