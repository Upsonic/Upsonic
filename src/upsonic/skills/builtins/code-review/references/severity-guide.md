# Issue Severity Guide

## Critical
Must be fixed before merge. These issues cause real harm.

| Category | Examples |
|----------|----------|
| Security vulnerability | SQL injection, XSS, auth bypass, hardcoded secrets |
| Data loss/corruption | Missing transactions, silent data truncation, race conditions on writes |
| Crash/availability | Unhandled null pointer, infinite loops, OOM on normal input |
| Privacy violation | PII leaked in logs, data exposed to unauthorized users |

## Warning
Should be fixed. These issues will cause problems eventually.

| Category | Examples |
|----------|----------|
| Bug (non-critical) | Wrong calculation in edge case, incorrect error message, off-by-one |
| Performance | N+1 queries, unbounded result sets, missing indexes on common queries |
| Error handling | Swallowed exceptions, missing retry logic on network calls, no timeout |
| Reliability | Missing input validation, no circuit breaker on external calls |

## Suggestion
Nice-to-have. These improve code quality but aren't blocking.

| Category | Examples |
|----------|----------|
| Readability | Confusing variable names, long functions, missing comments on complex logic |
| Maintainability | DRY violations, god classes, tight coupling |
| Style | Inconsistent formatting, non-idiomatic patterns |
| Testing | Missing tests for new behavior, test names unclear |

## Deciding Severity

Ask yourself:
1. **If this ships, what's the worst that happens?** Data loss = Critical. Slower page load = Warning. Ugly code = Suggestion.
2. **How likely is the bad outcome?** A crash on every request = Critical. A crash on leap-year February 29th = Warning.
3. **How many users are affected?** Everyone = bump up. One edge case = keep or bump down.
4. **Is it reversible?** Data loss is not. A UI glitch is. Irreversible outcomes get higher severity.
