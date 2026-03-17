# OWASP Top 10 Quick Reference

## A01:2021 — Broken Access Control
- Missing function-level access checks
- IDOR (Insecure Direct Object Reference): accessing other users' data by changing IDs
- Missing CORS configuration or overly permissive origins
- Accessing API endpoints without proper authentication

**What to look for in code:** Authorization checks on every endpoint, object-level permission validation, proper CORS headers.

## A02:2021 — Cryptographic Failures
- Sensitive data transmitted in cleartext (HTTP, FTP, SMTP)
- Weak or deprecated cryptographic algorithms (MD5, SHA1 for passwords, DES)
- Hardcoded encryption keys or passwords
- Missing encryption for PII at rest

**What to look for in code:** Password hashing (bcrypt/argon2), TLS configuration, key management, data classification.

## A03:2021 — Injection
- SQL injection: String concatenation in queries
- XSS: Unescaped user input in HTML output
- Command injection: User input in shell commands
- LDAP injection, NoSQL injection, template injection

**What to look for in code:** Parameterized queries, input sanitization, output encoding, use of ORM.

## A04:2021 — Insecure Design
- Missing rate limiting on authentication endpoints
- No account lockout after failed attempts
- Missing CAPTCHA on forms
- Trusting client-side validation only

**What to look for in code:** Rate limiters, server-side validation, threat modeling coverage.

## A05:2021 — Security Misconfiguration
- Default credentials left in place
- Unnecessary features enabled (debug mode in production)
- Missing security headers (CSP, X-Frame-Options, HSTS)
- Verbose error messages exposing stack traces

**What to look for in code:** Production configuration files, error handling, security headers middleware.

## A06:2021 — Vulnerable and Outdated Components
- Dependencies with known CVEs
- Outdated framework versions
- Unused dependencies increasing attack surface

**What to look for in code:** Dependency versions in package.json/requirements.txt/Cargo.toml, lock files.

## A07:2021 — Identification and Authentication Failures
- Weak password policies
- Missing multi-factor authentication for sensitive operations
- Session fixation or session IDs in URLs
- Credentials stored in plaintext

**What to look for in code:** Session management, password policies, MFA implementation, credential storage.

## A08:2021 — Software and Data Integrity Failures
- Deserialization of untrusted data (pickle, Java serialization)
- CI/CD pipeline without integrity verification
- Auto-update without signature verification
- Using CDN resources without SRI (Subresource Integrity)

**What to look for in code:** Deserialization calls, CI/CD configurations, integrity checks.

## A09:2021 — Security Logging and Monitoring Failures
- Login attempts not logged
- Errors and warnings not logged
- Logs not monitored for suspicious activity
- Logs containing sensitive data (passwords, tokens)

**What to look for in code:** Logging calls on auth events, error handlers, log content filtering.

## A10:2021 — Server-Side Request Forgery (SSRF)
- Fetching URLs from user input without validation
- Missing allowlist for external service calls
- Internal service endpoints accessible via URL manipulation

**What to look for in code:** URL fetching functions, input validation on URLs, network segmentation.
