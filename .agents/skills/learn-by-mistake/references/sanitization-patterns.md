# Sanitization Patterns

> Regex patterns for scrubbing secrets and PII from error context before storage.
> Used by the secret sanitizer to ensure no sensitive data leaks into lesson files.

---

## AWS Keys

- **Regex:** `(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}`
- **Replacement:** `<REDACTED_AWS_KEY>`
- **Example:** `AKIAIOSFODNN7EXAMPLE` -> `<REDACTED_AWS_KEY>`

## AWS Secret Keys

- **Regex:** `(?i)(?:aws_secret_access_key|aws_secret)\s*[=:]\s*['"]?([A-Za-z0-9/+=]{40})['"]?`
- **Replacement:** `aws_secret_access_key=<REDACTED_AWS_SECRET>`
- **Example:** `aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` -> `aws_secret_access_key=<REDACTED_AWS_SECRET>`

## GitHub Tokens

- **Regex:** `ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82}|gho_[A-Za-z0-9]{36}|ghs_[A-Za-z0-9]{36}|ghr_[A-Za-z0-9]{36}`
- **Replacement:** `<REDACTED_GITHUB_TOKEN>`
- **Example:** `ghp_ABCDEFghijklmnopqrstuvwxyz0123456789` -> `<REDACTED_GITHUB_TOKEN>`

## Generic API Keys

- **Regex:** `(?i)(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*[=:]\s*['"]?([A-Za-z0-9_\-/.+=]{16,})['"]?`
- **Replacement:** `<REDACTED_API_KEY>`
- **Example:** `api_key=sk-proj-abc123def456ghi789jkl012mno345` -> `api_key=<REDACTED_API_KEY>`

## OpenAI / Anthropic / Stripe Keys

- **Regex:** `sk-[A-Za-z0-9_\-]{20,}|pk_(live|test)_[A-Za-z0-9]{24,}|sk_(live|test)_[A-Za-z0-9]{24,}|sk-ant-[A-Za-z0-9_\-]{20,}`
- **Replacement:** `<REDACTED_API_KEY>`
- **Example:** `sk-ant-api03-xxxxxxxxxxxx` -> `<REDACTED_API_KEY>`

## JWT Tokens

- **Regex:** `eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_\-+/=]{10,}`
- **Replacement:** `<REDACTED_JWT>`
- **Example:** `eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.aBcDeFgHiJkLmNoPqRsT` -> `<REDACTED_JWT>`

## Connection Strings

- **Regex:** `(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|rediss|amqp|amqps):\/\/[^\s'"]+`
- **Replacement:** `<REDACTED_CONNECTION_STRING>`
- **Example:** `postgres://admin:s3cret@db.host.com:5432/mydb` -> `<REDACTED_CONNECTION_STRING>`

## Bearer Tokens

- **Regex:** `(?i)bearer\s+[A-Za-z0-9_\-/.+=]{20,}`
- **Replacement:** `Bearer <REDACTED_TOKEN>`
- **Example:** `Authorization: Bearer eyJhbGciOi...` -> `Authorization: Bearer <REDACTED_TOKEN>`

## High-Entropy Strings (Base64 secrets)

- **Regex:** `(?i)(?:secret|password|passwd|token|credential)\s*[=:]\s*['"]?([A-Za-z0-9+/=]{32,})['"]?`
- **Replacement:** `<REDACTED_SECRET>`
- **Example:** `secret=dGhpcyBpcyBhIHNlY3JldCBrZXkgdmFsdWU=` -> `secret=<REDACTED_SECRET>`

## Passwords in URLs

- **Regex:** `://([^:]+):([^@]{3,})@`
- **Replacement:** `://$1:<REDACTED_PASSWORD>@`
- **Example:** `https://user:p4ssw0rd@api.example.com` -> `https://user:<REDACTED_PASSWORD>@api.example.com`

## Absolute File Paths (Home Directories)

- **Regex:** `/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/|C:\\\\Users\\\\[A-Za-z0-9._-]+\\\\`
- **Replacement:** `<PROJECT_PATH>/`
- **Example:** `/home/marche/projects/myapp/src/index.ts` -> `<PROJECT_PATH>/projects/myapp/src/index.ts`

## Email Addresses

- **Regex:** `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`
- **Replacement:** `<REDACTED_EMAIL>`
- **Example:** `admin@company.internal` -> `<REDACTED_EMAIL>`

## Internal IP Addresses

- **Regex:** `(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})`
- **Replacement:** `<REDACTED_INTERNAL_IP>`
- **Example:** `192.168.1.42` -> `<REDACTED_INTERNAL_IP>`

## Webhook URLs

- **Regex:** `https?://(?:hooks\.slack\.com|discord(?:app)?\.com/api/webhooks|hooks\.zapier\.com|webhook\.site)/[^\s'"]+`
- **Replacement:** `<REDACTED_WEBHOOK_URL>`
- **Example:** `https://hooks.slack.com/services/T00/B00/xxxx` -> `<REDACTED_WEBHOOK_URL>`

## .env File Content Lines

- **Regex:** `^[A-Z][A-Z0-9_]*=.{8,}$`
- **Replacement:** `<REDACTED_ENV_LINE>`
- **Example:** `DATABASE_URL=postgres://localhost/mydb` -> `<REDACTED_ENV_LINE>`
- **Note:** Apply only when source context is identified as `.env` file content. The broad pattern requires contextual gating to avoid false positives.

## Private Keys (PEM blocks)

- **Regex:** `-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----`
- **Replacement:** `<REDACTED_PRIVATE_KEY>`
- **Example:** `-----BEGIN RSA PRIVATE KEY-----\nMIIEow...` -> `<REDACTED_PRIVATE_KEY>`

---

## Application Order

Apply patterns in this order to avoid partial matches:

1. Private keys (multiline, match first)
2. JWT tokens (long, could overlap with generic base64)
3. Connection strings (contain passwords)
4. Passwords in URLs (subset of connection strings)
5. AWS keys
6. AWS secret keys
7. GitHub tokens
8. OpenAI / Anthropic / Stripe keys
9. Generic API keys
10. Bearer tokens
11. High-entropy strings
12. Webhook URLs
13. Email addresses
14. Internal IPs
15. Absolute file paths
16. .env file content (contextual, apply last)
