# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security reports. Use GitHub's
[private vulnerability reporting](https://github.com/lordx64/phantom-kv/security/advisories/new)
(Report a vulnerability) and include:

- the phantom-kv version or commit,
- the model id and platform,
- a minimal repro (commands + input), and
- the expected vs. actual behavior.

We aim to acknowledge reports within 7 days.

## Trust model notes (read before shipping grafts)

- `phantom.bin` grafts are **executable context**: a graft file is loaded
  into the target process's KV cache and influences its output for every
  request it is spliced into. Treat graft files like binaries: verify the
  payload `sha256` in the sidecar against a trusted source.
- The loader validates the container contract (layer coverage, dimensions,
  payload hash) but **cannot** validate intent — only install grafts you
  trained or obtained from a party you trust.
- phantom-kv performs **no network access at runtime**; if you observe any,
  that's a reportable bug.
- Graft files are model- and revision-specific: a graft built for one model
  must be refused by another (model id is recorded in the sidecar).
