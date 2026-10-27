# Open issue: freshly published versions of well-known packages

Status: **decided 2026-10-25: option D.** Seven of the nine have publish provenance
from their official repositories and are kept; importlib-metadata and zipp have none
and are held back to versions older than 14 days. See decision P2 in
[package-security-audit.md](package-security-audit.md). Related: [package-security-audit.md](package-security-audit.md).

## The question

Decision P1 held `httpx2` and `httpcore2` back from a four-day-old release to one a
month old. Nine other packages in the same set are also younger than 14 days:

| Package | Version | Uploaded | Age on 2026-10-25 |
|---|---|---|---|
| diffusers | 0.42.0 | 2026-10-23 | 1 day |
| huggingface-hub | 2.3.0 | 2026-10-23 | 1 day |
| certifi | 2026.10.23 | 2026-10-23 | 1 day |
| importlib-metadata | 10.1.0 | 2026-10-24 | 0 days |
| zipp | 4.1.2 | 2026-10-21 | 3 days |
| filelock | 4.2.4 | 2026-10-19 | 5 days |
| transformers | 5.20.2 | 2026-10-19 | 5 days |
| pillow | 13.0.0 | 2026-10-16 | 8 days |
| truststore | 0.10.5 | 2026-10-12 | 12 days |

All nine passed every check in the audit: official hash, RECORD integrity, no
unexpected `.pth` files or executables, a clean antivirus scan. Should they be held
to the same rule?

## Two different risks

The first recommendation was to keep these nine because they are well-known packages
with no grounds for suspicion. That reasoning covered only one of two risks.

**1. Impostor packages (typosquatting).** A look-alike name (`reqeusts` for
`requests`) carrying malware. Familiarity *does* rule this out: all nine are
long-established projects maintained by known organizations. `httpx2` was examined
precisely because its name was unfamiliar.

**2. A compromised release of a genuine package.** A maintainer account or a
project's release pipeline is taken over, and a malicious version is published
under the real name. Familiarity gives *no* protection here; popular packages are
the more attractive targets. In December 2024 two official releases of
`ultralytics`, a widely used AI package, shipped with a cryptocurrency miner after its
release pipeline was attacked; they were caught and pulled within days.

A waiting period exists for risk 2: most compromised releases are discovered and
removed within days, so a version that has been public for two weeks has survived
that window. Risk 2 applies to `diffusers` or `transformers` exactly as it did to
`httpx2`. Measured by the reason P1 was taken, the nine are in the same position.

## Ways to reduce risk 2

| Option | Effect | Cost |
|---|---|---|
| Wait until each is 14 days old | Strongest | `diffusers` and `huggingface-hub` reach 14 days on 2026-11-06 |
| Check PyPI publish provenance (attestations) | Proves an upload came from the project's own CI (Trusted Publishing), which rules out a stolen upload token | Does not cover a compromised pipeline: in the `ultralytics` case the malicious build came from the project's own CI, and provenance would have looked valid |
| Use older versions (14+ days) | Removes the exposure | `diffusers`: the engine's image-model code was checked against 0.42.0's pipeline source, and earlier versions count sampling steps differently, so that check must be redone; version compatibility across the set must be re-resolved |
| Keep as is | No work | Accepts risk 2 for nine packages, inconsistent with P1 |

## Options under consideration

- **A.** Hold all nine to the 14-day rule (older versions, re-check the image-model code).
- **B.** Keep all nine.
- **C.** Hold back the seven lightweight ones, keep `diffusers` and `transformers`.
- **D.** Check publish provenance for the nine first; keep those proven to come from
  the official repository's CI, hold back the rest. (Proposed next step.)

The decision, when made, will be recorded here and in
[package-security-audit.md](package-security-audit.md).
