# Project rules

- Maintain this as an independent detector; exchange versioned files with motion-evidence.
- Original-session sources and data are read-only. Never run their build or analysis scripts in place.
- Use the externally configured work root for environments, builds, caches, tests and run artifacts.
- Commit only generic source, schemas, synthetic tests and documentation. Real media, derived records, private paths and credentials stay outside Git.
- Detector output is measurement, not semantic truth or a person-presence retention rule.
- Preserve real stream/PTS identity, model ordering, independent low/high masks and exact 4-connectivity.
- Update product, architecture, core design and validation docs with delivered behavior.
- Follow docs/code-style.md; run Ruff checks/format validation and native formatting/build checks for touched languages.
- After checks and staged-content privacy review, commit and push each delivered update. Do not change global Git configuration.
