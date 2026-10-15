# Public release checklist

Completed locally:

- [x] Published-claim data, configurations, source, and figures collected.
- [x] Numerical and Maxwell audits pass from the repository root.
- [x] Conference manuscript builds in five pages.
- [x] Python dependencies are pinned in `requirements.txt` and
  `environment.yml`; the lighter audit set is in `requirements-audit.txt`.
- [x] GitHub Actions runs both evidence audits.
- [x] Citation and Zenodo metadata are present.
- [x] No API tokens, passwords, or private keys were detected.
- [x] Former hard-coded imports in the retained neural audit scripts were made
  repository-relative, and the missing first-order helper was included.

Required before public release:

- [x] Confirm the correct GitHub owner and repository name.
- [x] Select an explicit software/data license (MIT).
- [ ] Create and push the public repository.
- [ ] Replace the paper's pending Code Availability statement with the public
  repository URL and rebuild the five-page PDF.
- [ ] Create GitHub release `v1.0.0`.
- [ ] Connect the repository to Zenodo, archive `v1.0.0`, and add the assigned
  DOI to `CITATION.cff`, `.zenodo.json`, the README, and the paper if timing
  permits.
- [ ] Rerun both audits and verify the release archive from a clean extraction.

The GitHub display name “Addy Tong Lok” and manuscript name “Le Dong” identify
the same author; this alias is disclosed in the README and citation metadata.
