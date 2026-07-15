# Release notes

Per-release highlights for sparkfit. The authoritative, dated list of every change
is in the [CHANGELOG](../../CHANGELOG.md); these files are the longer-form notes
used for each GitHub release.

- [v0.4.0](v0.4.0.md)
- [v0.3.0](v0.3.0.md)
- [v0.2.0](v0.2.0.md)
- [v0.1.0](v0.1.0.md)

To cut a release, add `vX.Y.Z.md` here (same structure as the previous ones),
move the `Unreleased` section of the CHANGELOG under a dated `X.Y.Z` heading, then
publish with `gh release create vX.Y.Z --notes-file docs/releases/vX.Y.Z.md`.
