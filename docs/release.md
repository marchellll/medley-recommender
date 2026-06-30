# Release

```sh
cargo install cargo-release
```

## Dry run (preview, no changes)

```sh
cargo release patch --no-publish
```

## Execute

```sh
cargo release patch --no-publish --execute
```

Bumps `[workspace.package] version` in root `Cargo.toml`, commits, creates annotated tag (`v*`). Sub-crates use `version.workspace = true` — one bump propagates.

Flag | Meaning
-----|--------
`patch` | bump patch segment (`0.2.2` → `0.2.3`). Also `minor`, `major`.
`--no-publish` | skip `cargo publish`. Use without when ready to ship.
`--execute` | actually apply changes. Omit for dry run.
