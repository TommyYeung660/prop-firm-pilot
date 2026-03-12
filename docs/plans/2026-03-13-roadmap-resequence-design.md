# v1.4.x Roadmap Resequencing Design

**Date:** 2026-03-13

## Context

The current `v1.4.0` roadmap still presents `v1.4.5` as the next broad
"decision quality + exit quality" release. That no longer matches the
execution priority. Tactical entry and tactical exit stabilization now
take precedence, and the broader unfinished `v1.4.5` scope needs to move
back to `v1.4.8`.

## Decision

Re-sequence the `v1.4.x` roadmap as follows:

- `v1.4.6`: tactical entry fixes and optimization
- `v1.4.7`: tactical exit fixes and optimization
- `v1.4.8`: deferred broader `v1.4.5` scope

The broader deferred scope includes:

- lesson expansion across more agent nodes
- memory unification
- dynamic exit baseline work
- correlation and portfolio guard

## Documentation Changes

Modify the formal roadmap and changelog only:

- `docs/PropFirmPilot_v1.4.0_road_map.md`
- `docs/PropFirmPilot_changelog.md`

Also record this design and an implementation plan in `docs/plans/`.

## Writing Direction

- Keep the roadmap focused on sequencing and scope boundaries.
- Make `v1.4.6` and `v1.4.7` clearly tactical.
- Make `v1.4.8` explicitly the deferred broader scope that used to live in
  `v1.4.5`.
- Preserve released history in the changelog by adding an `Unreleased`
  planning section instead of pretending future versions are shipped.
