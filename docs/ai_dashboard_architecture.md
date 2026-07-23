# TEN AI Dashboard architecture

## Purpose

The dashboard presents the Phase 1–6 analytical pipeline as one decision story. It is a read-only interface: backend records remain authoritative, the UI does not rescore evidence, and no broker execution capability is exposed.

## Primary information architecture

The primary navigation is intentionally limited to:

1. **Overview** — current final action, decision pipeline, intelligence summaries, guardrails, monitoring, validation, and health.
2. **Signals** — the current analytical publication and its lifecycle.
3. **Performance** — measured proposal and publication outcomes.
4. **Calibration** — probability reliability and quantitative forecast diagnostics.
5. **System** — readiness, dependencies, data freshness, operating profile, and pipeline state.

Legacy diagnostic pages remain addressable under legacy/direct routes for operational continuity, but they are not shown in primary navigation or composed into the main dashboard.

## Frontend boundaries

- `useAIDashboardData` fetches the four existing read-only analytical resources in parallel every five seconds.
- A failed refresh preserves the last successfully received value and exposes the failure beside a freshness indicator.
- `aiDashboard.ts` only converts backend enums and fields into presentation language, tone, and pipeline status. It contains no trading thresholds or scoring.
- Components receive typed API records and represent absent values as unavailable, never as zero.
- The final-decision hero leads the page; supporting cards follow the analytical pipeline rather than engine ownership.

## Runtime safety

The backend latest-reasoning response now includes observational runtime metadata: the active AI feature flags, derived operating profile, analytical-only status, and the explicit absence of broker execution. This is reporting only. Defaults and decision behavior are unchanged.

No schema migration is required. The redesign reads the current Phase 1–6 records and adds no tables, columns, writes, or state transitions.

## Responsive and accessibility behavior

Desktop uses a compact top navigation and a two-column decision workspace. Tablet collapses the decision pipeline into stacked stages. Mobile uses one-column cards, horizontally scrollable navigation, full-width actions, visible keyboard focus, semantic headings, and reduced-motion support.
