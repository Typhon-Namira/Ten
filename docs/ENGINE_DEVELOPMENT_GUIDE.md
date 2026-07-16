# Engine development guide

An engine is a versioned implementation plus a registration hook. Runtime code must never instantiate an engine directly.

## Add a new engine

1. Create `backend/app/engines/<name>_engine/` with `models.py`, `config.py`, interface, implementation, `__init__.py`, and `registration.py`.
2. Define stable input/output models. Provider DTOs cannot cross the engine boundary.
3. Add `configs/<config_key>.yaml`; runtime values belong in YAML, not composition code.
4. In `registration.py`, call `EngineFactory.register` with `EngineMetadata`, a builder, and an async executor.
5. The executor reads only `PipelineExecutionContext`; it must not import or invoke another engine.
6. Return `EngineExecutionResult` with an output, serializable feature mapping, namespace, typed completion event, and optional confidence factor.
7. Add the version to `configs/engine_registry.yaml` and its execution position to `configs/pipeline.yaml`.
8. Add a feature flag when rollout must be independently controlled.
9. Add unit, configuration-validation, registration, empty-input, and event/feature integration tests.
10. Document methodology, data provenance, and limitations.

`EngineLoader` discovers registration modules using package inspection. No central engine list should be edited.

## Upgrade an engine without breaking consumers

Register the new semantic version alongside the old version. Keep the compatibility version unchanged when inputs, output schema, feature namespace, and executor contract remain compatible. Increment compatibility version for a breaking contract and migrate consumers explicitly. Select the promoted version in `engine_registry.yaml` only after out-of-sample and serialization tests pass.

Never edit or delete deployed prompt versions or engine metadata needed for replay/audit history.

## Feature and event rules

- Every successful stage writes an immutable `FeatureRecord` with correlation ID and engine versions.
- Feature namespaces are stable API contracts.
- AI receives `FeatureSnapshot`; raw engine objects and charts are prohibited.
- Completion is published through a typed event.
- Subscriber failures propagate in the in-process adapter so a run cannot silently appear complete.
- Engines never call other engines. Dependency ordering is validated by `PipelineManager` before execution.

## Deterministic confidence

Executors may expose a bounded numeric confidence factor. `ConfidenceCalculator` applies YAML weights. AI contributes only capped `quality_score / 100`; the deprecated LLM confidence field is ignored by the production pipeline. Signal explanations preserve the resulting breakdown.

## Prompt-driven AI upgrades

1. Add an immutable prompt such as `signal_analysis_v2.txt`.
2. Keep the response provider-neutral: direction, quality score, risks, and reasoning—never confidence.
3. Add contract tests for malformed responses and frozen feature snapshots.
4. Select the prompt in `configs/ai.yaml`.
5. Replay both versions over the same snapshot set before promotion.

## Quality gate

An engine is ready only when typed configuration, semantic metadata, deterministic fixture output, serialization, event publication, feature storage, disabled-state behavior, and pipeline compatibility pass.
