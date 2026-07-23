"""Persistence boundary for guardrail evaluations, final actions, and validation artifacts."""

from __future__ import annotations

import asyncio
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    AIPerformanceReportRecord,
    AIProductionReadinessReportRecord,
    DetailedSignalOutcomeRecord,
    FinalSystemActionRecord,
    GuardrailEvaluationRecord,
    HardGateVersionRecord,
    LLMUsageMetricRecord,
    PublishedAnalyticalSignalRecord,
)
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import (
    DetailedSignalOutcome,
    FinalSystemAction,
    GateEvaluation,
    HardGateDefinition,
    LLMUsageMetric,
    PerformanceReport,
    ProductionReadinessReport,
    PublishedAnalyticalSignal,
)


class FinalDecisionRepository(Protocol):
    async def save_gate_definition(self, value: HardGateDefinition, registry_version: str) -> None: ...
    async def save_action(self, value: FinalSystemAction) -> FinalSystemAction: ...
    async def save_evaluation(self, value: GateEvaluation) -> GateEvaluation: ...
    async def save_publication(self, value: PublishedAnalyticalSignal) -> PublishedAnalyticalSignal: ...
    async def publication_for_signal(self, signal_id: object) -> PublishedAnalyticalSignal | None: ...
    async def latest_action(self, signal_id: object | None = None) -> FinalSystemAction | None: ...
    async def action_for_state(self, market_state_id: object) -> FinalSystemAction | None: ...
    async def action_history(self, signal_id: object) -> tuple[FinalSystemAction, ...]: ...
    async def save_usage(self, value: LLMUsageMetric) -> LLMUsageMetric: ...
    async def usage_for_date(self, usage_date: str) -> tuple[LLMUsageMetric, ...]: ...
    async def save_outcome(self, value: DetailedSignalOutcome) -> DetailedSignalOutcome: ...
    async def outcomes(self) -> tuple[DetailedSignalOutcome, ...]: ...
    async def outcome_for_signal(self, signal_id: object) -> DetailedSignalOutcome | None: ...
    async def save_performance_report(self, value: PerformanceReport) -> PerformanceReport: ...
    async def latest_performance_report(self) -> PerformanceReport | None: ...
    async def save_readiness_report(self, value: ProductionReadinessReport) -> ProductionReadinessReport: ...
    async def latest_readiness_report(self) -> ProductionReadinessReport | None: ...


class InMemoryFinalDecisionRepository:
    def __init__(self) -> None:
        self.gate_definitions: dict[tuple[str, str], HardGateDefinition] = {}
        self.actions: dict[object, FinalSystemAction] = {}
        self.evaluations: dict[object, GateEvaluation] = {}
        self.publications: dict[object, PublishedAnalyticalSignal] = {}
        self.usage: dict[object, LLMUsageMetric] = {}
        self.signal_outcomes: dict[object, DetailedSignalOutcome] = {}
        self.performance_reports: dict[object, PerformanceReport] = {}
        self.readiness_reports: dict[object, ProductionReadinessReport] = {}
        self._lock = asyncio.Lock()

    async def save_gate_definition(self, value: HardGateDefinition, registry_version: str) -> None:
        async with self._lock:
            self.gate_definitions[(value.gate_id, value.gate_version)] = value

    async def save_action(self, value: FinalSystemAction) -> FinalSystemAction:
        async with self._lock:
            self.actions[value.final_action_id] = value
        return value

    async def save_evaluation(self, value: GateEvaluation) -> GateEvaluation:
        async with self._lock:
            boundary = (value.final_action_id, value.gate_id)
            existing = self.evaluations.get(boundary)
            if existing is not None:
                return existing
            self.evaluations[boundary] = value
            return value

    async def save_publication(self, value: PublishedAnalyticalSignal) -> PublishedAnalyticalSignal:
        async with self._lock:
            existing = next((item for item in self.publications.values() if item.signal_id == value.signal_id), None)
            if existing is not None:
                return existing
            self.publications[value.publication_id] = value
        return value

    async def publication_for_signal(self, signal_id: object) -> PublishedAnalyticalSignal | None:
        return next((item for item in self.publications.values() if item.signal_id == signal_id), None)

    async def latest_action(self, signal_id: object | None = None) -> FinalSystemAction | None:
        values = list(self.actions.values())
        if signal_id is not None:
            values = [item for item in values if item.managed_signal_id == signal_id]
        return max(values, key=lambda item: item.created_at, default=None)

    async def action_for_state(self, market_state_id: object) -> FinalSystemAction | None:
        values = [item for item in self.actions.values() if item.market_state_id == market_state_id]
        return max(values, key=lambda item: (item.created_at, str(item.final_action_id)), default=None)

    async def action_history(self, signal_id: object) -> tuple[FinalSystemAction, ...]:
        values = [item for item in self.actions.values() if item.managed_signal_id == signal_id]
        return tuple(sorted(values, key=lambda item: item.created_at))

    async def save_usage(self, value: LLMUsageMetric) -> LLMUsageMetric:
        async with self._lock:
            self.usage[value.metric_id] = value
        return value

    async def usage_for_date(self, usage_date: str) -> tuple[LLMUsageMetric, ...]:
        return tuple(item for item in self.usage.values() if item.usage_date == usage_date)

    async def save_outcome(self, value: DetailedSignalOutcome) -> DetailedSignalOutcome:
        async with self._lock:
            self.signal_outcomes[value.signal_id] = value
        return value

    async def outcomes(self) -> tuple[DetailedSignalOutcome, ...]:
        return tuple(self.signal_outcomes.values())

    async def outcome_for_signal(self, signal_id: object) -> DetailedSignalOutcome | None:
        return self.signal_outcomes.get(signal_id)

    async def save_performance_report(self, value: PerformanceReport) -> PerformanceReport:
        self.performance_reports[value.report_id] = value
        return value

    async def latest_performance_report(self) -> PerformanceReport | None:
        return max(self.performance_reports.values(), key=lambda item: item.generated_at, default=None)

    async def save_readiness_report(self, value: ProductionReadinessReport) -> ProductionReadinessReport:
        self.readiness_reports[value.report_id] = value
        return value

    async def latest_readiness_report(self) -> ProductionReadinessReport | None:
        return max(self.readiness_reports.values(), key=lambda item: item.generated_at, default=None)


class SqlAlchemyFinalDecisionRepository(ScopedSessionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        super().__init__(session_factory)

    @scoped_session
    async def save_gate_definition(self, value: HardGateDefinition, registry_version: str) -> None:
        await self.session.execute(
            insert(HardGateVersionRecord)
            .values(
                gate_id=value.gate_id,
                gate_version=value.gate_version,
                registry_version=registry_version,
                category=value.category,
                payload=value.model_dump(mode="json"),
            )
            .on_conflict_do_nothing(index_elements=["gate_id", "gate_version"])
        )
        await self.session.commit()

    @scoped_session
    async def save_action(self, value: FinalSystemAction) -> FinalSystemAction:
        await self.session.execute(
            insert(FinalSystemActionRecord)
            .values(
                final_action_id=value.final_action_id,
                ai_proposal_id=value.ai_proposal_id,
                managed_signal_id=value.managed_signal_id,
                market_state_id=value.market_state_id,
                quantitative_forecast_id=value.quantitative_forecast_id,
                ai_forecast_id=value.ai_forecast_id,
                action=value.action.value,
                approval_state=value.approval_state.value,
                publication_state=value.publication_state.value,
                payload=value.model_dump(mode="json"),
                created_at=value.created_at,
            )
            .on_conflict_do_update(
                index_elements=["final_action_id"],
                set_={
                    "action": value.action.value,
                    "approval_state": value.approval_state.value,
                    "publication_state": value.publication_state.value,
                    "payload": value.model_dump(mode="json"),
                },
            )
        )
        await self.session.commit()
        return value

    @scoped_session
    async def save_evaluation(self, value: GateEvaluation) -> GateEvaluation:
        await self.session.execute(
            insert(GuardrailEvaluationRecord)
            .values(
                evaluation_id=value.evaluation_id,
                final_action_id=value.final_action_id,
                gate_id=value.gate_id,
                gate_version=value.gate_version,
                status=value.status.value,
                payload=value.model_dump(mode="json"),
                evaluated_at=value.evaluated_at,
            )
            .on_conflict_do_nothing(index_elements=["final_action_id", "gate_id"])
        )
        await self.session.commit()
        return value

    @scoped_session
    async def save_publication(self, value: PublishedAnalyticalSignal) -> PublishedAnalyticalSignal:
        await self.session.execute(
            insert(PublishedAnalyticalSignalRecord)
            .values(
                publication_id=value.publication_id,
                signal_id=value.signal_id,
                final_action_id=value.final_action_id,
                proposal_id=value.proposal_id,
                instrument=value.instrument,
                direction=value.direction.value,
                setup_family=value.setup_family,
                lifecycle_state=value.lifecycle_state.value,
                payload=value.model_dump(mode="json"),
                published_at=value.published_at,
            )
            .on_conflict_do_nothing(index_elements=["signal_id"])
        )
        await self.session.commit()
        return await self.publication_for_signal(value.signal_id) or value

    @scoped_session
    async def publication_for_signal(self, signal_id: object) -> PublishedAnalyticalSignal | None:
        record = (await self.session.scalars(select(PublishedAnalyticalSignalRecord).where(PublishedAnalyticalSignalRecord.signal_id == signal_id).limit(1))).first()
        return PublishedAnalyticalSignal.model_validate(record.payload) if record else None

    @scoped_session
    async def latest_action(self, signal_id: object | None = None) -> FinalSystemAction | None:
        query = select(FinalSystemActionRecord)
        if signal_id is not None:
            query = query.where(FinalSystemActionRecord.managed_signal_id == signal_id)
        record = (await self.session.scalars(query.order_by(FinalSystemActionRecord.created_at.desc()).limit(1))).first()
        return FinalSystemAction.model_validate(record.payload) if record else None

    @scoped_session
    async def action_for_state(self, market_state_id: object) -> FinalSystemAction | None:
        query = (
            select(FinalSystemActionRecord)
            .where(FinalSystemActionRecord.market_state_id == market_state_id)
            .order_by(FinalSystemActionRecord.created_at.desc(), FinalSystemActionRecord.final_action_id.desc())
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        return FinalSystemAction.model_validate(record.payload) if record else None

    @scoped_session
    async def action_history(self, signal_id: object) -> tuple[FinalSystemAction, ...]:
        records = (await self.session.scalars(select(FinalSystemActionRecord).where(FinalSystemActionRecord.managed_signal_id == signal_id).order_by(FinalSystemActionRecord.created_at))).all()
        return tuple(FinalSystemAction.model_validate(record.payload) for record in records)

    @scoped_session
    async def save_usage(self, value: LLMUsageMetric) -> LLMUsageMetric:
        await self.session.execute(
            insert(LLMUsageMetricRecord)
            .values(
                metric_id=value.metric_id,
                usage_date=value.usage_date,
                request_hash=value.request_hash,
                market_state_hash=value.market_state_hash,
                model_identifier=value.model_identifier,
                success=value.success,
                payload=value.model_dump(mode="json"),
                created_at=value.created_at,
            )
            .on_conflict_do_nothing(index_elements=["metric_id"])
        )
        await self.session.commit()
        return value

    @scoped_session
    async def usage_for_date(self, usage_date: str) -> tuple[LLMUsageMetric, ...]:
        records = (await self.session.scalars(select(LLMUsageMetricRecord).where(LLMUsageMetricRecord.usage_date == usage_date))).all()
        return tuple(LLMUsageMetric.model_validate(record.payload) for record in records)

    @scoped_session
    async def save_outcome(self, value: DetailedSignalOutcome) -> DetailedSignalOutcome:
        await self.session.execute(
            insert(DetailedSignalOutcomeRecord)
            .values(outcome_id=value.outcome_id, signal_id=value.signal_id, status=value.status, payload=value.model_dump(mode="json"), evaluated_at=value.evaluated_at)
            .on_conflict_do_update(index_elements=["signal_id"], set_={"status": value.status, "payload": value.model_dump(mode="json"), "evaluated_at": value.evaluated_at})
        )
        await self.session.commit()
        return value

    @scoped_session
    async def outcomes(self) -> tuple[DetailedSignalOutcome, ...]:
        records = (await self.session.scalars(select(DetailedSignalOutcomeRecord))).all()
        return tuple(DetailedSignalOutcome.model_validate(record.payload) for record in records)

    @scoped_session
    async def outcome_for_signal(self, signal_id: object) -> DetailedSignalOutcome | None:
        record = (
            await self.session.scalars(
                select(DetailedSignalOutcomeRecord)
                .where(DetailedSignalOutcomeRecord.signal_id == signal_id)
                .limit(1)
            )
        ).first()
        return DetailedSignalOutcome.model_validate(record.payload) if record else None

    async def _save_report(self, record_type: type, values: dict[str, object], key: str) -> None:
        await self.session.execute(insert(record_type).values(**values).on_conflict_do_nothing(index_elements=[key]))
        await self.session.commit()

    @scoped_session
    async def save_performance_report(self, value: PerformanceReport) -> PerformanceReport:
        await self._save_report(AIPerformanceReportRecord, {"report_id": value.report_id, "sample_count": value.sample_count, "payload": value.model_dump(mode="json"), "generated_at": value.generated_at}, "report_id")
        return value

    @scoped_session
    async def latest_performance_report(self) -> PerformanceReport | None:
        record = (await self.session.scalars(select(AIPerformanceReportRecord).order_by(AIPerformanceReportRecord.generated_at.desc()).limit(1))).first()
        return PerformanceReport.model_validate(record.payload) if record else None

    @scoped_session
    async def save_readiness_report(self, value: ProductionReadinessReport) -> ProductionReadinessReport:
        await self._save_report(AIProductionReadinessReportRecord, {"report_id": value.report_id, "status": value.status, "sample_count": value.sample_count, "payload": value.model_dump(mode="json"), "generated_at": value.generated_at}, "report_id")
        return value

    @scoped_session
    async def latest_readiness_report(self) -> ProductionReadinessReport | None:
        record = (await self.session.scalars(select(AIProductionReadinessReportRecord).order_by(AIProductionReadinessReportRecord.generated_at.desc()).limit(1))).first()
        return ProductionReadinessReport.model_validate(record.payload) if record else None
