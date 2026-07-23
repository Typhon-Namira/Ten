"""Persistence ports for quantitative shadow forecasting."""

from __future__ import annotations

import asyncio
from typing import Protocol
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.storage.models import (
    QuantCalibrationReportRecord,
    QuantCalibrationBucketRecord,
    QuantFeatureReferenceRecord,
    QuantFeatureVectorRecord,
    QuantForecastOutcomeRecord,
    QuantForecastRequestRecord,
    QuantForecastResultRecord,
    QuantForecastHorizonRecord,
    QuantModelMetadataRecord,
)
from backend.app.storage.scoped_session import ScopedSessionRepository, scoped_session

from .models import (
    CalibrationReport,
    ForecastOutcome,
    ModelMetadata,
    QuantFeatureVector,
    QuantForecastRequest,
    QuantForecastResult,
)


class QuantForecastRepository(Protocol):
    async def save_model_metadata(self, value: ModelMetadata) -> ModelMetadata: ...
    async def save_request(self, value: QuantForecastRequest) -> QuantForecastRequest: ...
    async def save_features(self, value: QuantFeatureVector) -> QuantFeatureVector: ...
    async def save_result(self, value: QuantForecastResult) -> QuantForecastResult: ...
    async def latest_result(self, instrument: str) -> QuantForecastResult | None: ...
    async def save_outcome(self, value: ForecastOutcome) -> ForecastOutcome: ...
    async def outcomes_for_result(self, result_id: UUID) -> tuple[ForecastOutcome, ...]: ...
    async def save_calibration(self, value: CalibrationReport) -> CalibrationReport: ...
    async def latest_calibration(self, model_name: str) -> CalibrationReport | None: ...


class InMemoryQuantForecastRepository:
    def __init__(self) -> None:
        self.models: dict[tuple[str, str], ModelMetadata] = {}
        self.requests: dict[object, QuantForecastRequest] = {}
        self.features: dict[object, QuantFeatureVector] = {}
        self.results: dict[object, QuantForecastResult] = {}
        self.outcomes: dict[object, ForecastOutcome] = {}
        self.calibrations: dict[object, CalibrationReport] = {}
        self._lock = asyncio.Lock()

    async def save_model_metadata(self, value: ModelMetadata) -> ModelMetadata:
        async with self._lock:
            self.models[(value.model_name, value.model_version)] = value
        return value

    async def save_request(self, value: QuantForecastRequest) -> QuantForecastRequest:
        async with self._lock:
            self.requests[value.request_id] = value
        return value

    async def save_features(self, value: QuantFeatureVector) -> QuantFeatureVector:
        async with self._lock:
            self.features[value.vector_id] = value
        return value

    async def save_result(self, value: QuantForecastResult) -> QuantForecastResult:
        async with self._lock:
            self.results[value.result_id] = value
        return value

    async def latest_result(self, instrument: str) -> QuantForecastResult | None:
        async with self._lock:
            values = [value for value in self.results.values() if value.instrument == instrument]
        return max(values, key=lambda value: (value.point_in_time, str(value.result_id)), default=None)

    async def save_outcome(self, value: ForecastOutcome) -> ForecastOutcome:
        async with self._lock:
            self.outcomes[value.outcome_id] = value
        return value

    async def outcomes_for_result(self, result_id: UUID) -> tuple[ForecastOutcome, ...]:
        async with self._lock:
            values = [value for value in self.outcomes.values() if value.forecast_result_id == result_id]
        return tuple(sorted(values, key=lambda value: value.horizon_id))

    async def save_calibration(self, value: CalibrationReport) -> CalibrationReport:
        async with self._lock:
            self.calibrations[value.report_id] = value
        return value

    async def latest_calibration(self, model_name: str) -> CalibrationReport | None:
        async with self._lock:
            values = [value for value in self.calibrations.values() if value.model_name == model_name]
        return max(values, key=lambda value: (value.generated_at, str(value.report_id)), default=None)


class SqlAlchemyQuantForecastRepository(ScopedSessionRepository):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        super().__init__(session_factory)

    @scoped_session
    async def save_model_metadata(self, value: ModelMetadata) -> ModelMetadata:
        await self.session.execute(
            insert(QuantModelMetadataRecord)
            .values(
                model_name=value.model_name,
                model_version=value.model_version,
                payload=value.model_dump(mode="json"),
            )
            .on_conflict_do_update(
                index_elements=["model_name", "model_version"],
                set_={"payload": value.model_dump(mode="json")},
            )
        )
        await self.session.commit()
        return value

    @scoped_session
    async def save_request(self, value: QuantForecastRequest) -> QuantForecastRequest:
        await self.session.execute(
            insert(QuantForecastRequestRecord)
            .values(
                request_id=value.request_id,
                market_state_id=value.market_state_id,
                cycle_id=value.cycle_id,
                instrument=value.instrument,
                point_in_time=value.point_in_time,
                model_name=value.model_name,
                model_version=value.model_version,
                mode=value.mode.value,
                payload=value.model_dump(mode="json"),
                created_at=value.created_at,
            )
            .on_conflict_do_nothing(index_elements=["request_id"])
        )
        await self.session.commit()
        return value

    @scoped_session
    async def save_features(self, value: QuantFeatureVector) -> QuantFeatureVector:
        await self.session.execute(
            insert(QuantFeatureVectorRecord)
            .values(
                vector_id=value.vector_id,
                market_state_id=value.market_state_id,
                instrument=value.instrument,
                point_in_time=value.point_in_time,
                schema_version=value.schema_version,
                payload=value.model_dump(mode="json"),
                created_at=value.created_at,
            )
            .on_conflict_do_nothing(index_elements=["vector_id"])
        )
        for feature in value.features:
            for evidence_id in feature.source_evidence_ids:
                await self.session.execute(
                    insert(QuantFeatureReferenceRecord)
                    .values(
                        vector_id=value.vector_id,
                        feature_name=feature.name,
                        evidence_id=evidence_id,
                        source_paths=list(feature.source_paths),
                    )
                    .on_conflict_do_nothing(index_elements=["vector_id", "feature_name", "evidence_id"])
                )
        await self.session.commit()
        return value

    @scoped_session
    async def save_result(self, value: QuantForecastResult) -> QuantForecastResult:
        await self.session.execute(
            insert(QuantForecastResultRecord)
            .values(
                result_id=value.result_id,
                request_id=value.request_id,
                market_state_id=value.market_state_id,
                instrument=value.instrument,
                point_in_time=value.point_in_time,
                status=value.status.value,
                model_name=value.model_name,
                model_version=value.model_version,
                payload=value.model_dump(mode="json"),
                generated_at=value.generated_at,
            )
            .on_conflict_do_nothing(index_elements=["result_id"])
        )
        for prediction in value.predictions:
            await self.session.execute(
                insert(QuantForecastHorizonRecord)
                .values(
                    result_id=value.result_id,
                    horizon_id=prediction.horizon.horizon_id,
                    duration_seconds=prediction.horizon.duration_seconds,
                    payload=prediction.model_dump(mode="json"),
                )
                .on_conflict_do_nothing(index_elements=["result_id", "horizon_id"])
            )
        await self.session.commit()
        return value

    @scoped_session
    async def latest_result(self, instrument: str) -> QuantForecastResult | None:
        query = (
            select(QuantForecastResultRecord)
            .where(QuantForecastResultRecord.instrument == instrument)
            .order_by(QuantForecastResultRecord.point_in_time.desc())
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        return QuantForecastResult.model_validate(record.payload) if record else None

    @scoped_session
    async def save_outcome(self, value: ForecastOutcome) -> ForecastOutcome:
        await self.session.execute(
            insert(QuantForecastOutcomeRecord)
            .values(
                outcome_id=value.outcome_id,
                result_id=value.forecast_result_id,
                horizon_id=value.horizon_id,
                status=value.status.value,
                payload=value.model_dump(mode="json"),
                evaluated_at=value.evaluated_at,
            )
            .on_conflict_do_update(
                index_elements=["result_id", "horizon_id"],
                set_={
                    "status": value.status.value,
                    "payload": value.model_dump(mode="json"),
                    "evaluated_at": value.evaluated_at,
                },
            )
        )
        await self.session.commit()
        return value

    @scoped_session
    async def outcomes_for_result(self, result_id: UUID) -> tuple[ForecastOutcome, ...]:
        query = (
            select(QuantForecastOutcomeRecord)
            .where(QuantForecastOutcomeRecord.result_id == result_id)
            .order_by(QuantForecastOutcomeRecord.horizon_id)
        )
        records = (await self.session.scalars(query)).all()
        return tuple(ForecastOutcome.model_validate(record.payload) for record in records)

    @scoped_session
    async def save_calibration(self, value: CalibrationReport) -> CalibrationReport:
        await self.session.execute(
            insert(QuantCalibrationReportRecord)
            .values(
                report_id=value.report_id,
                model_name=value.model_name,
                model_version=value.model_version,
                sample_count=value.sample_count,
                status=value.status.value,
                payload=value.model_dump(mode="json"),
                generated_at=value.generated_at,
            )
            .on_conflict_do_nothing(index_elements=["report_id"])
        )
        for ordinal, bucket in enumerate(value.buckets):
            await self.session.execute(
                insert(QuantCalibrationBucketRecord)
                .values(
                    report_id=value.report_id,
                    ordinal=ordinal,
                    horizon_id=bucket.horizon_id,
                    dimension=bucket.dimension,
                    payload=bucket.model_dump(mode="json"),
                )
                .on_conflict_do_nothing(index_elements=["report_id", "ordinal"])
            )
        await self.session.commit()
        return value

    @scoped_session
    async def latest_calibration(self, model_name: str) -> CalibrationReport | None:
        query = (
            select(QuantCalibrationReportRecord)
            .where(QuantCalibrationReportRecord.model_name == model_name)
            .order_by(QuantCalibrationReportRecord.generated_at.desc())
            .limit(1)
        )
        record = (await self.session.scalars(query)).first()
        return CalibrationReport.model_validate(record.payload) if record else None
