"""OperationalMeasurement domain model.

Represents a single time-stamped operational reading captured from an
asset's sensors/SCADA/historian systems. Maps to the future
``OperationalMeasurement`` database table.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class OperationalMeasurement(BaseModel):
    """Represents one operational measurement record for an asset.

    Attributes:
        measurement_id: Unique identifier of the measurement record.
        asset_id: Foreign key reference to the owning :class:`Asset`.
        timestamp: UTC timestamp when the measurement was captured.
        speed: Rotational speed (RPM).
        frequency: Electrical frequency (Hz).
        voltage: Terminal voltage (V).
        current: Line current (A).
        active_power: Active (real) power output (MW).
        reactive_power: Reactive power output (MVAr).
        power_factor: Power factor (dimensionless, -1.0 to 1.0).
    """

    model_config = ConfigDict(from_attributes=True)

    measurement_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1)
    timestamp: datetime

    speed: float = Field(..., description="Rotational speed (RPM).")
    frequency: float = Field(..., description="Electrical frequency (Hz).")
    voltage: float = Field(..., description="Terminal voltage (V).")
    current: float = Field(..., description="Line current (A).")
    active_power: float = Field(..., description="Active power output (MW).")
    reactive_power: float = Field(..., description="Reactive power output (MVAr).")
    power_factor: float = Field(..., ge=-1.0, le=1.0, description="Power factor.")

    def to_feature_vector(self) -> list[float]:
        """Return the seven model input variables in canonical order.

        Order matches ``INPUT_FEATURE_NAMES`` in
        ``src/utils/constants.py`` and the Han et al. (2023) methodology:
        speed, frequency, voltage, current, active power, reactive power,
        power factor.
        """
        return [
            self.speed,
            self.frequency,
            self.voltage,
            self.current,
            self.active_power,
            self.reactive_power,
            self.power_factor,
        ]


class OperationalMeasurementCreate(BaseModel):
    """Payload model used to create a new measurement record."""

    measurement_id: str = Field(..., min_length=1)
    asset_id: str = Field(..., min_length=1)
    timestamp: datetime
    speed: float
    frequency: float
    voltage: float
    current: float
    active_power: float
    reactive_power: float
    power_factor: float = Field(..., ge=-1.0, le=1.0)


class OperationalMeasurementBatch(BaseModel):
    """A batch of operational measurements, e.g. for bulk ingestion."""

    measurements: list[OperationalMeasurement] = Field(default_factory=list)

    def asset_ids(self) -> set[str]:
        """Return the distinct set of asset IDs present in this batch."""
        return {m.asset_id for m in self.measurements}
