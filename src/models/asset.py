"""Asset domain model.

Represents a physical power generation asset (e.g., generator, turbine)
tracked by the Asset Health Index platform. Maps to the future ``Asset``
database table.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AssetCriticality(str, Enum):
    """Business criticality ranking of an asset."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class AssetStatus(str, Enum):
    """Operational status of an asset."""

    ACTIVE = "Active"
    STANDBY = "Standby"
    MAINTENANCE = "Maintenance"
    DECOMMISSIONED = "Decommissioned"


class Asset(BaseModel):
    """Represents a power generation asset.

    Attributes:
        asset_id: Unique identifier of the asset (primary key).
        asset_name: Human-readable asset name.
        asset_type: Type/category of asset (e.g., "Generator", "Turbine").
        location: Physical location or site name.
        commissioning_date: Date the asset was commissioned into service.
        manufacturer: Original equipment manufacturer name.
        criticality: Business criticality ranking.
        status: Current operational status.
    """

    model_config = ConfigDict(use_enum_values=True, from_attributes=True)

    asset_id: str = Field(..., min_length=1, description="Unique asset identifier.")
    asset_name: str = Field(..., min_length=1, description="Human-readable asset name.")
    asset_type: str = Field(..., min_length=1, description="Asset type/category.")
    location: str = Field(..., min_length=1, description="Physical location or site.")
    commissioning_date: date = Field(..., description="Date the asset entered service.")
    manufacturer: str = Field(..., min_length=1, description="Original equipment manufacturer.")
    criticality: AssetCriticality = Field(
        default=AssetCriticality.MEDIUM, description="Business criticality ranking."
    )
    status: AssetStatus = Field(
        default=AssetStatus.ACTIVE, description="Current operational status."
    )


class AssetCreate(BaseModel):
    """Payload model used to create a new :class:`Asset` record."""

    asset_id: str = Field(..., min_length=1)
    asset_name: str = Field(..., min_length=1)
    asset_type: str = Field(..., min_length=1)
    location: str = Field(..., min_length=1)
    commissioning_date: date
    manufacturer: str = Field(..., min_length=1)
    criticality: AssetCriticality = AssetCriticality.MEDIUM
    status: AssetStatus = AssetStatus.ACTIVE


class AssetUpdate(BaseModel):
    """Partial update payload for an existing :class:`Asset` record."""

    asset_name: Optional[str] = None
    asset_type: Optional[str] = None
    location: Optional[str] = None
    manufacturer: Optional[str] = None
    criticality: Optional[AssetCriticality] = None
    status: Optional[AssetStatus] = None
