"""Data models and abstract interfaces for AquaQuery."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Domain Models ---


@dataclass(frozen=True)
class OceanProfile:
    """A single Argo ocean profile measurement."""

    latitude: float
    longitude: float
    timestamp: datetime
    depth_levels: tuple[float, ...]
    variables: dict[str, tuple[float, ...]]
    float_id: str = ""
    cycle_number: int = 0


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata about available data coverage."""

    lat_bounds: tuple[float, float]
    lon_bounds: tuple[float, float]
    depth_range: tuple[float, float]
    time_range: tuple[str, str]
    available_variables: tuple[str, ...]
    total_profiles: int
    data_source: str = "Argo GDAC"
    last_updated: str = ""


# --- Request/Response Models ---


class QueryParams(BaseModel):
    """Parameters for querying ocean data."""

    variable: str = Field(description="Variable name (TEMP, PSAL, PRES, DOXY)")
    lat_min: float = Field(default=-90.0, ge=-90.0, le=90.0)
    lat_max: float = Field(default=90.0, ge=-90.0, le=90.0)
    lon_min: float = Field(default=-180.0, ge=-180.0, le=180.0)
    lon_max: float = Field(default=180.0, ge=-180.0, le=180.0)
    depth_min: float = Field(default=0.0, ge=0.0)
    depth_max: float = Field(default=2000.0, ge=0.0)
    start_date: str | None = None
    end_date: str | None = None


class VariableInfo(BaseModel):
    """Information about an available variable."""

    name: str
    display_name: str
    unit: str
    description: str
    typical_range: tuple[float, float]


class ChatRequest(BaseModel):
    """Request to send a chat message."""

    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    session_id: str
    message_id: str
    content: str
    visualization: dict[str, Any] | None = None
    sources: list[str] = Field(default_factory=list)
    agent_path: list[str] = Field(default_factory=list)
    timestamp: str = ""


class Message(BaseModel):
    """A single chat message."""

    id: str
    role: str  # "user" or "assistant"
    content: str
    visualization: dict[str, Any] | None = None
    sources: list[str] = Field(default_factory=list)
    timestamp: str = ""


class ErrorResponse(BaseModel):
    """Standard error response."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


# --- Abstract Interface ---


class DataSource(ABC):
    """Abstract base class for data sources.

    Implement this interface to add new spatial-temporal datasets
    (climate, satellite, seismic, etc.) to AquaQuery.
    """

    @abstractmethod
    def fetch_region(
        self,
        lat_bounds: tuple[float, float],
        lon_bounds: tuple[float, float],
        depth_range: tuple[float, float] = (0.0, 2000.0),
        time_range: tuple[str, str] | None = None,
    ) -> Any:
        """Fetch data for a geographic region.

        Returns an xarray Dataset or equivalent.
        """

    @abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        """Return metadata about dataset coverage."""

    @abstractmethod
    def get_available_variables(self) -> list[VariableInfo]:
        """Return list of available variables with descriptions."""
