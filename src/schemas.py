"""Pydantic data contracts (see docs/DATA_SCHEMAS.md)."""

from pydantic import BaseModel, Field
from typing import List, Optional


class RawCurvePoint(BaseModel):
    time_hours: float = Field(..., ge=0.0, description="Elapsed time in hours")
    measurement: float = Field(..., ge=0.0, description="Optical density OD600 or Log CFU/mL")


class KineticSeries(BaseModel):
    series_id: str
    organism_name: str
    temperature_celsius: float
    ph: Optional[float] = Field(7.0, ge=0.0, le=14.0)
    water_activity: Optional[float] = Field(0.99, ge=0.0, le=1.0)
    points: List[RawCurvePoint]


class PrimaryFitResult(BaseModel):
    series_id: str
    mu_max: float = Field(..., description="Maximum specific growth rate (1/h)")
    lag_time_lambda: float = Field(..., description="Lag phase duration (h)")
    carrying_capacity_ymax: float = Field(..., description="Asymptotic maximum population")
    r_squared: float
    rmse: float


class DynamicTelemetryPoint(BaseModel):
    timestamp_hours: float = Field(..., ge=0.0)
    temperature_celsius: float
    ph: Optional[float] = Field(None, ge=0.0, le=14.0)
    water_activity: Optional[float] = Field(None, gt=0.0, le=1.0)


class SimulationScenario(BaseModel):
    organism: str
    initial_cfu_per_g: float = Field(100.0, gt=0.0)
    critical_threshold_cfu_per_g: float = Field(1e5, gt=0.0)
    environmental_log: List[DynamicTelemetryPoint]
    ph: float = Field(6.5, ge=0.0, le=14.0)
    water_activity: float = Field(0.98, gt=0.0, le=1.0)


class SimulationOutput(BaseModel):
    time_hours: List[float]
    temperatures: List[float]
    ph: List[float] = Field(default_factory=list)
    water_activity: List[float] = Field(default_factory=list)
    predicted_cfu_per_g: List[float]
    spoilage_breached: bool
    time_to_breach_hours: Optional[float] = None
