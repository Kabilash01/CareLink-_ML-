"""
CARELINK AI - SQLite ORM Models

Covers all PRD entities:
  - User / Patient profile
  - HealthRecord
  - Consultation
  - Prescription + PrescriptionItem
  - TriagePrediction (AI audit log)
  - PharmacyInventory
  - AuditLog
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text,
    DateTime, ForeignKey, JSON, Enum as SAEnum
)
from sqlalchemy.orm import relationship

from app.db.base import Base


def _uuid():
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# User / Patient
# ---------------------------------------------------------------------------

class User(Base):
    """
    Platform user — can be a Patient, Doctor, Pharmacist, Admin, or CHW.
    """
    __tablename__ = "users"

    id              = Column(String(36), primary_key=True, default=_uuid)
    name            = Column(String(255), nullable=False)
    age             = Column(Integer, nullable=True)
    gender          = Column(String(20), nullable=True)          # male/female/other
    contact         = Column(String(20), nullable=True)
    email           = Column(String(255), unique=True, nullable=True)
    address         = Column(Text, nullable=True)
    blood_group     = Column(String(10), nullable=True)
    national_id     = Column(String(100), nullable=True)         # Aadhaar / NID
    preferred_lang  = Column(String(10), default="en")
    role            = Column(String(30), default="patient")      # patient/doctor/pharmacist/admin/chw
    password_hash   = Column(String(255), nullable=True)
    pin_hash        = Column(String(255), nullable=True)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    health_records  = relationship("HealthRecord",   back_populates="patient", cascade="all, delete-orphan")
    consultations   = relationship("Consultation",   back_populates="patient", cascade="all, delete-orphan",
                                   foreign_keys="Consultation.patient_id")
    triage_logs     = relationship("TriagePrediction", back_populates="patient", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Health Record
# ---------------------------------------------------------------------------

class HealthRecord(Base):
    """
    A single entry in the patient's longitudinal health record.
    type: consultation | test | prescription | immunization | vitals
    """
    __tablename__ = "health_records"

    id              = Column(String(36), primary_key=True, default=_uuid)
    patient_id      = Column(String(36), ForeignKey("users.id"), nullable=False)
    record_type     = Column(String(50), nullable=False)         # consultation/test/prescription
    title           = Column(String(255), nullable=True)
    content         = Column(JSON, nullable=True)                # structured payload
    attachments     = Column(JSON, nullable=True)                # list of file paths/URLs
    is_shared       = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relations
    patient         = relationship("User", back_populates="health_records")


# ---------------------------------------------------------------------------
# Consultation
# ---------------------------------------------------------------------------

class Consultation(Base):
    """
    Telemedicine or in-person consultation record.
    """
    __tablename__ = "consultations"

    id              = Column(String(36), primary_key=True, default=_uuid)
    patient_id      = Column(String(36), ForeignKey("users.id"), nullable=False)
    doctor_id       = Column(String(36), ForeignKey("users.id"), nullable=True)
    mode            = Column(String(20), default="video")        # video/audio/text
    status          = Column(String(20), default="scheduled")    # scheduled/in_progress/completed/cancelled
    symptoms_text   = Column(Text, nullable=True)
    diagnosis       = Column(Text, nullable=True)
    notes           = Column(Text, nullable=True)
    follow_up_date  = Column(DateTime, nullable=True)
    duration_min    = Column(Integer, nullable=True)
    recording_path  = Column(String(500), nullable=True)
    scheduled_at    = Column(DateTime, nullable=True)
    started_at      = Column(DateTime, nullable=True)
    ended_at        = Column(DateTime, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    # Relations
    patient         = relationship("User", back_populates="consultations", foreign_keys=[patient_id])
    doctor          = relationship("User", foreign_keys=[doctor_id])
    prescriptions   = relationship("Prescription", back_populates="consultation", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Prescription
# ---------------------------------------------------------------------------

class Prescription(Base):
    __tablename__ = "prescriptions"

    id              = Column(String(36), primary_key=True, default=_uuid)
    consultation_id = Column(String(36), ForeignKey("consultations.id"), nullable=False)
    patient_id      = Column(String(36), ForeignKey("users.id"), nullable=False)
    doctor_id       = Column(String(36), ForeignKey("users.id"), nullable=True)
    valid_until     = Column(DateTime, nullable=True)
    special_notes   = Column(Text, nullable=True)
    is_dispensed    = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow)

    # Relations
    consultation    = relationship("Consultation", back_populates="prescriptions")
    items           = relationship("PrescriptionItem", back_populates="prescription", cascade="all, delete-orphan")


class PrescriptionItem(Base):
    __tablename__ = "prescription_items"

    id              = Column(String(36), primary_key=True, default=_uuid)
    prescription_id = Column(String(36), ForeignKey("prescriptions.id"), nullable=False)
    medicine_name   = Column(String(255), nullable=False)
    generic_name    = Column(String(255), nullable=True)
    dosage          = Column(String(100), nullable=True)         # e.g. "500mg"
    frequency       = Column(String(100), nullable=True)         # e.g. "twice daily"
    duration_days   = Column(Integer, nullable=True)
    instructions    = Column(Text, nullable=True)

    # Relations
    prescription    = relationship("Prescription", back_populates="items")


# ---------------------------------------------------------------------------
# AI Triage Prediction (audit log for every ML inference)
# ---------------------------------------------------------------------------

class TriagePrediction(Base):
    """
    Immutable audit record of every AI triage inference.
    Supports HIPAA audit trails and model performance monitoring.
    """
    __tablename__ = "triage_predictions"

    id              = Column(String(36), primary_key=True, default=_uuid)
    request_id      = Column(String(100), unique=True, nullable=False)
    patient_id      = Column(String(36), ForeignKey("users.id"), nullable=True)
    symptoms_text   = Column(Text, nullable=False)
    patient_age     = Column(Integer, nullable=True)
    duration_days   = Column(Integer, nullable=True)
    chronic_conds   = Column(JSON, nullable=True)
    language        = Column(String(10), default="en")

    # ML output
    prediction      = Column(String(10), nullable=False)         # LOW/MEDIUM/HIGH
    confidence      = Column(Float, nullable=False)
    prob_low        = Column(Float, nullable=True)
    prob_medium     = Column(Float, nullable=True)
    prob_high       = Column(Float, nullable=True)
    rules_triggered = Column(JSON, nullable=True)
    escalated       = Column(Boolean, default=False)
    emergency_flag  = Column(Boolean, default=False)
    explanation     = Column(Text, nullable=True)
    model_version   = Column(String(50), nullable=True)

    # Performance
    processing_ms   = Column(Float, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    # Relations
    patient         = relationship("User", back_populates="triage_logs")


# ---------------------------------------------------------------------------
# Medicine & Pharmacy Inventory
# ---------------------------------------------------------------------------

class Medicine(Base):
    __tablename__ = "medicines"

    id              = Column(String(36), primary_key=True, default=_uuid)
    name            = Column(String(255), nullable=False, index=True)
    generic_name    = Column(String(255), nullable=True)
    brand           = Column(String(255), nullable=True)
    category        = Column(String(100), nullable=True)
    usage           = Column(Text, nullable=True)
    side_effects    = Column(Text, nullable=True)
    contraindications = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    inventory       = relationship("PharmacyInventory", back_populates="medicine")


class Pharmacy(Base):
    __tablename__ = "pharmacies"

    id              = Column(String(36), primary_key=True, default=_uuid)
    name            = Column(String(255), nullable=False)
    contact         = Column(String(20), nullable=True)
    address         = Column(Text, nullable=True)
    latitude        = Column(Float, nullable=True)
    longitude       = Column(Float, nullable=True)
    opening_hours   = Column(String(100), nullable=True)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    inventory       = relationship("PharmacyInventory", back_populates="pharmacy")


class PharmacyInventory(Base):
    __tablename__ = "pharmacy_inventory"

    id              = Column(String(36), primary_key=True, default=_uuid)
    pharmacy_id     = Column(String(36), ForeignKey("pharmacies.id"), nullable=False)
    medicine_id     = Column(String(36), ForeignKey("medicines.id"), nullable=False)
    stock_level     = Column(String(20), default="available")    # available/low/out_of_stock
    quantity        = Column(Integer, nullable=True)
    price           = Column(Float, nullable=True)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    pharmacy        = relationship("Pharmacy", back_populates="inventory")
    medicine        = relationship("Medicine", back_populates="inventory")


# ---------------------------------------------------------------------------
# Audit Log (HIPAA access audit trail)
# ---------------------------------------------------------------------------

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id              = Column(String(36), primary_key=True, default=_uuid)
    user_id         = Column(String(36), ForeignKey("users.id"), nullable=True)
    action          = Column(String(100), nullable=False)        # e.g. "view_record", "triage_predict"
    resource_type   = Column(String(50), nullable=True)
    resource_id     = Column(String(36), nullable=True)
    ip_address      = Column(String(50), nullable=True)
    user_agent      = Column(String(500), nullable=True)
    details         = Column(JSON, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)
