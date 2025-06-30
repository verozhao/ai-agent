"""
SQLAlchemy database models for document extraction and feedback system.
"""

from datetime import datetime
from typing import Dict, Any
import uuid

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    Text, JSON, ForeignKey, Index, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from core.extraction.engine import ProcessingStatus, Quarter


Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class Document(Base):
    """Document model for uploaded files"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=False)
    
    # Metadata
    quarter = Column(SQLEnum(Quarter), nullable=False)
    document_type = Column(String(100))
    source = Column(String(200))
    
    # Ownership and tracking
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="documents")
    extractions = relationship("Extraction", back_populates="document")
    
    # Indexes for performance
    __table_args__ = (
        Index("idx_document_quarter", "quarter"),
        Index("idx_document_owner", "owner_id"),
        Index("idx_document_created", "created_at"),
    )


class Extraction(Base):
    """Extraction results and metadata"""
    __tablename__ = "extractions"
    
    id = Column(String(100), primary_key=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Extraction results
    extracted_data = Column(JSON, nullable=False)
    confidence_scores = Column(JSON, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    
    # Processing metadata
    status = Column(SQLEnum(ProcessingStatus), nullable=False)
    model_version = Column(String(50), nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    
    # Audit tracking
    audit_completed_at = Column(DateTime)
    auditor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="extractions")
    audit_logs = relationship("AuditLog", back_populates="extraction")
    feedback_data = relationship("FeedbackData", back_populates="extraction")
    
    # Indexes
    __table_args__ = (
        Index("idx_extraction_status", "status"),
        Index("idx_extraction_anomaly", "anomaly_score"),
        Index("idx_extraction_document", "document_id"),
    )


class AuditLog(Base):
    """Audit log for human review tasks"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    extraction_id = Column(String(100), ForeignKey("extractions.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Audit details
    auditor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    status = Column(String(50), nullable=False, default="pending")
    priority = Column(String(20), nullable=False, default="normal")
    
    # Routing reason
    reason = Column(String(100))
    anomaly_score = Column(Float)
    
    # Audit results
    corrections = Column(JSON)
    quality_score = Column(Float)
    notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    
    # Relationships
    extraction = relationship("Extraction", back_populates="audit_logs")
    auditor = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index("idx_audit_status", "status"),
        Index("idx_audit_priority", "priority"),
        Index("idx_audit_created", "created_at"),
    )


class FeedbackData(Base):
    """Processed feedback data for model improvement"""
    __tablename__ = "feedback_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    extraction_id = Column(String(100), ForeignKey("extractions.id"), nullable=False)
    
    # Feedback content
    original_data = Column(JSON, nullable=False)
    corrections = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=False)
    
    # Metadata
    auditor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    quarter = Column(SQLEnum(Quarter), nullable=False)
    confidence_delta = Column(Float)
    
    # Training status
    used_in_training = Column(Boolean, default=False)
    training_version = Column(String(50))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    extraction = relationship("Extraction", back_populates="feedback_data")
    
    # Indexes
    __table_args__ = (
        Index("idx_feedback_quarter", "quarter"),
        Index("idx_feedback_training", "used_in_training"),
    )


class ModelVersion(Base):
    """ML model version tracking"""
    __tablename__ = "model_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    version = Column(String(50), unique=True, nullable=False)
    model_type = Column(String(50), nullable=False)
    
    # Model metadata
    architecture = Column(JSON)
    hyperparameters = Column(JSON)
    training_metrics = Column(JSON)
    
    # Performance metrics
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Deployment info
    is_active = Column(Boolean, default=False)
    rollout_percentage = Column(Integer, default=0)
    deployment_date = Column(DateTime)
    
    # Training metadata
    training_samples = Column(Integer)
    training_duration_seconds = Column(Integer)
    improvement_from_previous = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_model_active", "is_active"),
        Index("idx_model_version", "version"),
    )


class TrainingMetrics(Base):
    """Detailed training metrics and history"""
    __tablename__ = "training_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    model_version_id = Column(UUID(as_uuid=True), ForeignKey("model_versions.id"))
    
    # Training details
    epoch = Column(Integer)
    step = Column(Integer)
    
    # Metrics
    loss = Column(Float)
    accuracy = Column(Float)
    learning_rate = Column(Float)
    
    # Validation metrics
    val_loss = Column(Float)
    val_accuracy = Column(Float)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_training_version", "model_version_id"),
        Index("idx_training_epoch", "epoch"),
    )


class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    
    # Roles and permissions
    is_admin = Column(Boolean, default=False)
    is_auditor = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # Organization
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime)
    
    # Relationships
    documents = relationship("Document", back_populates="owner")
    audit_logs = relationship("AuditLog", back_populates="auditor")


class Organization(Base):
    """Organization for multi-tenancy"""
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    name = Column(String(255), unique=True, nullable=False)
    
    # Configuration
    settings = Column(JSON, default={})
    quota_documents = Column(Integer, default=10000)
    quota_api_calls = Column(Integer, default=100000)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    users = relationship("User", backref="organization")
    documents = relationship("Document", backref="organization")


class AnomalyRecord(Base):
    """Anomaly detection history and patterns"""
    __tablename__ = "anomaly_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    extraction_id = Column(String(100), ForeignKey("extractions.id"), nullable=False)
    
    # Anomaly details
    anomaly_type = Column(String(50))
    component_scores = Column(JSON)
    final_score = Column(Float, nullable=False)
    
    # Pattern analysis
    pattern_id = Column(String(100))
    pattern_confidence = Column(Float)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index("idx_anomaly_score", "final_score"),
        Index("idx_anomaly_type", "anomaly_type"),
        Index("idx_anomaly_pattern", "pattern_id"),
    )
