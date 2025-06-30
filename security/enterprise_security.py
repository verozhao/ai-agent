"""
Enterprise-grade security and compliance framework for AI Agent System.
Implements zero-trust architecture, end-to-end encryption, and compliance controls.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import hashlib
import secrets

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
from passlib.context import CryptContext

from security.audit_logger import AuditLogger
from security.threat_detector import ThreatDetector
from security.compliance_manager import ComplianceManager


logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ComplianceFramework(Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    roles: Set[str]
    permissions: Set[str]
    security_level: SecurityLevel
    ip_address: str
    session_id: str
    mfa_verified: bool
    compliance_frameworks: Set[ComplianceFramework]


class EnterpriseSecurityManager:
    """
    Comprehensive security manager implementing defense-in-depth strategy.
    """
    
    def __init__(
        self,
        encryption_key: Optional[bytes] = None,
        compliance_frameworks: List[ComplianceFramework] = None
    ):
        # Encryption setup
        self.master_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.master_key)
        
        # RSA key pair for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        self.public_key = self.private_key.public_key()
        
        # Password hashing
        self.pwd_context = CryptContext(
            schemes=["argon2", "bcrypt"],
            deprecated="auto"
        )
        
        # JWT configuration
        self.jwt_secret = secrets.token_urlsafe(32)
        self.jwt_algorithm = "HS256"
        
        # Components
        self.audit_logger = AuditLogger()
        self.threat_detector = ThreatDetector()
        self.compliance_manager = ComplianceManager(
            frameworks=compliance_frameworks or []
        )
        
        # Security policies
        self.policies = self._load_security_policies()
        
        # Session management
        self.active_sessions = {}
        self.failed_attempts = {}
        
    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security policies"""
        return {
            "password_policy": {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": True,
                "history_count": 5,
                "max_age_days": 90
            },
            "session_policy": {
                "max_duration_minutes": 480,
                "idle_timeout_minutes": 30,
                "concurrent_sessions": 3,
                "require_mfa": True
            },
            "access_control": {
                "default_deny": True,
                "ip_whitelist": [],
                "rate_limiting": {
                    "requests_per_minute": 100,
                    "burst_size": 200
                }
            },
            "data_protection": {
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "data_retention_days": 365,
                "secure_deletion": True
            }
        }
    
    # =========================================================================
    # Authentication & Authorization
    # =========================================================================
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        mfa_token: Optional[str] = None,
        ip_address: str = None
    ) -> Optional[SecurityContext]:
        """
        Authenticate user with multi-factor authentication.
        """
        # Check for brute force attempts
        if await self._check_brute_force(username, ip_address):
            await self.audit_logger.log_security_event(
                event_type="authentication_blocked",
                user_id=username,
                details={"reason": "brute_force_protection"}
            )
            return None
        
        # Verify credentials
        user = await self._get_user(username)
        if not user or not self.pwd_context.verify(password, user["password_hash"]):
            await self._record_failed_attempt(username, ip_address)
            return None
        
        # Verify MFA if required
        if self.policies["session_policy"]["require_mfa"]:
            if not mfa_token or not await self._verify_mfa(user["id"], mfa_token):
                await self.audit_logger.log_security_event(
                    event_type="mfa_failed",
                    user_id=user["id"]
                )
                return None
        
        # Create security context
        context = SecurityContext(
            user_id=user["id"],
            roles=set(user["roles"]),
            permissions=await self._get_user_permissions(user["id"]),
            security_level=SecurityLevel(user["security_level"]),
            ip_address=ip_address,
            session_id=secrets.token_urlsafe(32),
            mfa_verified=True,
            compliance_frameworks=set(user.get("compliance_frameworks", []))
        )
        
        # Create session
        await self._create_session(context)
        
        # Log successful authentication
        await self.audit_logger.log_security_event(
            event_type="authentication_success",
            user_id=user["id"],
            details={"ip_address": ip_address}
        )
        
        return context
    
    async def authorize_action(
        self,
        context: SecurityContext,
        resource: str,
        action: str,
        resource_attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Authorize action using ABAC (Attribute-Based Access Control).
        """
        # Check session validity
        if not await self._validate_session(context.session_id):
            return False
        
        # Build authorization request
        auth_request = {
            "subject": {
                "id": context.user_id,
                "roles": list(context.roles),
                "security_level": context.security_level.value
            },
            "resource": {
                "type": resource,
                "attributes": resource_attributes or {}
            },
            "action": action,
            "environment": {
                "ip_address": context.ip_address,
                "time": datetime.utcnow().isoformat(),
                "mfa_verified": context.mfa_verified
            }
        }
        
        # Evaluate policies
        decision = await self._evaluate_policies(auth_request)
        
        # Log authorization attempt
        await self.audit_logger.log_authorization(
            user_id=context.user_id,
            resource=resource,
            action=action,
            decision=decision,
            context=auth_request
        )
        
        return decision
    
    # =========================================================================
    # Encryption & Data Protection
    # =========================================================================
    
    async def encrypt_document(
        self,
        document: Dict[str, Any],
        security_level: SecurityLevel,
        context: SecurityContext
    ) -> Dict[str, Any]:
        """
        Encrypt document with appropriate encryption based on security level.
        """
        # Validate authorization
        if not await self.authorize_action(
            context, "document", "encrypt",
            {"security_level": security_level.value}
        ):
            raise PermissionError("Not authorized to encrypt document")
        
        # Serialize document
        document_bytes = json.dumps(document).encode()
        
        # Apply encryption based on security level
        if security_level in [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]:
            # Use asymmetric encryption for high security
            encrypted = self.public_key.encrypt(
                document_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        else:
            # Use symmetric encryption for standard security
            encrypted = self.cipher_suite.encrypt(document_bytes)
        
        # Create encrypted document
        encrypted_doc = {
            "id": document.get("id", secrets.token_urlsafe(16)),
            "encrypted_data": encrypted.hex(),
            "security_level": security_level.value,
            "encryption_algorithm": "RSA-OAEP" if security_level.value in ["secret", "top_secret"] else "Fernet",
            "encrypted_by": context.user_id,
            "encrypted_at": datetime.utcnow().isoformat(),
            "key_version": "v1"
        }
        
        # Add integrity check
        encrypted_doc["integrity_hash"] = self._calculate_integrity_hash(encrypted_doc)
        
        # Log encryption event
        await self.audit_logger.log_encryption_event(
            user_id=context.user_id,
            document_id=encrypted_doc["id"],
            security_level=security_level.value
        )
        
        return encrypted_doc
    
    async def decrypt_document(
        self,
        encrypted_doc: Dict[str, Any],
        context: SecurityContext
    ) -> Dict[str, Any]:
        """
        Decrypt document with proper authorization.
        """
        # Verify integrity
        if not self._verify_integrity(encrypted_doc):
            raise ValueError("Document integrity check failed")
        
        # Check authorization
        security_level = SecurityLevel(encrypted_doc["security_level"])
        if not await self.authorize_action(
            context, "document", "decrypt",
            {"security_level": security_level.value}
        ):
            raise PermissionError("Not authorized to decrypt document")
        
        # Decrypt based on algorithm
        encrypted_data = bytes.fromhex(encrypted_doc["encrypted_data"])
        
        if encrypted_doc["encryption_algorithm"] == "RSA-OAEP":
            decrypted = self.private_key.decrypt(
                encrypted_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        else:
            decrypted = self.cipher_suite.decrypt(encrypted_data)
        
        # Parse document
        document = json.loads(decrypted.decode())
        
        # Log decryption event
        await self.audit_logger.log_decryption_event(
            user_id=context.user_id,
            document_id=encrypted_doc["id"],
            security_level=security_level.value
        )
        
        return document
    
    # =========================================================================
    # Threat Detection & Response
    # =========================================================================
    
    async def detect_threats(
        self,
        activity_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect security threats using ML and rule-based detection.
        """
        threats = []
        
        # Check for anomalous behavior
        anomaly_score = await self.threat_detector.analyze_behavior(activity_data)
        if anomaly_score > 0.8:
            threats.append({
                "type": "anomalous_behavior",
                "severity": "high",
                "score": anomaly_score,
                "details": activity_data
            })
        
        # Check for known attack patterns
        attack_patterns = await self.threat_detector.check_attack_patterns(
            activity_data
        )
        threats.extend(attack_patterns)
        
        # Check for data exfiltration
        if await self._detect_data_exfiltration(activity_data):
            threats.append({
                "type": "data_exfiltration",
                "severity": "critical",
                "details": activity_data
            })
        
        # Automated response for critical threats
        for threat in threats:
            if threat["severity"] == "critical":
                await self._respond_to_threat(threat)
        
        return threats
    
    async def _respond_to_threat(
        self,
        threat: Dict[str, Any]
    ):
        """
        Automated threat response.
        """
        if threat["type"] == "data_exfiltration":
            # Block user access immediately
            user_id = threat["details"].get("user_id")
            if user_id:
                await self._revoke_all_sessions(user_id)
                await self._block_user(user_id)
        
        elif threat["type"] == "brute_force":
            # Implement IP blocking
            ip_address = threat["details"].get("ip_address")
            if ip_address:
                await self._block_ip(ip_address)
        
        # Alert security team
        await self._alert_security_team(threat)
        
        # Log threat response
        await self.audit_logger.log_security_event(
            event_type="threat_response",
            details=threat
        )
    
    # =========================================================================
    # Compliance & Privacy
    # =========================================================================
    
    async def apply_privacy_controls(
        self,
        data: Dict[str, Any],
        compliance_framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """
        Apply privacy controls based on compliance requirements.
        """
        if compliance_framework == ComplianceFramework.GDPR:
            return await self._apply_gdpr_controls(data)
        elif compliance_framework == ComplianceFramework.HIPAA:
            return await self._apply_hipaa_controls(data)
        elif compliance_framework == ComplianceFramework.PCI_DSS:
            return await self._apply_pci_controls(data)
        else:
            return data
    
    async def _apply_gdpr_controls(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply GDPR privacy controls.
        """
        # Implement right to be forgotten
        if data.get("deletion_requested"):
            return await self._secure_deletion(data)
        
        # Pseudonymization of personal data
        sensitive_fields = ["name", "email", "phone", "address", "ip_address"]
        
        processed_data = data.copy()
        for field in sensitive_fields:
            if field in processed_data:
                processed_data[field] = self._pseudonymize(processed_data[field])
        
        # Add consent tracking
        processed_data["gdpr_consent"] = {
            "purpose": data.get("processing_purpose"),
            "timestamp": datetime.utcnow().isoformat(),
            "withdrawal_method": "DELETE /api/v1/consent/{user_id}"
        }
        
        return processed_data
    
    async def _apply_hipaa_controls(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply HIPAA privacy controls for healthcare data.
        """
        # Identify PHI (Protected Health Information)
        phi_fields = [
            "patient_name", "ssn", "medical_record_number",
            "diagnosis", "treatment", "medications"
        ]
        
        processed_data = data.copy()
        
        # Encrypt PHI fields
        for field in phi_fields:
            if field in processed_data:
                processed_data[field] = {
                    "encrypted": True,
                    "value": self.cipher_suite.encrypt(
                        str(processed_data[field]).encode()
                    ).decode(),
                    "access_control": "hipaa_authorized_only"
                }
        
        # Add HIPAA audit requirements
        processed_data["hipaa_audit"] = {
            "access_log_required": True,
            "retention_years": 6,
            "breach_notification_required": True
        }
        
        return processed_data
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate compliance report for audits.
        """
        report = {
            "framework": framework.value,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {}
        }
        
        # Get compliance metrics
        metrics = await self.compliance_manager.get_metrics(
            framework, start_date, end_date
        )
        
        report["metrics"] = metrics
        
        # Get audit logs
        audit_logs = await self.audit_logger.get_logs(
            start_date, end_date,
            compliance_framework=framework
        )
        
        report["audit_summary"] = {
            "total_events": len(audit_logs),
            "access_events": sum(1 for log in audit_logs if log["type"] == "access"),
            "modification_events": sum(1 for log in audit_logs if log["type"] == "modification"),
            "security_incidents": sum(1 for log in audit_logs if log["type"] == "security_incident")
        }
        
        # Framework-specific sections
        if framework == ComplianceFramework.SOC2:
            report["soc2_controls"] = await self._generate_soc2_controls()
        elif framework == ComplianceFramework.ISO27001:
            report["iso27001_controls"] = await self._generate_iso27001_controls()
        
        return report
    
    # =========================================================================
    # Zero Trust Architecture
    # =========================================================================
    
    async def verify_zero_trust(
        self,
        context: SecurityContext,
        resource: str,
        action: str
    ) -> bool:
        """
        Implement zero trust verification for every access.
        """
        # Never trust, always verify
        checks = {
            "identity_verified": await self._verify_identity(context),
            "device_trusted": await self._verify_device(context),
            "location_allowed": await self._verify_location(context),
            "risk_acceptable": await self._assess_risk(context, resource, action),
            "session_valid": await self._validate_session(context.session_id)
        }
        
        # All checks must pass
        all_passed = all(checks.values())
        
        # Log zero trust decision
        await self.audit_logger.log_security_event(
            event_type="zero_trust_verification",
            user_id=context.user_id,
            details={
                "resource": resource,
                "action": action,
                "checks": checks,
                "decision": "allow" if all_passed else "deny"
            }
        )
        
        return all_passed
    
    async def _assess_risk(
        self,
        context: SecurityContext,
        resource: str,
        action: str
    ) -> bool:
        """
        Assess risk score for the access request.
        """
        risk_factors = {
            "user_risk": await self._calculate_user_risk(context.user_id),
            "resource_sensitivity": self._get_resource_sensitivity(resource),
            "action_risk": self._get_action_risk(action),
            "environmental_risk": await self._calculate_environmental_risk(context)
        }
        
        # Calculate composite risk score
        risk_score = sum(risk_factors.values()) / len(risk_factors)
        
        # Determine if risk is acceptable
        risk_threshold = 0.7
        if context.security_level == SecurityLevel.TOP_SECRET:
            risk_threshold = 0.3
        elif context.security_level == SecurityLevel.SECRET:
            risk_threshold = 0.5
        
        return risk_score < risk_threshold
    
    # =========================================================================
    # Security Utilities
    # =========================================================================
    
    def _calculate_integrity_hash(self, data: Dict[str, Any]) -> str:
        """Calculate integrity hash for data."""
        # Remove existing hash if present
        data_copy = {k: v for k, v in data.items() if k != "integrity_hash"}
        
        # Canonical JSON serialization
        canonical = json.dumps(data_copy, sort_keys=True)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def _verify_integrity(self, data: Dict[str, Any]) -> bool:
        """Verify data integrity."""
        stored_hash = data.get("integrity_hash")
        calculated_hash = self._calculate_integrity_hash(data)
        
        return stored_hash == calculated_hash
    
    def _pseudonymize(self, value: str) -> str:
        """Pseudonymize sensitive data."""
        # Use HMAC for deterministic pseudonymization
        key = self.master_key[:32]  # Use part of master key
        h = hashlib.blake2b(value.encode(), key=key, digest_size=16)
        return f"PSEUDO_{h.hexdigest()}"
    
    async def _secure_deletion(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Securely delete sensitive data."""
        # Overwrite memory
        for key in list(data.keys()):
            if isinstance(data[key], str):
                data[key] = "X" * len(data[key])
            data[key] = None
        
        # Return deletion confirmation
        return {
            "status": "deleted",
            "deletion_timestamp": datetime.utcnow().isoformat(),
            "deletion_method": "secure_overwrite"
        }