"""
Document extraction engine with feedback loop integration.
Implements quarterly processing with anomaly detection and human audit routing.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import torch
import numpy as np
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image
import pytesseract
from sqlalchemy.orm import Session

from core.anomaly.detector import AnomalyDetector
from core.feedback.loop import FeedbackLoop
from database.models import Document, Extraction, AuditLog
from services.cache_service import CacheService
from utils.metrics import MetricsCollector


logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    EXTRACTED = "extracted"
    ANOMALY_DETECTED = "anomaly_detected"
    EVAL_SET_2 = "eval_set_2"
    AUDITED = "audited"
    COMPLETED = "completed"
    FAILED = "failed"


class Quarter(Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


@dataclass
class ExtractionResult:
    """Structured extraction result with metadata"""
    extraction_id: str
    document_id: str
    extracted_data: Dict[str, Any]
    confidence_scores: Dict[str, float]
    anomaly_score: float
    processing_time_ms: int
    quarter: Quarter
    status: ProcessingStatus
    model_version: str
    metadata: Dict[str, Any]


class DocumentExtractionEngine:
    """
    Core extraction engine implementing state-of-the-art document processing
    with integrated feedback loop and anomaly detection.
    """
    
    def __init__(
        self,
        model_path: str = "microsoft/layoutlmv3-base",
        anomaly_threshold: float = 0.7,
        eval_set_2_threshold: float = 0.5,
        cache_ttl: int = 3600
    ):
        self.model_path = model_path
        self.anomaly_threshold = anomaly_threshold
        self.eval_set_2_threshold = eval_set_2_threshold
        
        # Initialize ML components
        self._init_models()
        
        # Initialize services
        self.anomaly_detector = AnomalyDetector()
        self.feedback_loop = FeedbackLoop()
        self.cache = CacheService(ttl=cache_ttl)
        self.metrics = MetricsCollector()
        
        # Performance tracking
        self.quarterly_metrics = {q: {} for q in Quarter}
        
    def _init_models(self):
        """Initialize ML models with GPU support if available"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load LayoutLMv3 for document understanding
        self.processor = LayoutLMv3Processor.from_pretrained(self.model_path)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.model_path
        ).to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        
    async def process_document(
        self,
        document: Document,
        quarter: Quarter,
        session: Session,
        force_eval_set_2: bool = False
    ) -> ExtractionResult:
        """
        Process a document through the extraction pipeline with full feedback loop.
        
        Args:
            document: Document object to process
            quarter: Current processing quarter
            session: Database session
            force_eval_set_2: Force routing to evaluation set 2
            
        Returns:
            ExtractionResult with all metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cached_result = await self.cache.get(f"extraction:{document.id}")
            if cached_result and not force_eval_set_2:
                logger.info(f"Cache hit for document {document.id}")
                return cached_result
            
            # Extract document content
            logger.info(f"Starting extraction for document {document.id}")
            extracted_data, confidence_scores = await self._extract_content(document)
            
            # Calculate anomaly score
            anomaly_score = await self.anomaly_detector.calculate_score(
                document=document,
                extracted_data=extracted_data,
                confidence_scores=confidence_scores
            )
            
            # Determine routing based on anomaly score
            status = self._determine_routing(anomaly_score, force_eval_set_2)
            
            # Create extraction result
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            result = ExtractionResult(
                extraction_id=f"ext_{document.id}_{datetime.utcnow().timestamp()}",
                document_id=document.id,
                extracted_data=extracted_data,
                confidence_scores=confidence_scores,
                anomaly_score=anomaly_score,
                processing_time_ms=processing_time,
                quarter=quarter,
                status=status,
                model_version=self.model.config._name_or_path,
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "device": str(self.device),
                    "confidence_threshold": self.anomaly_threshold
                }
            )
            
            # Save to database
            await self._save_extraction(result, session)
            
            # Cache result
            await self.cache.set(f"extraction:{document.id}", result)
            
            # Update metrics
            self._update_metrics(result, quarter)
            
            # If anomaly detected, trigger appropriate workflow
            if status == ProcessingStatus.EVAL_SET_2:
                await self._route_to_eval_set_2(result, session)
            
            logger.info(
                f"Extraction completed for document {document.id} "
                f"with status {status.value} in {processing_time}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed for document {document.id}: {str(e)}")
            raise
    
    async def _extract_content(
        self,
        document: Document
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Extract content from document using LayoutLMv3.
        
        Returns:
            Tuple of (extracted_data, confidence_scores)
        """
        # Load and preprocess document
        if document.file_type in ["pdf", "image"]:
            image = await self._load_document_as_image(document)
            text = pytesseract.image_to_string(image)
        else:
            text = await self._extract_text_content(document)
        
        # Process with LayoutLMv3
        encoding = self.processor(
            image if document.file_type in ["pdf", "image"] else None,
            text=text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            confidence_scores = torch.softmax(outputs.logits, dim=-1).max(-1).values
        
        # Post-process predictions
        extracted_data = self._post_process_predictions(
            text, predictions, confidence_scores
        )
        
        # Calculate field-level confidence
        field_confidence = self._calculate_field_confidence(
            extracted_data, confidence_scores
        )
        
        return extracted_data, field_confidence
    
    def _determine_routing(
        self,
        anomaly_score: float,
        force_eval_set_2: bool
    ) -> ProcessingStatus:
        """Determine document routing based on anomaly score"""
        if force_eval_set_2 or anomaly_score >= self.anomaly_threshold:
            return ProcessingStatus.EVAL_SET_2
        elif anomaly_score >= self.eval_set_2_threshold:
            return ProcessingStatus.ANOMALY_DETECTED
        else:
            return ProcessingStatus.EXTRACTED
    
    async def _route_to_eval_set_2(
        self,
        result: ExtractionResult,
        session: Session
    ):
        """Route document to evaluation set 2 for human review"""
        logger.warning(
            f"Document {result.document_id} routed to eval set 2 "
            f"with anomaly score {result.anomaly_score}"
        )
        
        # Create audit task with high priority
        audit_task = AuditLog(
            extraction_id=result.extraction_id,
            document_id=result.document_id,
            status="pending",
            priority="high",
            reason="anomaly_detected",
            anomaly_score=result.anomaly_score,
            created_at=datetime.utcnow()
        )
        
        session.add(audit_task)
        await session.commit()
        
        # Notify audit team
        await self._notify_audit_team(result)
    
    async def apply_feedback(
        self,
        extraction_id: str,
        corrections: Dict[str, Any],
        auditor_id: str,
        session: Session
    ):
        """
        Apply human feedback to improve the model.
        
        Args:
            extraction_id: ID of the extraction
            corrections: Dictionary of field corrections
            auditor_id: ID of the human auditor
            session: Database session
        """
        # Retrieve original extraction
        extraction = await self._get_extraction(extraction_id, session)
        
        # Apply corrections
        feedback_data = {
            "extraction_id": extraction_id,
            "original": extraction.extracted_data,
            "corrections": corrections,
            "auditor_id": auditor_id,
            "timestamp": datetime.utcnow()
        }
        
        # Send to feedback loop for model improvement
        await self.feedback_loop.process_feedback(feedback_data)
        
        # Update extraction status
        extraction.status = ProcessingStatus.AUDITED
        await session.commit()
        
        logger.info(f"Feedback applied for extraction {extraction_id}")
    
    async def process_quarterly_batch(
        self,
        quarter: Quarter,
        document_ids: List[str],
        session: Session,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Process documents in quarterly batches with optimized performance.
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Starting {quarter.value} batch processing for {len(document_ids)} documents")
        
        results = {
            "total": len(document_ids),
            "processed": 0,
            "anomalies": 0,
            "eval_set_2": 0,
            "failed": 0,
            "avg_confidence": 0.0,
            "processing_time_ms": 0
        }
        
        start_time = datetime.utcnow()
        
        # Process in batches for efficiency
        for i in range(0, len(document_ids), batch_size):
            batch_ids = document_ids[i:i + batch_size]
            
            # Parallel processing
            tasks = []
            for doc_id in batch_ids:
                document = await self._get_document(doc_id, session)
                task = asyncio.create_task(
                    self.process_document(document, quarter, session)
                )
                tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update statistics
            for result in batch_results:
                if isinstance(result, Exception):
                    results["failed"] += 1
                    logger.error(f"Batch processing error: {str(result)}")
                else:
                    results["processed"] += 1
                    if result.status == ProcessingStatus.ANOMALY_DETECTED:
                        results["anomalies"] += 1
                    elif result.status == ProcessingStatus.EVAL_SET_2:
                        results["eval_set_2"] += 1
                    
                    # Update average confidence
                    avg_conf = np.mean(list(result.confidence_scores.values()))
                    results["avg_confidence"] = (
                        (results["avg_confidence"] * (results["processed"] - 1) + avg_conf)
                        / results["processed"]
                    )
        
        results["processing_time_ms"] = int(
            (datetime.utcnow() - start_time).total_seconds() * 1000
        )
        
        # Store quarterly metrics
        self.quarterly_metrics[quarter] = results
        
        # Trigger model retraining if needed
        if results["anomalies"] / results["total"] > 0.1:
            logger.warning(f"High anomaly rate ({results['anomalies'] / results['total']:.2%}) in {quarter.value}")
            await self.feedback_loop.trigger_retraining(quarter)
        
        logger.info(f"Completed {quarter.value} batch processing: {results}")
        
        return results
    
    def _update_metrics(self, result: ExtractionResult, quarter: Quarter):
        """Update performance metrics"""
        self.metrics.record_extraction(
            document_id=result.document_id,
            processing_time=result.processing_time_ms,
            confidence=np.mean(list(result.confidence_scores.values())),
            anomaly_score=result.anomaly_score,
            quarter=quarter.value
        )
    
    async def get_quarterly_report(self, quarter: Quarter) -> Dict[str, Any]:
        """Generate comprehensive quarterly report"""
        metrics = self.quarterly_metrics.get(quarter, {})
        
        report = {
            "quarter": quarter.value,
            "summary": metrics,
            "model_performance": await self.feedback_loop.get_model_metrics(),
            "anomaly_trends": await self.anomaly_detector.get_quarterly_trends(quarter),
            "audit_metrics": await self._get_audit_metrics(quarter),
            "recommendations": self._generate_recommendations(metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on metrics"""
        recommendations = []
        
        if metrics.get("anomalies", 0) / metrics.get("total", 1) > 0.1:
            recommendations.append(
                "High anomaly rate detected. Consider reviewing extraction rules "
                "and updating training data."
            )
        
        if metrics.get("avg_confidence", 1.0) < 0.85:
            recommendations.append(
                "Low average confidence scores. Model retraining recommended "
                "with recent audit feedback."
            )
        
        if metrics.get("eval_set_2", 0) > 50:
            recommendations.append(
                "Significant eval set 2 routing. Review anomaly detection "
                "thresholds and common failure patterns."
            )
        
        return recommendations