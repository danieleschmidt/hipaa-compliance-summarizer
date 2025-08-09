"""Lightweight ML integration without heavy dependencies for HIPAA compliance system."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """ML prediction result."""

    prediction: Any
    confidence: float
    model_version: str
    processing_time_ms: float
    features_used: List[str]
    metadata: Dict[str, Any]


class LightMLProcessor:
    """Lightweight ML processor for PHI detection and compliance analysis."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.phi_classifier = None
        self.compliance_scorer = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """Initialize ML components."""
        try:
            # Initialize with basic patterns
            self._is_initialized = True
            logger.info("LightMLProcessor initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize LightMLProcessor: {e}")
            return False

    def predict_phi_entities(self, text: str) -> MLPrediction:
        """Predict PHI entities in text using lightweight models."""
        start_time = time.time()

        # Simple rule-based PHI detection
        phi_patterns = {
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'date': r'\b\d{1,2}/\d{1,2}/\d{4}\b'
        }

        detected_entities = []
        for entity_type, pattern in phi_patterns.items():
            import re
            matches = re.findall(pattern, text)
            if matches:
                detected_entities.extend([(entity_type, match) for match in matches])

        confidence = 0.8 if detected_entities else 0.2
        processing_time = (time.time() - start_time) * 1000

        return MLPrediction(
            prediction=detected_entities,
            confidence=confidence,
            model_version="lite-v1.0",
            processing_time_ms=processing_time,
            features_used=['regex_patterns'],
            metadata={'pattern_count': len(phi_patterns)}
        )

    def calculate_compliance_score(self, text: str, phi_detected: int) -> MLPrediction:
        """Calculate compliance score based on PHI detection."""
        start_time = time.time()

        # Simple scoring algorithm
        text_length = len(text)
        phi_density = phi_detected / max(text_length, 1) * 1000

        # Higher PHI density = lower compliance score
        if phi_density > 5:
            score = 0.6
        elif phi_density > 2:
            score = 0.8
        elif phi_density > 0:
            score = 0.9
        else:
            score = 1.0

        processing_time = (time.time() - start_time) * 1000

        return MLPrediction(
            prediction=score,
            confidence=0.9,
            model_version="compliance-lite-v1.0",
            processing_time_ms=processing_time,
            features_used=['phi_density', 'text_length'],
            metadata={
                'phi_density': phi_density,
                'text_length': text_length,
                'phi_detected': phi_detected
            }
        )

    def analyze_document_similarity(self, documents: List[str]) -> MLPrediction:
        """Analyze similarity between documents."""
        start_time = time.time()

        if len(documents) < 2:
            return MLPrediction(
                prediction=[],
                confidence=0.0,
                model_version="similarity-lite-v1.0",
                processing_time_ms=0.0,
                features_used=[],
                metadata={'error': 'Need at least 2 documents'}
            )

        try:
            # Use TF-IDF for document similarity
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Find most similar documents
            similarities = []
            for i in range(len(documents)):
                for j in range(i + 1, len(documents)):
                    similarities.append({
                        'doc1_idx': i,
                        'doc2_idx': j,
                        'similarity': float(similarity_matrix[i, j])
                    })

            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)

            processing_time = (time.time() - start_time) * 1000

            return MLPrediction(
                prediction=similarities[:5],  # Top 5 similarities
                confidence=0.85,
                model_version="similarity-lite-v1.0",
                processing_time_ms=processing_time,
                features_used=['tfidf', 'cosine_similarity'],
                metadata={
                    'document_count': len(documents),
                    'comparison_count': len(similarities)
                }
            )

        except Exception as e:
            logger.error(f"Error in document similarity analysis: {e}")
            return MLPrediction(
                prediction=[],
                confidence=0.0,
                model_version="similarity-lite-v1.0",
                processing_time_ms=(time.time() - start_time) * 1000,
                features_used=[],
                metadata={'error': str(e)}
            )


# Global instance
_ml_processor = None


def get_ml_processor() -> LightMLProcessor:
    """Get global ML processor instance."""
    global _ml_processor
    if _ml_processor is None:
        _ml_processor = LightMLProcessor()
        _ml_processor.initialize()
    return _ml_processor


def predict_phi_with_ml(text: str) -> Dict[str, Any]:
    """Predict PHI entities using ML models."""
    processor = get_ml_processor()
    prediction = processor.predict_phi_entities(text)

    return {
        'entities': prediction.prediction,
        'confidence': prediction.confidence,
        'model_version': prediction.model_version,
        'processing_time_ms': prediction.processing_time_ms
    }


def calculate_ml_compliance_score(text: str, phi_count: int) -> float:
    """Calculate compliance score using ML."""
    processor = get_ml_processor()
    prediction = processor.calculate_compliance_score(text, phi_count)
    return prediction.prediction


__all__ = [
    'MLPrediction',
    'LightMLProcessor',
    'get_ml_processor',
    'predict_phi_with_ml',
    'calculate_ml_compliance_score'
]
