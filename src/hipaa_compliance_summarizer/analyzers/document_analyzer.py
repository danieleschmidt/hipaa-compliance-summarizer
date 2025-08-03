"""Document analysis for HIPAA compliance processing."""

import re
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
import statistics

from ..monitoring.tracing import trace_operation

logger = logging.getLogger(__name__)


@dataclass
class DocumentAnalysisResult:
    """Result of document analysis."""
    
    document_id: str
    document_type: str
    language: str
    readability_score: float
    sentence_count: int
    word_count: int
    character_count: int
    paragraph_count: int
    complexity_score: float
    medical_terminology_density: float
    structured_data_elements: Dict[str, int]
    potential_phi_sections: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "language": self.language,
            "readability_score": self.readability_score,
            "sentence_count": self.sentence_count,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "paragraph_count": self.paragraph_count,
            "complexity_score": self.complexity_score,
            "medical_terminology_density": self.medical_terminology_density,
            "structured_data_elements": self.structured_data_elements,
            "potential_phi_sections": self.potential_phi_sections,
            "metadata": self.metadata,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }


class DocumentAnalyzer:
    """Analyzes documents for structure, content, and potential PHI locations."""
    
    def __init__(self):
        """Initialize document analyzer."""
        # Medical terminology patterns
        self.medical_terms = {
            "anatomy": [
                "heart", "lung", "liver", "kidney", "brain", "spine", "bone",
                "muscle", "tissue", "organ", "vessel", "artery", "vein"
            ],
            "conditions": [
                "diabetes", "hypertension", "cancer", "infection", "disease",
                "syndrome", "disorder", "injury", "fracture", "lesion"
            ],
            "procedures": [
                "surgery", "operation", "procedure", "examination", "test",
                "scan", "x-ray", "mri", "ct", "ultrasound", "biopsy"
            ],
            "medications": [
                "medication", "drug", "prescription", "dosage", "treatment",
                "therapy", "antibiotic", "insulin", "vaccine"
            ]
        }
        
        # Structured data patterns
        self.structured_patterns = {
            "dates": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',
            "times": r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            "phone_numbers": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "medical_record_numbers": r'\b(?:MRN|MR|ID)[:=]?\s*\d+\b',
            "addresses": r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "urls": r'https?://[^\s]+',
            "dosages": r'\b\d+\s*(?:mg|mcg|g|ml|cc|units?)\b',
            "vital_signs": r'\b(?:BP|Blood Pressure)[:=]?\s*\d+/\d+\b|\b(?:HR|Heart Rate)[:=]?\s*\d+\b|\b(?:Temp|Temperature)[:=]?\s*\d+\.?\d*\b'
        }
        
        # PHI section indicators
        self.phi_section_indicators = [
            "patient information", "demographics", "contact information",
            "emergency contact", "insurance", "billing", "address",
            "phone", "social security", "date of birth", "dob"
        ]
        
        # Document type indicators
        self.document_type_patterns = {
            "clinical_note": ["chief complaint", "history of present illness", "physical examination", "assessment", "plan"],
            "lab_report": ["laboratory", "results", "reference range", "normal", "abnormal"],
            "radiology": ["impression", "findings", "comparison", "technique", "contrast"],
            "pathology": ["specimen", "gross description", "microscopic", "diagnosis"],
            "discharge_summary": ["admission", "discharge", "hospital course", "medications on discharge"],
            "operative_report": ["procedure", "surgeon", "anesthesia", "operative technique", "postoperative"],
            "consultation": ["consultant", "reason for consultation", "recommendations"],
            "medication_list": ["medications", "prescriptions", "dosage", "frequency"]
        }
    
    @trace_operation("document_analysis")
    def analyze_document(self, content: str, document_id: str = None, 
                        metadata: Dict[str, Any] = None) -> DocumentAnalysisResult:
        """Perform comprehensive document analysis.
        
        Args:
            content: Document content to analyze
            document_id: Optional document identifier
            metadata: Optional metadata about the document
            
        Returns:
            Document analysis result
        """
        logger.info(f"Starting document analysis for document: {document_id}")
        
        # Basic text statistics
        text_stats = self._calculate_text_statistics(content)
        
        # Language detection (simplified)
        language = self._detect_language(content)
        
        # Document type classification
        document_type = self._classify_document_type(content)
        
        # Readability analysis
        readability_score = self._calculate_readability(content, text_stats)
        
        # Complexity analysis
        complexity_score = self._calculate_complexity(content, text_stats)
        
        # Medical terminology analysis
        medical_density = self._analyze_medical_terminology(content)
        
        # Structured data extraction
        structured_elements = self._extract_structured_data(content)
        
        # PHI section identification
        phi_sections = self._identify_phi_sections(content)
        
        result = DocumentAnalysisResult(
            document_id=document_id or "unknown",
            document_type=document_type,
            language=language,
            readability_score=readability_score,
            sentence_count=text_stats["sentences"],
            word_count=text_stats["words"],
            character_count=text_stats["characters"],
            paragraph_count=text_stats["paragraphs"],
            complexity_score=complexity_score,
            medical_terminology_density=medical_density,
            structured_data_elements=structured_elements,
            potential_phi_sections=phi_sections,
            metadata=metadata or {}
        )
        
        logger.info(f"Document analysis completed: {document_type}, {text_stats['words']} words, {len(phi_sections)} PHI sections")
        return result
    
    def _calculate_text_statistics(self, content: str) -> Dict[str, int]:
        """Calculate basic text statistics."""
        # Count characters
        character_count = len(content)
        
        # Count words
        words = re.findall(r'\b\w+\b', content)
        word_count = len(words)
        
        # Count sentences
        sentences = re.split(r'[.!?]+', content)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Count paragraphs
        paragraphs = content.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        return {
            "characters": character_count,
            "words": word_count,
            "sentences": sentence_count,
            "paragraphs": paragraph_count
        }
    
    def _detect_language(self, content: str) -> str:
        """Detect document language (simplified implementation)."""
        # Simple heuristic - check for common English medical terms
        english_indicators = [
            "patient", "doctor", "hospital", "medical", "treatment", "diagnosis",
            "the", "and", "or", "is", "was", "were", "have", "has"
        ]
        
        content_lower = content.lower()
        english_count = sum(1 for term in english_indicators if term in content_lower)
        
        # Simple threshold-based detection
        if english_count >= 3:
            return "en"
        else:
            return "unknown"
    
    def _classify_document_type(self, content: str) -> str:
        """Classify document type based on content patterns."""
        content_lower = content.lower()
        
        type_scores = {}
        
        for doc_type, indicators in self.document_type_patterns.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            if score > 0:
                type_scores[doc_type] = score
        
        if type_scores:
            # Return type with highest score
            return max(type_scores.items(), key=lambda x: x[1])[0]
        else:
            return "generic_medical"
    
    def _calculate_readability(self, content: str, text_stats: Dict[str, int]) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)."""
        if text_stats["sentences"] == 0 or text_stats["words"] == 0:
            return 0.0
        
        # Count syllables (simplified)
        syllable_count = 0
        words = re.findall(r'\b\w+\b', content.lower())
        
        for word in words:
            # Simple syllable counting
            vowels = 'aeiouy'
            syllables = 0
            previous_was_vowel = False
            
            for char in word:
                if char in vowels:
                    if not previous_was_vowel:
                        syllables += 1
                    previous_was_vowel = True
                else:
                    previous_was_vowel = False
            
            # Ensure at least one syllable per word
            if syllables == 0:
                syllables = 1
            
            syllable_count += syllables
        
        # Flesch Reading Ease formula
        avg_sentence_length = text_stats["words"] / text_stats["sentences"]
        avg_syllables_per_word = syllable_count / text_stats["words"]
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Clamp between 0 and 100
        return max(0.0, min(100.0, score))
    
    def _calculate_complexity(self, content: str, text_stats: Dict[str, int]) -> float:
        """Calculate document complexity score."""
        if text_stats["words"] == 0:
            return 0.0
        
        complexity_factors = []
        
        # Average word length
        words = re.findall(r'\b\w+\b', content)
        avg_word_length = sum(len(word) for word in words) / len(words)
        complexity_factors.append(min(avg_word_length / 10.0, 1.0))
        
        # Sentence length variance
        sentences = re.split(r'[.!?]+', content)
        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences if s.strip()]
        
        if len(sentence_lengths) > 1:
            length_variance = statistics.variance(sentence_lengths)
            complexity_factors.append(min(length_variance / 100.0, 1.0))
        else:
            complexity_factors.append(0.0)
        
        # Medical terminology density (contributes to complexity)
        medical_density = self._analyze_medical_terminology(content)
        complexity_factors.append(medical_density)
        
        # Structured data density
        structured_count = sum(len(re.findall(pattern, content)) 
                             for pattern in self.structured_patterns.values())
        structured_density = min(structured_count / text_stats["words"], 1.0)
        complexity_factors.append(structured_density)
        
        # Average complexity score
        return sum(complexity_factors) / len(complexity_factors)
    
    def _analyze_medical_terminology(self, content: str) -> float:
        """Analyze density of medical terminology."""
        content_lower = content.lower()
        total_words = len(re.findall(r'\b\w+\b', content))
        
        if total_words == 0:
            return 0.0
        
        medical_word_count = 0
        
        for category, terms in self.medical_terms.items():
            for term in terms:
                medical_word_count += len(re.findall(r'\b' + re.escape(term) + r'\b', content_lower))
        
        return min(medical_word_count / total_words, 1.0)
    
    def _extract_structured_data(self, content: str) -> Dict[str, int]:
        """Extract and count structured data elements."""
        structured_elements = {}
        
        for element_type, pattern in self.structured_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            structured_elements[element_type] = len(matches)
        
        return structured_elements
    
    def _identify_phi_sections(self, content: str) -> List[str]:
        """Identify sections that likely contain PHI."""
        phi_sections = []
        content_lower = content.lower()
        
        # Split content into sections (by headers or line breaks)
        sections = re.split(r'\n\s*\n|\n(?=[A-Z][A-Z\s]+:)', content)
        
        for i, section in enumerate(sections):
            section_lower = section.lower()
            
            # Check for PHI indicators
            for indicator in self.phi_section_indicators:
                if indicator in section_lower:
                    phi_sections.append(f"Section {i+1}: {indicator}")
                    break
            
            # Check for high density of structured data
            structured_count = sum(len(re.findall(pattern, section)) 
                                 for pattern in self.structured_patterns.values())
            
            if structured_count > 2:  # Threshold for high structured data density
                phi_sections.append(f"Section {i+1}: high structured data density ({structured_count} elements)")
        
        return phi_sections
    
    def get_analysis_summary(self, results: List[DocumentAnalysisResult]) -> Dict[str, Any]:
        """Generate summary statistics from multiple analysis results."""
        if not results:
            return {}
        
        document_types = Counter(r.document_type for r in results)
        languages = Counter(r.language for r in results)
        
        readability_scores = [r.readability_score for r in results]
        complexity_scores = [r.complexity_score for r in results]
        word_counts = [r.word_count for r in results]
        medical_densities = [r.medical_terminology_density for r in results]
        
        return {
            "total_documents": len(results),
            "document_types": dict(document_types),
            "languages": dict(languages),
            "avg_readability_score": statistics.mean(readability_scores),
            "avg_complexity_score": statistics.mean(complexity_scores),
            "avg_word_count": statistics.mean(word_counts),
            "avg_medical_terminology_density": statistics.mean(medical_densities),
            "total_phi_sections": sum(len(r.potential_phi_sections) for r in results),
            "analysis_period": {
                "start": min(r.analysis_timestamp for r in results).isoformat(),
                "end": max(r.analysis_timestamp for r in results).isoformat()
            }
        }