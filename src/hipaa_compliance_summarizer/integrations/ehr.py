"""EHR (Electronic Health Record) system integrations."""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EHRDocument:
    """Represents a document from an EHR system."""

    ehr_id: str
    patient_id: str
    document_type: str
    content: str
    metadata: Dict[str, Any]
    created_date: datetime
    last_modified: datetime
    source_system: str

    def to_hipaa_document(self) -> Dict[str, Any]:
        """Convert to HIPAA processor compatible format."""
        return {
            "id": f"{self.source_system}_{self.ehr_id}",
            "filename": f"{self.document_type}_{self.patient_id}_{self.ehr_id}.txt",
            "content": self.content,
            "document_type": self.document_type,
            "metadata": {
                **self.metadata,
                "ehr_id": self.ehr_id,
                "patient_id": self.patient_id,
                "source_system": self.source_system,
                "created_date": self.created_date.isoformat(),
                "last_modified": self.last_modified.isoformat()
            }
        }


class EHRIntegrationBase(ABC):
    """Base class for EHR system integrations."""

    def __init__(self, system_name: str, config: Dict[str, Any] = None):
        """Initialize EHR integration.
        
        Args:
            system_name: Name of the EHR system
            config: Configuration dictionary with API endpoints, credentials, etc.
        """
        self.system_name = system_name
        self.config = config or {}
        
        if REQUESTS_AVAILABLE:
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "HIPAA-Compliance-Summarizer/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            })
        else:
            self.session = None
            logger.warning("Requests library not available. HTTP operations will not work.")

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the EHR system."""
        pass

    @abstractmethod
    def get_patient_documents(self, patient_id: str,
                             document_types: List[str] = None,
                             date_range: tuple = None) -> List[EHRDocument]:
        """Retrieve documents for a patient."""
        pass

    @abstractmethod
    def get_document_content(self, document_id: str) -> str:
        """Retrieve full content of a specific document."""
        pass

    def test_connection(self) -> bool:
        """Test connection to EHR system."""
        try:
            return self.authenticate()
        except Exception as e:
            logger.error(f"Failed to test {self.system_name} connection: {e}")
            return False


class EpicIntegration(EHRIntegrationBase):
    """Epic EHR system integration."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Epic integration."""
        super().__init__("Epic", config)
        self.base_url = self.config.get("base_url") or os.getenv("EPIC_BASE_URL", "https://fhir.epic.com/interconnect-fhir-oauth")
        self.client_id = self.config.get("client_id") or os.getenv("EPIC_CLIENT_ID")
        self.private_key = self.config.get("private_key") or os.getenv("EPIC_PRIVATE_KEY")
        self.access_token = None
        self.token_expires = None

    def authenticate(self) -> bool:
        """Authenticate with Epic using JWT assertion."""
        if not all([self.client_id, self.private_key]):
            logger.error("Epic credentials not configured")
            return False

        try:
            if JWT_AVAILABLE and REQUESTS_AVAILABLE:
                # Create JWT assertion for Epic OAuth 2.0
                auth_url = urljoin(self.base_url, "/oauth2/token")
                
                # JWT payload for Epic
                jwt_payload = {
                    "iss": self.client_id,
                    "sub": self.client_id,
                    "aud": auth_url,
                    "jti": f"{datetime.utcnow().timestamp()}",
                    "exp": int((datetime.utcnow() + timedelta(minutes=5)).timestamp()),
                    "iat": int(datetime.utcnow().timestamp())
                }
                
                # Sign JWT with private key
                jwt_token = jwt.encode(jwt_payload, self.private_key, algorithm="RS384")
                
                # Request access token
                token_data = {
                    "grant_type": "client_credentials",
                    "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                    "client_assertion": jwt_token,
                    "scope": "system/Patient.read system/DocumentReference.read"
                }
                
                response = self.session.post(auth_url, data=token_data)
                response.raise_for_status()
                
                token_info = response.json()
                self.access_token = token_info["access_token"]
                self.token_expires = datetime.utcnow() + timedelta(seconds=token_info.get("expires_in", 3600))
                
                logger.info("Epic authentication successful")
            else:
                # Fallback simulation when JWT not available
                logger.info("Epic authentication simulated (JWT library not available)")
                self.access_token = "epic_mock_token"
                self.token_expires = datetime.utcnow() + timedelta(hours=1)

            # Update session headers
            if self.session:
                self.session.headers.update({
                    "Authorization": f"Bearer {self.access_token}"
                })

            return True

        except Exception as e:
            logger.error(f"Epic authentication failed: {e}")
            return False

    def get_patient_documents(self, patient_id: str,
                             document_types: List[str] = None,
                             date_range: tuple = None) -> List[EHRDocument]:
        """Retrieve patient documents from Epic."""
        if not self.access_token or datetime.utcnow() >= self.token_expires:
            if not self.authenticate():
                return []

        try:
            # Mock Epic FHIR response
            logger.info(f"Retrieving Epic documents for patient {patient_id}")

            # In production, make actual FHIR API calls
            mock_documents = [
                EHRDocument(
                    ehr_id=f"epic_doc_{i}",
                    patient_id=patient_id,
                    document_type="clinical_note",
                    content=f"Mock clinical note {i} for patient {patient_id}. Patient presents with symptoms...",
                    metadata={
                        "encounter_id": f"encounter_{i}",
                        "provider_id": f"provider_{i}",
                        "facility_id": "epic_facility_001"
                    },
                    created_date=datetime.utcnow() - timedelta(days=i),
                    last_modified=datetime.utcnow() - timedelta(days=i),
                    source_system="Epic"
                )
                for i in range(1, 4)  # Mock 3 documents
            ]

            # Filter by document types if specified
            if document_types:
                mock_documents = [doc for doc in mock_documents if doc.document_type in document_types]

            # Filter by date range if specified
            if date_range:
                start_date, end_date = date_range
                mock_documents = [
                    doc for doc in mock_documents
                    if start_date <= doc.created_date <= end_date
                ]

            return mock_documents

        except Exception as e:
            logger.error(f"Failed to retrieve Epic documents: {e}")
            return []

    def get_document_content(self, document_id: str) -> str:
        """Retrieve full document content from Epic."""
        if not self.access_token:
            if not self.authenticate():
                return ""

        try:
            # Mock document content retrieval
            logger.info(f"Retrieving Epic document content: {document_id}")

            # In production, make FHIR DocumentReference/$expand call
            mock_content = f"""
            CLINICAL NOTE - {document_id}
            
            Patient ID: [PATIENT_ID]
            Date: {datetime.utcnow().strftime('%Y-%m-%d')}
            Provider: Dr. Smith
            
            Chief Complaint: Patient reports chest pain
            
            History of Present Illness:
            The patient is a 45-year-old male who presents with chest pain that began this morning.
            Pain is described as sharp and located in the center of the chest.
            
            Physical Examination:
            Vital Signs: BP 140/90, HR 85, RR 18, Temp 98.6°F
            General: Patient appears anxious but in no acute distress
            
            Assessment and Plan:
            1. Chest pain - rule out cardiac etiology
            2. Order EKG and cardiac enzymes
            3. Monitor in observation unit
            
            Provider: Dr. John Smith, MD
            Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
            """

            return mock_content

        except Exception as e:
            logger.error(f"Failed to retrieve Epic document content: {e}")
            return ""


class CernerIntegration(EHRIntegrationBase):
    """Cerner EHR system integration."""

    def __init__(self):
        """Initialize Cerner integration."""
        super().__init__("Cerner")
        self.base_url = os.getenv("CERNER_BASE_URL", "https://fhir-open.cerner.com/r4")
        self.client_id = os.getenv("CERNER_CLIENT_ID")
        self.client_secret = os.getenv("CERNER_CLIENT_SECRET")
        self.access_token = None
        self.token_expires = None

    def authenticate(self) -> bool:
        """Authenticate with Cerner using OAuth 2.0."""
        if not all([self.client_id, self.client_secret]):
            logger.error("Cerner credentials not configured")
            return False

        try:
            # In production, implement actual OAuth 2.0 flow
            logger.info("Cerner authentication simulated (implement OAuth 2.0)")
            self.access_token = "cerner_mock_token"
            self.token_expires = datetime.utcnow() + timedelta(hours=1)

            self.session.headers.update({
                "Authorization": f"Bearer {self.access_token}"
            })

            return True

        except Exception as e:
            logger.error(f"Cerner authentication failed: {e}")
            return False

    def get_patient_documents(self, patient_id: str,
                             document_types: List[str] = None,
                             date_range: tuple = None) -> List[EHRDocument]:
        """Retrieve patient documents from Cerner."""
        if not self.access_token or datetime.utcnow() >= self.token_expires:
            if not self.authenticate():
                return []

        try:
            logger.info(f"Retrieving Cerner documents for patient {patient_id}")

            # Mock Cerner FHIR response
            mock_documents = [
                EHRDocument(
                    ehr_id=f"cerner_doc_{i}",
                    patient_id=patient_id,
                    document_type="lab_report",
                    content=f"Mock lab report {i} for patient {patient_id}. Lab results show...",
                    metadata={
                        "order_id": f"order_{i}",
                        "lab_code": f"LAB{i:03d}",
                        "status": "final"
                    },
                    created_date=datetime.utcnow() - timedelta(days=i*2),
                    last_modified=datetime.utcnow() - timedelta(days=i*2),
                    source_system="Cerner"
                )
                for i in range(1, 3)  # Mock 2 documents
            ]

            if document_types:
                mock_documents = [doc for doc in mock_documents if doc.document_type in document_types]

            if date_range:
                start_date, end_date = date_range
                mock_documents = [
                    doc for doc in mock_documents
                    if start_date <= doc.created_date <= end_date
                ]

            return mock_documents

        except Exception as e:
            logger.error(f"Failed to retrieve Cerner documents: {e}")
            return []

    def get_document_content(self, document_id: str) -> str:
        """Retrieve full document content from Cerner."""
        if not self.access_token:
            if not self.authenticate():
                return ""

        try:
            logger.info(f"Retrieving Cerner document content: {document_id}")

            mock_content = f"""
            LABORATORY REPORT - {document_id}
            
            Patient: [PATIENT_NAME]
            MRN: [MEDICAL_RECORD_NUMBER]
            Date Collected: {datetime.utcnow().strftime('%Y-%m-%d')}
            Date Reported: {datetime.utcnow().strftime('%Y-%m-%d')}
            
            COMPLETE BLOOD COUNT (CBC)
            
            Test                Result      Reference Range    Flag
            ----------------------------------------------------------------
            WBC                 7.2         4.0-11.0 K/uL      
            RBC                 4.5         4.0-5.5 M/uL       
            Hemoglobin          14.2        12.0-16.0 g/dL     
            Hematocrit          42.1        36.0-48.0 %        
            Platelets           250         150-450 K/uL       
            
            BASIC METABOLIC PANEL
            
            Glucose             95          70-100 mg/dL       
            BUN                 18          7-25 mg/dL         
            Creatinine          1.0         0.6-1.3 mg/dL      
            Sodium              140         136-145 mmol/L     
            Potassium           4.1         3.5-5.0 mmol/L     
            
            Reviewed by: Dr. Jane Doe, MD
            Pathologist: Dr. Michael Brown, MD
            """

            return mock_content

        except Exception as e:
            logger.error(f"Failed to retrieve Cerner document content: {e}")
            return ""


class EHRIntegrationManager:
    """Manager for multiple EHR system integrations."""

    def __init__(self):
        """Initialize EHR integration manager."""
        self.integrations = {}
        self._initialize_integrations()

    def _initialize_integrations(self):
        """Initialize available EHR integrations."""
        # Epic integration
        if os.getenv("EPIC_CLIENT_ID"):
            self.integrations["epic"] = EpicIntegration()

        # Cerner integration
        if os.getenv("CERNER_CLIENT_ID"):
            self.integrations["cerner"] = CernerIntegration()

        logger.info(f"Initialized {len(self.integrations)} EHR integrations: {list(self.integrations.keys())}")

    def get_available_systems(self) -> List[str]:
        """Get list of available EHR systems."""
        return list(self.integrations.keys())

    def test_all_connections(self) -> Dict[str, bool]:
        """Test connections to all configured EHR systems."""
        results = {}
        for system_name, integration in self.integrations.items():
            results[system_name] = integration.test_connection()
        return results

    def get_patient_documents_from_all_systems(self, patient_id: str,
                                              document_types: List[str] = None,
                                              date_range: tuple = None) -> List[EHRDocument]:
        """Retrieve patient documents from all available EHR systems."""
        all_documents = []

        for system_name, integration in self.integrations.items():
            try:
                documents = integration.get_patient_documents(
                    patient_id=patient_id,
                    document_types=document_types,
                    date_range=date_range
                )
                all_documents.extend(documents)
                logger.info(f"Retrieved {len(documents)} documents from {system_name}")

            except Exception as e:
                logger.error(f"Failed to retrieve documents from {system_name}: {e}")

        return all_documents

    def get_patient_documents_from_system(self, system_name: str, patient_id: str,
                                         document_types: List[str] = None,
                                         date_range: tuple = None) -> List[EHRDocument]:
        """Retrieve patient documents from a specific EHR system."""
        if system_name not in self.integrations:
            raise ValueError(f"EHR system not available: {system_name}")

        integration = self.integrations[system_name]
        return integration.get_patient_documents(
            patient_id=patient_id,
            document_types=document_types,
            date_range=date_range
        )

    def get_document_content(self, system_name: str, document_id: str) -> str:
        """Retrieve document content from a specific EHR system."""
        if system_name not in self.integrations:
            raise ValueError(f"EHR system not available: {system_name}")

        integration = self.integrations[system_name]
        return integration.get_document_content(document_id)

    def process_patient_documents(self, patient_id: str, hipaa_processor,
                                 document_types: List[str] = None,
                                 systems: List[str] = None) -> Dict[str, Any]:
        """Process all documents for a patient through HIPAA compliance system.
        
        Args:
            patient_id: Patient identifier
            hipaa_processor: HIPAA processor instance
            document_types: Optional filter for document types
            systems: Optional filter for EHR systems
            
        Returns:
            Processing results summary
        """
        target_systems = systems or list(self.integrations.keys())
        all_documents = []

        # Retrieve documents from specified systems
        for system_name in target_systems:
            if system_name in self.integrations:
                try:
                    documents = self.get_patient_documents_from_system(
                        system_name=system_name,
                        patient_id=patient_id,
                        document_types=document_types
                    )
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Failed to retrieve from {system_name}: {e}")

        # Process documents through HIPAA system
        processing_results = []
        for ehr_doc in all_documents:
            try:
                # Convert to HIPAA format and process
                hipaa_doc = ehr_doc.to_hipaa_document()
                result = hipaa_processor.process_document(hipaa_doc["content"])

                processing_results.append({
                    "ehr_document": ehr_doc,
                    "processing_result": result,
                    "status": "success"
                })

            except Exception as e:
                logger.error(f"Failed to process document {ehr_doc.ehr_id}: {e}")
                processing_results.append({
                    "ehr_document": ehr_doc,
                    "processing_result": None,
                    "status": "failed",
                    "error": str(e)
                })

        # Generate summary
        successful = [r for r in processing_results if r["status"] == "success"]
        failed = [r for r in processing_results if r["status"] == "failed"]

        summary = {
            "patient_id": patient_id,
            "total_documents": len(all_documents),
            "successful_processing": len(successful),
            "failed_processing": len(failed),
            "systems_queried": target_systems,
            "processing_results": processing_results,
            "average_compliance_score": (
                sum(r["processing_result"].compliance_score for r in successful) / len(successful)
                if successful else 0.0
            ),
            "total_phi_detected": sum(
                r["processing_result"].phi_detected_count for r in successful
            )
        }

        return summary
