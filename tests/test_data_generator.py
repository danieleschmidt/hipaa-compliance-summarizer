"""Test data generator for comprehensive testing scenarios."""

import random
import string
from datetime import datetime, timedelta
from typing import Dict, List


class SyntheticTestDataGenerator:
    """Generate synthetic healthcare data for testing purposes."""

    # Synthetic names and locations (clearly fake)
    FAKE_FIRST_NAMES = [
        "TestPatient", "MockUser", "SamplePerson", "DemoUser", "TestCase",
        "ExamplePatient", "FakeUser", "SyntheticPerson", "DummyPatient", "TestData"
    ]
    
    FAKE_LAST_NAMES = [
        "TestSuite", "MockData", "SampleSet", "DemoCase", "TestScenario",
        "ExampleData", "FakeEntry", "SyntheticCase", "DummyRecord", "TestInstance"
    ]
    
    FAKE_CITIES = [
        "TestCity", "MockTown", "SampleVille", "DemoPlace", "TestBurg",
        "ExampleCity", "FakeVille", "SyntheticTown", "DummyCity", "TestLocation"
    ]

    def __init__(self):
        """Initialize the test data generator."""
        self.seed = 12345  # Fixed seed for reproducible tests
        random.seed(self.seed)

    def generate_synthetic_mrn(self) -> str:
        """Generate a clearly synthetic Medical Record Number."""
        return f"TEST{random.randint(100000, 999999)}"

    def generate_synthetic_ssn(self) -> str:
        """Generate a clearly synthetic SSN (using invalid ranges)."""
        # Use 900-999 area numbers which are invalid for real SSNs
        area = random.randint(900, 999)
        group = random.randint(10, 99)
        serial = random.randint(1000, 9999)
        return f"{area}-{group:02d}-{serial:04d}"

    def generate_synthetic_phone(self) -> str:
        """Generate a clearly synthetic phone number."""
        # Use 555 prefix which is reserved for fictional use
        return f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

    def generate_synthetic_date(self, years_back: int = 50) -> str:
        """Generate a synthetic date within specified years back."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        return random_date.strftime("%m/%d/%Y")

    def generate_synthetic_address(self) -> Dict[str, str]:
        """Generate a clearly synthetic address."""
        return {
            "street": f"{random.randint(100, 9999)} Test Street",
            "city": random.choice(self.FAKE_CITIES),
            "state": "TS",  # Test State
            "zip": f"99{random.randint(100, 999)}"  # Invalid ZIP range
        }

    def generate_clinical_note(self, include_phi: bool = True) -> str:
        """Generate a synthetic clinical note."""
        patient_name = "TEST_PATIENT"
        mrn = "TEST000000"
        dob = "01/01/1900"
        
        if include_phi:
            first_name = random.choice(self.FAKE_FIRST_NAMES)
            last_name = random.choice(self.FAKE_LAST_NAMES)
            patient_name = f"{first_name} {last_name}"
            mrn = self.generate_synthetic_mrn()
            dob = self.generate_synthetic_date(80)

        return f"""
CLINICAL NOTE

Patient: {patient_name}
DOB: {dob}
MRN: {mrn}
Date: {datetime.now().strftime('%B %d, %Y')}

CHIEF COMPLAINT: Synthetic test complaint for automated testing

HISTORY OF PRESENT ILLNESS:
This is a synthetic patient case generated for testing purposes only.
All data is completely artificial and not derived from real patients.

ASSESSMENT AND PLAN:
1. This is test data for HIPAA compliance validation
2. No real patient information is contained herein
3. All identifiers are clearly synthetic and fictional

Provider: Dr. Test Physician
"""

    def generate_lab_report(self, include_phi: bool = True) -> str:
        """Generate a synthetic lab report."""
        patient_name = "TEST_PATIENT"
        ssn = "999-99-9999"
        account = "TEST000000"
        
        if include_phi:
            first_name = random.choice(self.FAKE_FIRST_NAMES)
            last_name = random.choice(self.FAKE_LAST_NAMES)
            patient_name = f"{first_name} {last_name}"
            ssn = self.generate_synthetic_ssn()
            account = f"LAB{random.randint(100000, 999999)}"

        return f"""
LABORATORY REPORT

Patient: {patient_name}
DOB: {self.generate_synthetic_date(80)}
SSN: {ssn}
Account: {account}
Test Date: {datetime.now().strftime('%m/%d/%Y')}

CHEMISTRY PANEL:
- Glucose: {random.randint(70, 140)} mg/dL
- BUN: {random.randint(7, 25)} mg/dL
- Creatinine: {random.uniform(0.5, 1.5):.1f} mg/dL
- Sodium: {random.randint(135, 145)} mEq/L

INTERPRETATION: All values within normal synthetic testing ranges
Note: This is synthetic test data for compliance validation
"""

    def generate_test_dataset(self, size: int = 10, phi_ratio: float = 0.7) -> List[Dict]:
        """Generate a dataset for testing with specified PHI ratio."""
        dataset = []
        phi_count = int(size * phi_ratio)
        
        for i in range(size):
            include_phi = i < phi_count
            doc_type = random.choice(["clinical_note", "lab_report"])
            
            if doc_type == "clinical_note":
                content = self.generate_clinical_note(include_phi)
            else:
                content = self.generate_lab_report(include_phi)
            
            dataset.append({
                "id": f"test_doc_{i:03d}",
                "type": doc_type,
                "content": content,
                "has_phi": include_phi,
                "expected_phi_count": random.randint(3, 8) if include_phi else 0
            })
        
        return dataset

    def generate_edge_case_documents(self) -> List[Dict]:
        """Generate edge case documents for comprehensive testing."""
        edge_cases = []
        
        # Empty document
        edge_cases.append({
            "id": "edge_empty",
            "type": "empty",
            "content": "",
            "expected_phi_count": 0
        })
        
        # Document with only whitespace
        edge_cases.append({
            "id": "edge_whitespace",
            "type": "whitespace",
            "content": "   \n\t  \n  ",
            "expected_phi_count": 0
        })
        
        # Document with special characters
        edge_cases.append({
            "id": "edge_special_chars",
            "type": "special",
            "content": "Patient: TestUser@#$% MRN: 123-ABC-789!@# DOB: ??/??/????",
            "expected_phi_count": 3
        })
        
        # Very long document
        long_content = "This is a test document. " * 1000
        long_content += f"Patient: {random.choice(self.FAKE_FIRST_NAMES)} TestCase"
        edge_cases.append({
            "id": "edge_long_doc",
            "type": "long",
            "content": long_content,
            "expected_phi_count": 1
        })
        
        return edge_cases


def create_performance_test_data(count: int = 1000) -> List[str]:
    """Create large dataset for performance testing."""
    generator = SyntheticTestDataGenerator()
    documents = []
    
    for i in range(count):
        if i % 2 == 0:
            content = generator.generate_clinical_note(True)
        else:
            content = generator.generate_lab_report(True)
        documents.append(content)
    
    return documents