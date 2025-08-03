"""Cloud storage and service integrations."""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, BinaryIO
from dataclasses import dataclass
from abc import ABC, abstractmethod
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import json
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CloudDocument:
    """Represents a document stored in cloud storage."""
    
    key: str
    bucket: str
    size: int
    last_modified: datetime
    metadata: Dict[str, str]
    storage_class: str = "STANDARD"
    encryption: Optional[str] = None
    
    def get_full_path(self) -> str:
        """Get full cloud storage path."""
        return f"{self.bucket}/{self.key}"


class CloudStorageBase(ABC):
    """Base class for cloud storage integrations."""
    
    def __init__(self, provider_name: str):
        """Initialize cloud storage integration.
        
        Args:
            provider_name: Name of the cloud provider
        """
        self.provider_name = provider_name
    
    @abstractmethod
    def upload_document(self, file_path: str, key: str, metadata: Dict[str, str] = None) -> bool:
        """Upload a document to cloud storage."""
        pass
    
    @abstractmethod
    def download_document(self, key: str, local_path: str) -> bool:
        """Download a document from cloud storage."""
        pass
    
    @abstractmethod
    def list_documents(self, prefix: str = "", limit: int = 1000) -> List[CloudDocument]:
        """List documents in cloud storage."""
        pass
    
    @abstractmethod
    def delete_document(self, key: str) -> bool:
        """Delete a document from cloud storage."""
        pass
    
    @abstractmethod
    def get_document_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a specific document."""
        pass


class AWSStorageIntegration(CloudStorageBase):
    """AWS S3 storage integration with HIPAA compliance features."""
    
    def __init__(self):
        """Initialize AWS S3 integration."""
        super().__init__("AWS S3")
        
        self.region = os.getenv("AWS_REGION", "us-west-2")
        self.bucket_name = os.getenv("AWS_S3_BUCKET", "hipaa-compliance-documents")
        self.kms_key_id = os.getenv("AWS_KMS_KEY_ID")  # For encryption
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"AWS S3 integration initialized - bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            self.s3_client = None
        except ClientError as e:
            logger.error(f"AWS S3 initialization failed: {e}")
            self.s3_client = None
    
    def upload_document(self, file_path: str, key: str, metadata: Dict[str, str] = None) -> bool:
        """Upload document to S3 with encryption."""
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        try:
            # Prepare upload arguments
            upload_args = {
                'Bucket': self.bucket_name,
                'Key': key,
                'Filename': file_path
            }
            
            # Add metadata
            if metadata:
                # S3 metadata keys must be lowercase
                s3_metadata = {k.lower().replace('_', '-'): str(v) for k, v in metadata.items()}
                upload_args['ExtraArgs'] = {'Metadata': s3_metadata}
            else:
                upload_args['ExtraArgs'] = {}
            
            # Add HIPAA compliance settings
            upload_args['ExtraArgs'].update({
                'ServerSideEncryption': 'aws:kms' if self.kms_key_id else 'AES256',
                'StorageClass': 'STANDARD_IA',  # Cost-effective for infrequent access
                'Tagging': 'Classification=PHI&Compliance=HIPAA'
            })
            
            if self.kms_key_id:
                upload_args['ExtraArgs']['SSEKMSKeyId'] = self.kms_key_id
            
            # Upload file
            self.s3_client.upload_file(**upload_args)
            
            logger.info(f"Document uploaded to S3: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload document to S3: {e}")
            return False
    
    def download_document(self, key: str, local_path: str) -> bool:
        """Download document from S3."""
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        try:
            self.s3_client.download_file(
                Bucket=self.bucket_name,
                Key=key,
                Filename=local_path
            )
            
            logger.info(f"Document downloaded from S3: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download document from S3: {e}")
            return False
    
    def list_documents(self, prefix: str = "", limit: int = 1000) -> List[CloudDocument]:
        """List documents in S3 bucket."""
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return []
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=limit
            )
            
            documents = []
            for obj in response.get('Contents', []):
                # Get additional metadata
                try:
                    head_response = self.s3_client.head_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )
                    metadata = head_response.get('Metadata', {})
                    encryption = head_response.get('ServerSideEncryption')
                except Exception as e:
                    logger.warning(f"Failed to get metadata for {obj['Key']}: {e}")
                    metadata = {}
                    encryption = None
                
                doc = CloudDocument(
                    key=obj['Key'],
                    bucket=self.bucket_name,
                    size=obj['Size'],
                    last_modified=obj['LastModified'],
                    metadata=metadata,
                    storage_class=obj.get('StorageClass', 'STANDARD'),
                    encryption=encryption
                )
                documents.append(doc)
            
            logger.info(f"Listed {len(documents)} documents from S3")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list S3 documents: {e}")
            return []
    
    def delete_document(self, key: str) -> bool:
        """Delete document from S3."""
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
        
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=key
            )
            
            logger.info(f"Document deleted from S3: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document from S3: {e}")
            return False
    
    def get_document_metadata(self, key: str) -> Dict[str, Any]:
        """Get detailed metadata for S3 document."""
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return {}
        
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=key
            )
            
            return {
                'key': key,
                'bucket': self.bucket_name,
                'size': response.get('ContentLength'),
                'last_modified': response.get('LastModified'),
                'content_type': response.get('ContentType'),
                'encryption': response.get('ServerSideEncryption'),
                'metadata': response.get('Metadata', {}),
                'storage_class': response.get('StorageClass'),
                'etag': response.get('ETag')
            }
            
        except Exception as e:
            logger.error(f"Failed to get S3 document metadata: {e}")
            return {}
    
    def create_presigned_url(self, key: str, expiration: int = 3600) -> Optional[str]:
        """Create a presigned URL for secure document access."""
        if not self.s3_client:
            return None
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for {key}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    def setup_bucket_compliance(self) -> bool:
        """Setup HIPAA compliance settings for S3 bucket."""
        if not self.s3_client:
            return False
        
        try:
            # Enable versioning
            self.s3_client.put_bucket_versioning(
                Bucket=self.bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            # Setup lifecycle configuration
            lifecycle_config = {
                'Rules': [
                    {
                        'ID': 'HIPAACompliance',
                        'Status': 'Enabled',
                        'Transitions': [
                            {
                                'Days': 30,
                                'StorageClass': 'STANDARD_IA'
                            },
                            {
                                'Days': 90,
                                'StorageClass': 'GLACIER'
                            },
                            {
                                'Days': 2555,  # 7 years
                                'StorageClass': 'DEEP_ARCHIVE'
                            }
                        ],
                        'Expiration': {
                            'Days': 2555  # 7 years retention
                        }
                    }
                ]
            }
            
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name,
                LifecycleConfiguration=lifecycle_config
            )
            
            # Setup bucket encryption
            encryption_config = {
                'Rules': [
                    {
                        'ApplyServerSideEncryptionByDefault': {
                            'SSEAlgorithm': 'aws:kms' if self.kms_key_id else 'AES256'
                        },
                        'BucketKeyEnabled': True
                    }
                ]
            }
            
            if self.kms_key_id:
                encryption_config['Rules'][0]['ApplyServerSideEncryptionByDefault']['KMSMasterKeyID'] = self.kms_key_id
            
            self.s3_client.put_bucket_encryption(
                Bucket=self.bucket_name,
                ServerSideEncryptionConfiguration=encryption_config
            )
            
            logger.info(f"HIPAA compliance settings applied to bucket: {self.bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup bucket compliance: {e}")
            return False


class AzureStorageIntegration(CloudStorageBase):
    """Azure Blob Storage integration (placeholder implementation)."""
    
    def __init__(self):
        """Initialize Azure Blob Storage integration."""
        super().__init__("Azure Blob Storage")
        
        # Azure configuration
        self.account_name = os.getenv("AZURE_STORAGE_ACCOUNT")
        self.account_key = os.getenv("AZURE_STORAGE_KEY")
        self.container_name = os.getenv("AZURE_CONTAINER", "hipaa-documents")
        
        # Note: In production, implement actual Azure SDK integration
        logger.info("Azure Storage integration initialized (placeholder)")
    
    def upload_document(self, file_path: str, key: str, metadata: Dict[str, str] = None) -> bool:
        """Upload document to Azure Blob Storage (placeholder)."""
        logger.info(f"Azure upload simulated: {key}")
        return True
    
    def download_document(self, key: str, local_path: str) -> bool:
        """Download document from Azure Blob Storage (placeholder)."""
        logger.info(f"Azure download simulated: {key}")
        return True
    
    def list_documents(self, prefix: str = "", limit: int = 1000) -> List[CloudDocument]:
        """List documents in Azure container (placeholder)."""
        logger.info("Azure list documents simulated")
        return []
    
    def delete_document(self, key: str) -> bool:
        """Delete document from Azure Blob Storage (placeholder)."""
        logger.info(f"Azure delete simulated: {key}")
        return True
    
    def get_document_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for Azure blob (placeholder)."""
        logger.info(f"Azure metadata retrieval simulated: {key}")
        return {}


class CloudStorageManager:
    """Manager for multiple cloud storage providers."""
    
    def __init__(self):
        """Initialize cloud storage manager."""
        self.providers = {}
        self._initialize_providers()
        self.default_provider = self._determine_default_provider()
    
    def _initialize_providers(self):
        """Initialize available cloud storage providers."""
        # AWS S3
        if os.getenv("AWS_ACCESS_KEY_ID"):
            aws_provider = AWSStorageIntegration()
            if aws_provider.s3_client:
                self.providers["aws"] = aws_provider
        
        # Azure Blob Storage
        if os.getenv("AZURE_STORAGE_ACCOUNT"):
            self.providers["azure"] = AzureStorageIntegration()
        
        logger.info(f"Initialized {len(self.providers)} cloud storage providers: {list(self.providers.keys())}")
    
    def _determine_default_provider(self) -> Optional[str]:
        """Determine the default cloud storage provider."""
        # Prefer AWS if available
        if "aws" in self.providers:
            return "aws"
        elif "azure" in self.providers:
            return "azure"
        return None
    
    def get_provider(self, provider_name: str = None) -> Optional[CloudStorageBase]:
        """Get a specific cloud storage provider."""
        if provider_name is None:
            provider_name = self.default_provider
        
        return self.providers.get(provider_name)
    
    def upload_document_to_cloud(self, file_path: str, document_id: str,
                                provider: str = None, metadata: Dict[str, str] = None) -> bool:
        """Upload document to cloud storage with HIPAA compliance."""
        storage_provider = self.get_provider(provider)
        if not storage_provider:
            logger.error(f"Cloud storage provider not available: {provider}")
            return False
        
        # Generate cloud storage key with compliance structure
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")
        key = f"hipaa-documents/{timestamp}/{document_id}"
        
        # Add compliance metadata
        compliance_metadata = {
            "document-id": document_id,
            "upload-timestamp": datetime.utcnow().isoformat(),
            "classification": "phi",
            "compliance": "hipaa",
            "retention-years": "7"
        }
        
        if metadata:
            compliance_metadata.update(metadata)
        
        return storage_provider.upload_document(file_path, key, compliance_metadata)
    
    def download_document_from_cloud(self, document_id: str, local_path: str,
                                   provider: str = None) -> bool:
        """Download document from cloud storage."""
        storage_provider = self.get_provider(provider)
        if not storage_provider:
            logger.error(f"Cloud storage provider not available: {provider}")
            return False
        
        # Try to find document by searching with document_id
        documents = storage_provider.list_documents(prefix="hipaa-documents/")
        
        target_key = None
        for doc in documents:
            if document_id in doc.key:
                target_key = doc.key
                break
        
        if not target_key:
            logger.error(f"Document not found in cloud storage: {document_id}")
            return False
        
        return storage_provider.download_document(target_key, local_path)
    
    def list_all_documents(self, provider: str = None) -> List[CloudDocument]:
        """List all documents across cloud providers."""
        if provider:
            storage_provider = self.get_provider(provider)
            if storage_provider:
                return storage_provider.list_documents(prefix="hipaa-documents/")
            return []
        
        # List from all providers
        all_documents = []
        for provider_name, storage_provider in self.providers.items():
            try:
                documents = storage_provider.list_documents(prefix="hipaa-documents/")
                # Add provider info to each document
                for doc in documents:
                    doc.metadata["provider"] = provider_name
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to list documents from {provider_name}: {e}")
        
        return all_documents
    
    def delete_document_from_cloud(self, document_id: str, provider: str = None) -> bool:
        """Delete document from cloud storage."""
        storage_provider = self.get_provider(provider)
        if not storage_provider:
            logger.error(f"Cloud storage provider not available: {provider}")
            return False
        
        # Find and delete document
        documents = storage_provider.list_documents(prefix="hipaa-documents/")
        
        for doc in documents:
            if document_id in doc.key:
                return storage_provider.delete_document(doc.key)
        
        logger.error(f"Document not found for deletion: {document_id}")
        return False
    
    def setup_compliance_policies(self, provider: str = None) -> bool:
        """Setup HIPAA compliance policies across cloud providers."""
        if provider:
            storage_provider = self.get_provider(provider)
            if isinstance(storage_provider, AWSStorageIntegration):
                return storage_provider.setup_bucket_compliance()
            return True
        
        # Setup compliance for all providers
        success = True
        for provider_name, storage_provider in self.providers.items():
            try:
                if isinstance(storage_provider, AWSStorageIntegration):
                    if not storage_provider.setup_bucket_compliance():
                        success = False
                # Add Azure compliance setup when implemented
                
            except Exception as e:
                logger.error(f"Failed to setup compliance for {provider_name}: {e}")
                success = False
        
        return success
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get summary of cloud storage usage."""
        summary = {
            "providers": list(self.providers.keys()),
            "default_provider": self.default_provider,
            "total_documents": 0,
            "total_size_bytes": 0,
            "provider_details": {}
        }
        
        for provider_name, storage_provider in self.providers.items():
            try:
                documents = storage_provider.list_documents(prefix="hipaa-documents/")
                total_size = sum(doc.size for doc in documents)
                
                summary["provider_details"][provider_name] = {
                    "document_count": len(documents),
                    "total_size_bytes": total_size,
                    "provider_type": storage_provider.provider_name
                }
                
                summary["total_documents"] += len(documents)
                summary["total_size_bytes"] += total_size
                
            except Exception as e:
                logger.error(f"Failed to get summary for {provider_name}: {e}")
                summary["provider_details"][provider_name] = {
                    "error": str(e)
                }
        
        return summary