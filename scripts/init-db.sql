-- HIPAA Compliance Summarizer Database Initialization
-- This script initializes the PostgreSQL database for HIPAA compliance tracking

-- Create database if it doesn't exist (handled by Docker)
-- Note: This runs as part of docker-entrypoint-initdb.d

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create application user with limited privileges
-- Note: Main user 'hipaa' is created by Docker environment variables
-- This creates additional service accounts if needed

-- Grant necessary permissions
GRANT CONNECT ON DATABASE hipaa_compliance TO hipaa;
GRANT USAGE ON SCHEMA public TO hipaa;
GRANT CREATE ON SCHEMA public TO hipaa;

-- Create audit schema for compliance logging
CREATE SCHEMA IF NOT EXISTS audit;
GRANT USAGE ON SCHEMA audit TO hipaa;
GRANT CREATE ON SCHEMA audit TO hipaa;

-- Document processing tracking table
CREATE TABLE IF NOT EXISTS document_processing (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id VARCHAR(255) NOT NULL,
    filename VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) NOT NULL, -- SHA-256 hash for integrity
    processing_started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, processing, completed, failed
    phi_entities_detected INTEGER DEFAULT 0,
    redaction_count INTEGER DEFAULT 0,
    compliance_score DECIMAL(5,4), -- 0.0000 to 1.0000
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- PHI detection results table
CREATE TABLE IF NOT EXISTS phi_detection_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_processing_id UUID NOT NULL REFERENCES document_processing(id) ON DELETE CASCADE,
    entity_type VARCHAR(50) NOT NULL, -- NAME, SSN, DOB, etc.
    entity_text_hash VARCHAR(64) NOT NULL, -- Hashed PHI text for tracking without storing
    confidence_score DECIMAL(5,4) NOT NULL,
    position_start INTEGER,
    position_end INTEGER,
    redaction_method VARCHAR(50), -- masking, synthetic_replacement, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Compliance audit log table (in audit schema)
CREATE TABLE IF NOT EXISTS audit.compliance_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL, -- PHI_ACCESS, REDACTION, EXPORT, etc.
    document_processing_id UUID REFERENCES document_processing(id),
    user_id VARCHAR(255), -- User or service account
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    event_details JSONB, -- Flexible storage for event-specific data
    ip_address INET,
    user_agent TEXT,
    compliance_status VARCHAR(50), -- COMPLIANT, NON_COMPLIANT, REVIEW_REQUIRED
    risk_level VARCHAR(20) DEFAULT 'LOW' -- LOW, MEDIUM, HIGH, CRITICAL
);

-- System configuration table
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    is_encrypted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Batch processing jobs table
CREATE TABLE IF NOT EXISTS batch_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_name VARCHAR(255) NOT NULL,
    job_type VARCHAR(50) NOT NULL, -- DOCUMENT_BATCH, COMPLIANCE_REPORT, etc.
    status VARCHAR(50) NOT NULL DEFAULT 'queued', -- queued, running, completed, failed
    parameters JSONB, -- Job-specific parameters
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_log TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255)
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    metric_unit VARCHAR(20), -- seconds, bytes, count, etc.
    measurement_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    context_data JSONB, -- Additional context like document_size, batch_id, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_document_processing_status ON document_processing(status);
CREATE INDEX IF NOT EXISTS idx_document_processing_created_at ON document_processing(created_at);
CREATE INDEX IF NOT EXISTS idx_document_processing_file_hash ON document_processing(file_hash);

CREATE INDEX IF NOT EXISTS idx_phi_detection_doc_id ON phi_detection_results(document_processing_id);
CREATE INDEX IF NOT EXISTS idx_phi_detection_entity_type ON phi_detection_results(entity_type);

CREATE INDEX IF NOT EXISTS idx_compliance_events_timestamp ON audit.compliance_events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_compliance_events_type ON audit.compliance_events(event_type);
CREATE INDEX IF NOT EXISTS idx_compliance_events_doc_id ON audit.compliance_events(document_processing_id);

CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_type ON batch_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_created_at ON batch_jobs(created_at);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(measurement_timestamp);

-- Triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_document_processing_updated_at 
    BEFORE UPDATE ON document_processing 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_config_updated_at 
    BEFORE UPDATE ON system_config 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Data retention function for compliance (7 years for HIPAA)
CREATE OR REPLACE FUNCTION cleanup_old_records()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    retention_date TIMESTAMP WITH TIME ZONE;
BEGIN
    -- Calculate retention date (7 years ago)
    retention_date := NOW() - INTERVAL '7 years';
    
    -- Clean up old document processing records
    WITH deleted_docs AS (
        DELETE FROM document_processing 
        WHERE created_at < retention_date 
        RETURNING id
    )
    SELECT count(*) INTO deleted_count FROM deleted_docs;
    
    -- Clean up old compliance events (keep longer for audit purposes - 10 years)
    DELETE FROM audit.compliance_events 
    WHERE event_timestamp < (NOW() - INTERVAL '10 years');
    
    -- Clean up old performance metrics (keep 2 years)
    DELETE FROM performance_metrics 
    WHERE measurement_timestamp < (NOW() - INTERVAL '2 years');
    
    -- Log cleanup activity
    INSERT INTO audit.compliance_events (
        event_type, 
        event_details, 
        compliance_status
    ) VALUES (
        'DATA_RETENTION_CLEANUP',
        jsonb_build_object(
            'deleted_records', deleted_count,
            'retention_date', retention_date,
            'cleanup_timestamp', NOW()
        ),
        'COMPLIANT'
    );
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Insert default system configuration
INSERT INTO system_config (config_key, config_value, description) VALUES
    ('data_retention_years', '7', 'HIPAA-required data retention period in years'),
    ('phi_detection_threshold', '0.95', 'Minimum confidence score for PHI detection'),
    ('compliance_check_interval', '3600', 'Compliance check interval in seconds'),
    ('max_batch_size', '1000', 'Maximum number of documents per batch job'),
    ('encryption_key_rotation_days', '90', 'Encryption key rotation interval in days')
ON CONFLICT (config_key) DO NOTHING;

-- Create a view for compliance reporting
CREATE OR REPLACE VIEW compliance_summary AS
SELECT 
    DATE_TRUNC('day', dp.created_at) as processing_date,
    COUNT(*) as total_documents,
    COUNT(CASE WHEN dp.status = 'completed' THEN 1 END) as completed_documents,
    COUNT(CASE WHEN dp.status = 'failed' THEN 1 END) as failed_documents,
    AVG(dp.compliance_score) as avg_compliance_score,
    SUM(dp.phi_entities_detected) as total_phi_detected,
    SUM(dp.redaction_count) as total_redactions,
    AVG(EXTRACT(EPOCH FROM (dp.processing_completed_at - dp.processing_started_at))) as avg_processing_time_seconds
FROM document_processing dp
WHERE dp.created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', dp.created_at)
ORDER BY processing_date DESC;

-- Grant permissions on all created objects
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO hipaa;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA audit TO hipaa;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO hipaa;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO hipaa;
GRANT SELECT ON compliance_summary TO hipaa;

-- Log successful initialization
INSERT INTO audit.compliance_events (
    event_type, 
    event_details, 
    user_id,
    compliance_status
) VALUES (
    'DATABASE_INITIALIZATION',
    jsonb_build_object(
        'database_name', current_database(),
        'initialization_time', NOW(),
        'version', '1.0.0'
    ),
    'system',
    'COMPLIANT'
);