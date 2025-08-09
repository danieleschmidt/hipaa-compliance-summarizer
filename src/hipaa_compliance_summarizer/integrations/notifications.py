"""Notification services for alerts and compliance reporting."""

import json
import logging
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email import encoders
from email.mime.base import MimeBase
from email.mime.multipart import MimeMultipart
from email.mime.text import MimeText
from pathlib import Path
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


@dataclass
class NotificationTemplate:
    """Template for notifications."""

    name: str
    subject_template: str
    body_template: str
    notification_type: str  # email, slack, webhook
    priority: str = "normal"  # low, normal, high, critical

    def render_subject(self, **kwargs) -> str:
        """Render subject with variables."""
        return self.subject_template.format(**kwargs)

    def render_body(self, **kwargs) -> str:
        """Render body with variables."""
        return self.body_template.format(**kwargs)


class EmailService:
    """SMTP email service for notifications."""

    def __init__(self):
        """Initialize email service with SMTP configuration."""
        self.smtp_host = os.getenv("SMTP_HOST", "localhost")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
        self.from_email = os.getenv("SMTP_FROM_EMAIL", "noreply@hipaa-summarizer.com")
        self.from_name = os.getenv("SMTP_FROM_NAME", "HIPAA Compliance Summarizer")

        # Load email templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, NotificationTemplate]:
        """Load email templates from configuration."""
        templates = {
            "compliance_violation": NotificationTemplate(
                name="compliance_violation",
                subject_template="[URGENT] HIPAA Compliance Violation Detected - {document_id}",
                body_template="""
HIPAA Compliance Violation Alert

Document ID: {document_id}
Violation Type: {violation_type}
Risk Level: {risk_level}
Detected At: {detected_at}

Details:
{violation_details}

Action Required:
- Review the document immediately
- Investigate the compliance issue
- Document remediation steps
- Report to compliance officer if necessary

This is an automated alert from the HIPAA Compliance Summarizer system.
                """,
                notification_type="email",
                priority="critical"
            ),
            "processing_complete": NotificationTemplate(
                name="processing_complete",
                subject_template="Document Processing Complete - {document_count} documents",
                body_template="""
Document Processing Summary

Batch ID: {batch_id}
Documents Processed: {document_count}
PHI Entities Detected: {phi_count}
Average Compliance Score: {avg_score:.2f}
Processing Time: {processing_time}

High Risk Documents: {high_risk_count}
Failed Documents: {failed_count}

Dashboard: {dashboard_url}

This is an automated notification from the HIPAA Compliance Summarizer system.
                """,
                notification_type="email",
                priority="normal"
            ),
            "system_alert": NotificationTemplate(
                name="system_alert",
                subject_template="[ALERT] HIPAA System Alert - {alert_type}",
                body_template="""
System Alert Notification

Alert Type: {alert_type}
Severity: {severity}
Timestamp: {timestamp}
System Component: {component}

Description:
{description}

Impact:
{impact}

Recommended Actions:
{recommended_actions}

This is an automated alert from the HIPAA Compliance Summarizer system.
                """,
                notification_type="email",
                priority="high"
            ),
            "audit_report": NotificationTemplate(
                name="audit_report",
                subject_template="HIPAA Compliance Audit Report - {period}",
                body_template="""
HIPAA Compliance Audit Report

Reporting Period: {period}
Report Generated: {generated_at}

Summary:
- Total Documents: {total_documents}
- Compliance Score: {compliance_score:.2f}%
- Violations: {violations}
- PHI Entities: {phi_entities}

The detailed audit report is attached to this email.

For questions about this report, please contact the compliance team.
                """,
                notification_type="email",
                priority="normal"
            )
        }

        return templates

    def send_email(self, to_addresses: List[str], subject: str, body: str,
                  attachments: List[str] = None, template_name: str = None,
                  template_vars: Dict[str, Any] = None) -> bool:
        """Send an email notification.
        
        Args:
            to_addresses: List of recipient email addresses
            subject: Email subject
            body: Email body
            attachments: Optional list of file paths to attach
            template_name: Optional template name to use
            template_vars: Variables for template rendering
            
        Returns:
            True if email was sent successfully
        """
        try:
            # Use template if specified
            if template_name and template_name in self.templates:
                template = self.templates[template_name]
                template_vars = template_vars or {}
                subject = template.render_subject(**template_vars)
                body = template.render_body(**template_vars)

            # Create message
            msg = MimeMultipart()
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = ", ".join(to_addresses)
            msg['Subject'] = subject

            # Add body
            msg.attach(MimeText(body, 'plain'))

            # Add attachments
            if attachments:
                for file_path in attachments:
                    self._add_attachment(msg, file_path)

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.smtp_use_tls:
                    server.starttls()

                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)

                server.send_message(msg)

            logger.info(f"Email sent successfully to {len(to_addresses)} recipients")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _add_attachment(self, msg: MimeMultipart, file_path: str):
        """Add file attachment to email message."""
        try:
            with open(file_path, "rb") as attachment:
                part = MimeBase('application', 'octet-stream')
                part.set_payload(attachment.read())

            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {Path(file_path).name}'
            )
            msg.attach(part)

        except Exception as e:
            logger.error(f"Failed to add attachment {file_path}: {e}")

    def send_compliance_violation_alert(self, to_addresses: List[str],
                                       document_id: str, violation_type: str,
                                       risk_level: str, violation_details: str) -> bool:
        """Send compliance violation alert."""
        return self.send_email(
            to_addresses=to_addresses,
            subject="",  # Will be set by template
            body="",     # Will be set by template
            template_name="compliance_violation",
            template_vars={
                "document_id": document_id,
                "violation_type": violation_type,
                "risk_level": risk_level,
                "detected_at": datetime.utcnow().isoformat(),
                "violation_details": violation_details
            }
        )

    def send_processing_summary(self, to_addresses: List[str], batch_id: str,
                               document_count: int, phi_count: int,
                               avg_score: float, processing_time: str,
                               high_risk_count: int, failed_count: int,
                               dashboard_url: str = "") -> bool:
        """Send batch processing summary."""
        return self.send_email(
            to_addresses=to_addresses,
            subject="",  # Will be set by template
            body="",     # Will be set by template
            template_name="processing_complete",
            template_vars={
                "batch_id": batch_id,
                "document_count": document_count,
                "phi_count": phi_count,
                "avg_score": avg_score,
                "processing_time": processing_time,
                "high_risk_count": high_risk_count,
                "failed_count": failed_count,
                "dashboard_url": dashboard_url or "http://localhost:3000"
            }
        )


class SlackNotifier:
    """Slack integration for real-time notifications."""

    def __init__(self):
        """Initialize Slack notifier."""
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.default_channel = os.getenv("SLACK_DEFAULT_CHANNEL", "#hipaa-alerts")

        if not self.webhook_url and not self.bot_token:
            logger.warning("No Slack configuration found - notifications will be disabled")

    def send_message(self, message: str, channel: str = None,
                    username: str = "HIPAA Bot", emoji: str = ":shield:",
                    attachments: List[Dict[str, Any]] = None) -> bool:
        """Send a message to Slack.
        
        Args:
            message: Message text
            channel: Slack channel (defaults to configured channel)
            username: Bot username
            emoji: Bot emoji
            attachments: Optional message attachments
            
        Returns:
            True if message was sent successfully
        """
        if not self.webhook_url:
            logger.warning("No Slack webhook URL configured")
            return False

        try:
            payload = {
                "text": message,
                "channel": channel or self.default_channel,
                "username": username,
                "icon_emoji": emoji
            }

            if attachments:
                payload["attachments"] = attachments

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info("Slack message sent successfully")
                return True
            else:
                logger.error(f"Slack API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False

    def send_compliance_alert(self, document_id: str, violation_type: str,
                             risk_level: str, details: str) -> bool:
        """Send compliance violation alert to Slack."""
        color_map = {
            "critical": "danger",
            "high": "warning",
            "medium": "warning",
            "low": "good"
        }

        attachment = {
            "color": color_map.get(risk_level.lower(), "warning"),
            "title": f"🚨 HIPAA Compliance Violation - {violation_type}",
            "fields": [
                {
                    "title": "Document ID",
                    "value": document_id,
                    "short": True
                },
                {
                    "title": "Risk Level",
                    "value": risk_level.upper(),
                    "short": True
                },
                {
                    "title": "Details",
                    "value": details,
                    "short": False
                },
                {
                    "title": "Time",
                    "value": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "short": True
                }
            ],
            "footer": "HIPAA Compliance Summarizer",
            "ts": int(datetime.utcnow().timestamp())
        }

        return self.send_message(
            message=f"URGENT: HIPAA compliance violation detected in document {document_id}",
            attachments=[attachment]
        )

    def send_processing_update(self, batch_id: str, status: str,
                              document_count: int, progress: float = None) -> bool:
        """Send batch processing update."""
        emoji_map = {
            "started": ":hourglass_flowing_sand:",
            "processing": ":gear:",
            "completed": ":white_check_mark:",
            "failed": ":x:"
        }

        message = f"{emoji_map.get(status, ':information_source:')} Batch {batch_id}: {status.title()}"

        if progress is not None:
            message += f" ({progress:.1f}% complete)"

        if document_count:
            message += f" - {document_count} documents"

        return self.send_message(message)

    def send_system_alert(self, alert_type: str, severity: str,
                         component: str, description: str) -> bool:
        """Send system alert to Slack."""
        severity_colors = {
            "critical": "danger",
            "high": "warning",
            "medium": "warning",
            "low": "good"
        }

        severity_emojis = {
            "critical": ":rotating_light:",
            "high": ":warning:",
            "medium": ":information_source:",
            "low": ":speech_balloon:"
        }

        attachment = {
            "color": severity_colors.get(severity.lower(), "warning"),
            "title": f"{severity_emojis.get(severity.lower(), ':information_source:')} System Alert: {alert_type}",
            "fields": [
                {
                    "title": "Component",
                    "value": component,
                    "short": True
                },
                {
                    "title": "Severity",
                    "value": severity.upper(),
                    "short": True
                },
                {
                    "title": "Description",
                    "value": description,
                    "short": False
                }
            ],
            "footer": "HIPAA System Monitor",
            "ts": int(datetime.utcnow().timestamp())
        }

        return self.send_message(
            message=f"System alert: {alert_type} in {component}",
            attachments=[attachment]
        )


class WebhookNotifier:
    """Generic webhook notifications for external integrations."""

    def __init__(self):
        """Initialize webhook notifier."""
        self.webhook_urls = {
            "compliance": os.getenv("COMPLIANCE_WEBHOOK_URL"),
            "monitoring": os.getenv("MONITORING_WEBHOOK_URL"),
            "security": os.getenv("SECURITY_WEBHOOK_URL")
        }
        self.webhook_secret = os.getenv("WEBHOOK_SECRET")

    def send_webhook(self, webhook_type: str, payload: Dict[str, Any],
                    headers: Dict[str, str] = None) -> bool:
        """Send webhook notification.
        
        Args:
            webhook_type: Type of webhook (compliance, monitoring, security)
            payload: JSON payload to send
            headers: Optional additional headers
            
        Returns:
            True if webhook was sent successfully
        """
        webhook_url = self.webhook_urls.get(webhook_type)

        if not webhook_url:
            logger.warning(f"No webhook URL configured for type: {webhook_type}")
            return False

        try:
            request_headers = {
                "Content-Type": "application/json",
                "User-Agent": "HIPAA-Compliance-Summarizer/1.0"
            }

            if headers:
                request_headers.update(headers)

            # Add signature if secret is configured
            if self.webhook_secret:
                import hashlib
                import hmac

                payload_bytes = json.dumps(payload).encode()
                signature = hmac.new(
                    self.webhook_secret.encode(),
                    payload_bytes,
                    hashlib.sha256
                ).hexdigest()
                request_headers["X-Signature-SHA256"] = f"sha256={signature}"

            response = requests.post(
                webhook_url,
                json=payload,
                headers=request_headers,
                timeout=30
            )

            if response.status_code in [200, 201, 202]:
                logger.info(f"Webhook {webhook_type} sent successfully")
                return True
            else:
                logger.error(f"Webhook error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send webhook {webhook_type}: {e}")
            return False

    def send_compliance_event(self, event_type: str, document_id: str,
                             phi_count: int, compliance_score: float,
                             risk_level: str, details: Dict[str, Any] = None) -> bool:
        """Send compliance event webhook."""
        payload = {
            "event_type": "compliance_event",
            "event_subtype": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "document_id": document_id,
            "phi_count": phi_count,
            "compliance_score": compliance_score,
            "risk_level": risk_level,
            "details": details or {},
            "source": "hipaa-compliance-summarizer"
        }

        return self.send_webhook("compliance", payload)
