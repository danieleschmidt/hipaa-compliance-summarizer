"""Worker for notification and alerting jobs."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

from ..monitoring.tracing import trace_operation
from .queue_manager import Job

logger = logging.getLogger(__name__)


class NotificationWorker:
    """Worker for handling notification and alerting jobs."""

    def __init__(self, notification_service=None):
        """Initialize notification worker.
        
        Args:
            notification_service: Notification service instance
        """
        self.notification_service = notification_service
        self.notifications_sent = 0
        self.notification_errors = 0

    @trace_operation("email_notification_worker")
    async def process_email_notification_job(self, job: Job) -> Dict[str, Any]:
        """Process an email notification job.
        
        Args:
            job: Job containing email notification data
            
        Returns:
            Notification result
        """
        logger.info(f"Processing email notification job {job.job_id}")

        try:
            # Extract email data
            email_data = job.data.get("email_data")
            if not email_data:
                raise ValueError("No email data provided")

            recipients = email_data.get("recipients", [])
            subject = email_data.get("subject", "HIPAA Compliance Notification")
            body = email_data.get("body", "")
            template_name = email_data.get("template_name")
            template_vars = email_data.get("template_vars", {})
            attachments = email_data.get("attachments", [])

            if not recipients:
                raise ValueError("No recipients provided for email")

            # Send email
            if self.notification_service and hasattr(self.notification_service, 'send_email'):
                success = self.notification_service.send_email(
                    to_addresses=recipients,
                    subject=subject,
                    body=body,
                    template_name=template_name,
                    template_vars=template_vars,
                    attachments=attachments
                )
            else:
                # Simulate email sending
                logger.info(f"Simulated email to {len(recipients)} recipients: {subject}")
                success = True

            if success:
                self.notifications_sent += 1
            else:
                self.notification_errors += 1
                raise Exception("Email sending failed")

            result = {
                "notification_type": "email",
                "recipients_count": len(recipients),
                "subject": subject,
                "success": success,
                "sent_at": datetime.utcnow().isoformat()
            }

            logger.info(f"Email notification job {job.job_id} completed successfully")
            return result

        except Exception as e:
            self.notification_errors += 1
            logger.error(f"Email notification job {job.job_id} failed: {e}")
            raise

    @trace_operation("slack_notification_worker")
    async def process_slack_notification_job(self, job: Job) -> Dict[str, Any]:
        """Process a Slack notification job.
        
        Args:
            job: Job containing Slack notification data
            
        Returns:
            Notification result
        """
        logger.info(f"Processing Slack notification job {job.job_id}")

        try:
            # Extract Slack data
            slack_data = job.data.get("slack_data")
            if not slack_data:
                raise ValueError("No Slack data provided")

            message = slack_data.get("message", "")
            channel = slack_data.get("channel")
            username = slack_data.get("username", "HIPAA Bot")
            emoji = slack_data.get("emoji", ":shield:")
            attachments = slack_data.get("attachments", [])

            if not message:
                raise ValueError("No message provided for Slack notification")

            # Send Slack message
            if (self.notification_service and
                hasattr(self.notification_service, 'send_message')):
                success = self.notification_service.send_message(
                    message=message,
                    channel=channel,
                    username=username,
                    emoji=emoji,
                    attachments=attachments
                )
            else:
                # Simulate Slack sending
                logger.info(f"Simulated Slack message to {channel or 'default'}: {message[:50]}...")
                success = True

            if success:
                self.notifications_sent += 1
            else:
                self.notification_errors += 1
                raise Exception("Slack message sending failed")

            result = {
                "notification_type": "slack",
                "channel": channel,
                "message_preview": message[:100],
                "success": success,
                "sent_at": datetime.utcnow().isoformat()
            }

            logger.info(f"Slack notification job {job.job_id} completed successfully")
            return result

        except Exception as e:
            self.notification_errors += 1
            logger.error(f"Slack notification job {job.job_id} failed: {e}")
            raise

    @trace_operation("compliance_alert_worker")
    async def process_compliance_alert_job(self, job: Job) -> Dict[str, Any]:
        """Process a compliance alert notification job.
        
        Args:
            job: Job containing compliance alert data
            
        Returns:
            Alert notification result
        """
        logger.info(f"Processing compliance alert job {job.job_id}")

        try:
            # Extract alert data
            alert_data = job.data.get("alert_data")
            notification_config = job.data.get("notification_config", {})

            if not alert_data:
                raise ValueError("No alert data provided")

            # Determine notification channels based on severity
            severity = alert_data.get("severity", "medium")
            channels = self._get_notification_channels(severity, notification_config)

            results = []

            # Send notifications to each channel
            for channel_config in channels:
                channel_type = channel_config["type"]

                try:
                    if channel_type == "email":
                        result = await self._send_compliance_email(alert_data, channel_config)
                    elif channel_type == "slack":
                        result = await self._send_compliance_slack(alert_data, channel_config)
                    elif channel_type == "webhook":
                        result = await self._send_compliance_webhook(alert_data, channel_config)
                    else:
                        logger.warning(f"Unknown notification channel type: {channel_type}")
                        continue

                    results.append(result)

                except Exception as e:
                    logger.error(f"Failed to send {channel_type} notification: {e}")
                    results.append({
                        "channel_type": channel_type,
                        "success": False,
                        "error": str(e)
                    })

            # Calculate overall success
            successful_notifications = sum(1 for r in results if r.get("success", False))
            total_notifications = len(results)

            overall_result = {
                "alert_id": alert_data.get("alert_id", job.job_id),
                "severity": severity,
                "total_notifications": total_notifications,
                "successful_notifications": successful_notifications,
                "notification_results": results,
                "overall_success": successful_notifications > 0,
                "processed_at": datetime.utcnow().isoformat()
            }

            if successful_notifications > 0:
                self.notifications_sent += successful_notifications

            if total_notifications - successful_notifications > 0:
                self.notification_errors += total_notifications - successful_notifications

            logger.info(f"Compliance alert job {job.job_id} completed: {successful_notifications}/{total_notifications} notifications sent")
            return overall_result

        except Exception as e:
            self.notification_errors += 1
            logger.error(f"Compliance alert job {job.job_id} failed: {e}")
            raise

    def _get_notification_channels(self, severity: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get notification channels based on severity."""
        default_channels = {
            "low": [{"type": "email", "template": "low_priority"}],
            "medium": [
                {"type": "email", "template": "medium_priority"},
                {"type": "slack", "channel": "#hipaa-alerts"}
            ],
            "high": [
                {"type": "email", "template": "high_priority", "urgent": True},
                {"type": "slack", "channel": "#hipaa-alerts"},
                {"type": "webhook", "endpoint": "monitoring"}
            ],
            "critical": [
                {"type": "email", "template": "critical_priority", "urgent": True},
                {"type": "slack", "channel": "#hipaa-critical"},
                {"type": "webhook", "endpoint": "security"},
                {"type": "webhook", "endpoint": "compliance"}
            ]
        }

        # Use config override if provided
        if "channels" in config:
            return config["channels"].get(severity, default_channels.get(severity, []))

        return default_channels.get(severity, [])

    async def _send_compliance_email(self, alert_data: Dict[str, Any],
                                   channel_config: Dict[str, Any]) -> Dict[str, Any]:
        """Send compliance alert via email."""
        if not self.notification_service or not hasattr(self.notification_service, 'send_compliance_violation_alert'):
            # Simulate email
            return {
                "channel_type": "email",
                "success": True,
                "simulated": True,
                "message": "Email notification simulated"
            }

        # Get recipient list (would typically come from configuration)
        recipients = channel_config.get("recipients", ["compliance@example.com"])

        success = self.notification_service.send_compliance_violation_alert(
            to_addresses=recipients,
            document_id=alert_data.get("document_id", "unknown"),
            violation_type=alert_data.get("rule_name", "compliance_violation"),
            risk_level=alert_data.get("severity", "medium"),
            violation_details=alert_data.get("message", "Compliance violation detected")
        )

        return {
            "channel_type": "email",
            "success": success,
            "recipients_count": len(recipients)
        }

    async def _send_compliance_slack(self, alert_data: Dict[str, Any],
                                   channel_config: Dict[str, Any]) -> Dict[str, Any]:
        """Send compliance alert via Slack."""
        if not self.notification_service or not hasattr(self.notification_service, 'send_compliance_alert'):
            # Simulate Slack
            return {
                "channel_type": "slack",
                "success": True,
                "simulated": True,
                "message": "Slack notification simulated"
            }

        success = self.notification_service.send_compliance_alert(
            document_id=alert_data.get("document_id", "unknown"),
            violation_type=alert_data.get("rule_name", "compliance_violation"),
            risk_level=alert_data.get("severity", "medium"),
            details=alert_data.get("message", "Compliance violation detected")
        )

        return {
            "channel_type": "slack",
            "success": success,
            "channel": channel_config.get("channel", "#hipaa-alerts")
        }

    async def _send_compliance_webhook(self, alert_data: Dict[str, Any],
                                     channel_config: Dict[str, Any]) -> Dict[str, Any]:
        """Send compliance alert via webhook."""
        if not self.notification_service or not hasattr(self.notification_service, 'send_webhook'):
            # Simulate webhook
            return {
                "channel_type": "webhook",
                "success": True,
                "simulated": True,
                "message": "Webhook notification simulated"
            }

        webhook_payload = {
            "alert_type": "compliance_violation",
            "alert_data": alert_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        endpoint = channel_config.get("endpoint", "monitoring")
        success = self.notification_service.send_webhook(endpoint, webhook_payload)

        return {
            "channel_type": "webhook",
            "success": success,
            "endpoint": endpoint
        }

    @trace_operation("batch_notification_worker")
    async def process_batch_notification_job(self, job: Job) -> Dict[str, Any]:
        """Process a batch notification job.
        
        Args:
            job: Job containing batch notification data
            
        Returns:
            Batch notification result
        """
        logger.info(f"Processing batch notification job {job.job_id}")

        try:
            # Extract batch data
            notifications = job.data.get("notifications", [])
            max_concurrent = job.data.get("max_concurrent", 5)

            if not notifications:
                raise ValueError("No notifications provided in batch")

            # Process notifications concurrently
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_single_notification(notification_data):
                async with semaphore:
                    try:
                        notification_type = notification_data.get("type", "email")

                        if notification_type == "email":
                            # Create temporary job for email
                            temp_job = Job(
                                job_id=f"{job.job_id}_email_{len(notification_data)}",
                                job_type="email_notification",
                                data={"email_data": notification_data}
                            )
                            return await self.process_email_notification_job(temp_job)

                        elif notification_type == "slack":
                            # Create temporary job for Slack
                            temp_job = Job(
                                job_id=f"{job.job_id}_slack_{len(notification_data)}",
                                job_type="slack_notification",
                                data={"slack_data": notification_data}
                            )
                            return await self.process_slack_notification_job(temp_job)

                        else:
                            raise ValueError(f"Unsupported notification type: {notification_type}")

                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "notification_type": notification_data.get("type", "unknown")
                        }

            # Execute all notifications
            results = await asyncio.gather(
                *[process_single_notification(notification) for notification in notifications],
                return_exceptions=True
            )

            # Process results
            successful = 0
            failed = 0

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed += 1
                    results[i] = {"success": False, "error": str(result)}
                elif result.get("success", False):
                    successful += 1
                else:
                    failed += 1

            # Update statistics
            self.notifications_sent += successful
            self.notification_errors += failed

            batch_result = {
                "total_notifications": len(notifications),
                "successful": successful,
                "failed": failed,
                "success_rate": (successful / len(notifications) * 100) if notifications else 0,
                "results": results,
                "processed_at": datetime.utcnow().isoformat()
            }

            logger.info(f"Batch notification job {job.job_id} completed: {successful}/{len(notifications)} successful")
            return batch_result

        except Exception as e:
            self.notification_errors += 1
            logger.error(f"Batch notification job {job.job_id} failed: {e}")
            raise

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        total_notifications = self.notifications_sent + self.notification_errors

        return {
            "notifications_sent": self.notifications_sent,
            "notification_errors": self.notification_errors,
            "success_rate": (
                (self.notifications_sent / total_notifications * 100)
                if total_notifications > 0 else 0
            ),
            "worker_type": "notification"
        }
