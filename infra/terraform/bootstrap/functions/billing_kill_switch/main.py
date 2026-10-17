"""Unlink the project's billing account when a budget alert reports overspend."""

import base64
import json
import logging
import os

import functions_framework
from cloudevents.http import CloudEvent
from google.cloud import billing_v1

logger = logging.getLogger(__name__)


def is_over_budget(alert: dict[str, float]) -> bool:
    return alert["costAmount"] > alert["budgetAmount"]


@functions_framework.cloud_event
def handle_budget_alert(event: CloudEvent) -> None:
    alert = json.loads(base64.b64decode(event.data["message"]["data"]))
    if not is_over_budget(alert):
        logger.info("Spend %s within budget %s", alert["costAmount"], alert["budgetAmount"])
        return

    project_name = f"projects/{os.environ['GCP_PROJECT_ID']}"
    client = billing_v1.CloudBillingClient()
    if not client.get_project_billing_info(name=project_name).billing_enabled:
        logger.info("Billing already disabled for %s", project_name)
        return

    client.update_project_billing_info(
        name=project_name,
        project_billing_info=billing_v1.ProjectBillingInfo(billing_account_name=""),
    )
    logger.warning("Disabled billing for %s: spend %s", project_name, alert["costAmount"])
