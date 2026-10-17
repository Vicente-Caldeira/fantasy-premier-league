"""Unlink the project's billing account when a budget alert reports overspend.

A message carrying the attribute `dry_run=true` runs a self-test instead: it checks the
function holds the permission it needs to unlink billing, without unlinking anything.
"""

import base64
import json
import logging
import os
from typing import Any

import functions_framework
import google.auth
from cloudevents.http import CloudEvent
from google.auth.transport.requests import AuthorizedSession
from google.cloud import billing_v1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("billing-kill-switch")

UNLINK_PERMISSION = "resourcemanager.projects.deleteBillingAssignment"


def parse_alert(event: CloudEvent) -> tuple[dict[str, Any], dict[str, str]]:
    message = event.data["message"]
    payload = json.loads(base64.b64decode(message["data"]))
    return payload, message.get("attributes") or {}


def is_over_budget(alert: dict[str, Any]) -> bool:
    return float(alert["costAmount"]) > float(alert["budgetAmount"])


def can_unlink_billing(project_id: str) -> bool:
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    response = AuthorizedSession(credentials).post(
        f"https://cloudresourcemanager.googleapis.com/v3/projects/{project_id}:testIamPermissions",
        json={"permissions": [UNLINK_PERMISSION]},
        timeout=30,
    )
    response.raise_for_status()
    return UNLINK_PERMISSION in response.json().get("permissions", [])


def disable_billing(project_id: str) -> None:
    project_name = f"projects/{project_id}"
    client = billing_v1.CloudBillingClient()
    try:
        if not client.get_project_billing_info(name=project_name).billing_enabled:
            logger.info("Billing already disabled for %s", project_name)
            return
    except Exception:
        # Reading state is best-effort; never let it stop the unlink.
        logger.exception("Could not read billing state; unlinking anyway")

    client.update_project_billing_info(
        name=project_name,
        project_billing_info=billing_v1.ProjectBillingInfo(billing_account_name=""),
    )
    logger.critical("BILLING DISABLED for %s", project_name)


@functions_framework.cloud_event
def handle_budget_alert(event: CloudEvent) -> None:
    project_id = os.environ["GCP_PROJECT_ID"]
    try:
        alert, attributes = parse_alert(event)
        over_budget = is_over_budget(alert)
    except (KeyError, TypeError, ValueError):
        # Malformed messages are dropped, not retried: retrying cannot fix them.
        logger.exception("Ignoring malformed budget message")
        return

    if attributes.get("dry_run") == "true":
        ok = can_unlink_billing(project_id)
        logger.log(
            logging.INFO if ok else logging.CRITICAL,
            "SELF-TEST %s: over_budget=%s can_unlink_billing=%s",
            "PASSED" if ok else "FAILED",
            over_budget,
            ok,
        )
        return

    logger.info("Spend %s vs budget %s", alert["costAmount"], alert["budgetAmount"])
    if over_budget:
        # Exceptions propagate so Pub/Sub retries until billing is actually disabled.
        disable_billing(project_id)
