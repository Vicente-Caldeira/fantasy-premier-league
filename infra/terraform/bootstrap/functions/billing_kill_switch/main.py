"""Unlink the project's billing account when a budget alert reports overspend.

A message carrying the attribute `dry_run=true` runs a self-test instead: it checks the
function holds the permission it needs to unlink billing, without unlinking anything.
"""

import base64
import json
import os
from typing import Any

import functions_framework
import google.auth
from cloudevents.http import CloudEvent
from google.auth.transport.requests import AuthorizedSession
from google.cloud import billing_v1


def log(severity: str, message: str, **fields: object) -> None:
    """Structured log line; Cloud Run parses JSON on stdout into severity + fields."""
    print(json.dumps({"severity": severity, "message": message, **fields}), flush=True)


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
            log("INFO", "Billing already disabled", project=project_name)
            return
    except Exception as err:
        # Reading state is best-effort; never let it stop the unlink.
        log("WARNING", "Could not read billing state; unlinking anyway", error=repr(err))

    client.update_project_billing_info(
        name=project_name,
        project_billing_info=billing_v1.ProjectBillingInfo(billing_account_name=""),
    )
    log("CRITICAL", "BILLING DISABLED", project=project_name)


@functions_framework.cloud_event
def handle_budget_alert(event: CloudEvent) -> None:
    project_id = os.environ["GCP_PROJECT_ID"]
    try:
        alert, attributes = parse_alert(event)
        over_budget = is_over_budget(alert)
    except (KeyError, TypeError, ValueError) as err:
        # Malformed messages are dropped, not retried: retrying cannot fix them.
        log("ERROR", "Ignoring malformed budget message", error=repr(err))
        return

    if attributes.get("dry_run") == "true":
        ok = can_unlink_billing(project_id)
        log(
            "NOTICE" if ok else "CRITICAL",
            f"SELF-TEST {'PASSED' if ok else 'FAILED'}",
            over_budget=over_budget,
            can_unlink_billing=ok,
        )
        return

    log(
        "WARNING" if over_budget else "INFO",
        "Budget update",
        cost=alert["costAmount"],
        budget=alert["budgetAmount"],
        over_budget=over_budget,
    )
    if over_budget:
        # Exceptions propagate so Pub/Sub retries until billing is actually disabled.
        disable_billing(project_id)
