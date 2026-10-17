"""Create, update or delete a Cloud Billing spend cap budget.

The Terraform google provider has no spend-cap support yet, so Terraform calls this
via `terraform_data`. Uses the Budget API v1 `spendCap` field and only the standard
library plus the gcloud CLI for an access token.

    python3 spend_cap.py apply '<json spec>'
    python3 spend_cap.py delete '<json spec>'

Spec keys: billing_account, display_name, project_number, service_id, amount,
currency, quota_project.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any

Spec = dict[str, Any]
Json = dict[str, Any]

API = "https://billingbudgets.googleapis.com/v1"


def call(method: str, url: str, spec: Spec, body: Json | None = None) -> Json:
    token = subprocess.run(
        ["gcloud", "auth", "print-access-token"],  # noqa: S607 - gcloud from PATH by design
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    request = urllib.request.Request(  # noqa: S310 - fixed https API
        url,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-goog-user-project": spec["quota_project"],
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https API
            raw = response.read()
    except urllib.error.HTTPError as err:
        sys.exit(f"{method} {url} failed: {err.code} {err.read().decode()}")
    return json.loads(raw) if raw else {}


def find(spec: Spec) -> Json | None:
    url = f"{API}/billingAccounts/{spec['billing_account']}/budgets?pageSize=100"
    for budget in call("GET", url, spec).get("budgets", []):
        if budget.get("displayName") == spec["display_name"]:
            return budget
    return None


def body(spec: Spec) -> Json:
    return {
        "displayName": spec["display_name"],
        "budgetFilter": {
            "projects": [f"projects/{spec['project_number']}"],
            "services": [f"services/{spec['service_id']}"],
            "creditTypesTreatment": "EXCLUDE_ALL_CREDITS",
            "calendarPeriod": "MONTH",
        },
        "amount": {
            "specifiedAmount": {"currencyCode": spec["currency"], "units": str(spec["amount"])}
        },
        "thresholdRules": [
            {"thresholdPercent": 0.5},
            {"thresholdPercent": 0.8},
            {"thresholdPercent": 1.0},
        ],
        "notificationsRule": {"enableProjectLevelRecipients": True},
        "spendCap": {"inputState": "CONFIGURED"},
    }


def apply(spec: Spec) -> None:
    existing = find(spec)
    if existing is None:
        url = f"{API}/billingAccounts/{spec['billing_account']}/budgets"
        created = call("POST", url, spec, body(spec))
        print(f"created {created['name']}: {created.get('spendCap')}")
        return
    if existing.get("spendCap", {}).get("outputState") == "ENFORCED":
        sys.exit(f"{existing['name']} is ENFORCED; lift it in the console before changing it")
    url = f"{API}/{existing['name']}?updateMask=amount,budgetFilter,thresholdRules"
    updated = call("PATCH", url, spec, body(spec))
    print(f"updated {updated['name']}: {updated.get('spendCap')}")


def delete(spec: Spec) -> None:
    existing = find(spec)
    if existing is not None:
        call("DELETE", f"{API}/{existing['name']}", spec)
        print(f"deleted {existing['name']}")


if __name__ == "__main__":
    action, spec = sys.argv[1], json.loads(sys.argv[2])
    {"apply": apply, "delete": delete}[action](spec)
