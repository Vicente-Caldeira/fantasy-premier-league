import base64
import json
from unittest import mock

import main
import pytest
from cloudevents.http import CloudEvent


def make_event(payload, attributes=None):
    data = base64.b64encode(json.dumps(payload).encode()).decode()
    message = {"data": data, "attributes": attributes or {}}
    return CloudEvent(
        {"type": "google.cloud.pubsub.topic.v1.messagePublished", "source": "test"},
        {"message": message},
    )


@pytest.fixture(autouse=True)
def project_env(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project")


@pytest.fixture
def disable():
    with mock.patch.object(main, "disable_billing") as m:
        yield m


@pytest.mark.parametrize(
    ("cost", "budget", "expected"),
    [(0.0, 1.0, False), (1.0, 1.0, False), (1.01, 1.0, True), ("2.5", "1", True)],
)
def test_is_over_budget(cost, budget, expected):
    assert main.is_over_budget({"costAmount": cost, "budgetAmount": budget}) is expected


def test_within_budget_does_nothing(disable):
    main.handle_budget_alert(make_event({"costAmount": 0.4, "budgetAmount": 1}))
    disable.assert_not_called()


def test_over_budget_disables_billing(disable):
    main.handle_budget_alert(make_event({"costAmount": 1.5, "budgetAmount": 1}))
    disable.assert_called_once_with("test-project")


def test_malformed_message_is_dropped_not_retried(disable):
    main.handle_budget_alert(make_event({"unexpected": True}))
    disable.assert_not_called()


def test_dry_run_never_disables(disable):
    with mock.patch.object(main, "can_unlink_billing", return_value=True) as check:
        main.handle_budget_alert(
            make_event({"costAmount": 99, "budgetAmount": 1}, {"dry_run": "true"})
        )
    check.assert_called_once_with("test-project")
    disable.assert_not_called()


def test_disable_failure_propagates_so_pubsub_retries():
    event = make_event({"costAmount": 5, "budgetAmount": 1})
    with mock.patch.object(main.billing_v1, "CloudBillingClient") as client_cls:
        client = client_cls.return_value
        client.get_project_billing_info.return_value.billing_enabled = True
        client.update_project_billing_info.side_effect = RuntimeError("denied")
        with pytest.raises(RuntimeError):
            main.handle_budget_alert(event)


def test_unreadable_billing_state_still_unlinks():
    with mock.patch.object(main.billing_v1, "CloudBillingClient") as client_cls:
        client = client_cls.return_value
        client.get_project_billing_info.side_effect = RuntimeError("cannot read")
        main.disable_billing("test-project")
    client.update_project_billing_info.assert_called_once()
