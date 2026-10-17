# Spend guardrails beyond the kill-switch. GCP has no hard spending cap, so we layer:
#   1. Quotas: enforced in real time by the service itself (strongest).
#   2. Account-wide budget: emails billing admins about spend in ANY project on the account.
#   3. Kill-switch (billing_kill_switch.tf): unlinks billing from this project.

resource "google_cloud_quotas_quota_preference" "bigquery_query_per_day" {
  parent        = "projects/${var.project_id}"
  name          = "bigquery-query-usage-per-day"
  service       = "bigquery.googleapis.com"
  quota_id      = "QueryUsagePerDay"
  justification = "Keep BigQuery on-demand query spend inside the free tier."
  # Decreasing from the 200 TiB default is far more than 10%, which needs this flag.
  ignore_safety_checks = "QUOTA_DECREASE_PERCENTAGE_TOO_HIGH"

  quota_config {
    preferred_value = tostring(var.bigquery_daily_query_limit_gib * 1024) # unit is MiB
  }

  depends_on = [google_project_service.this]
}

resource "google_billing_budget" "account_wide" {
  billing_account = var.billing_account_id
  display_name    = "Billing account: any spend"

  budget_filter {
    credit_types_treatment = "INCLUDE_SPECIFIED_CREDITS"
    credit_types           = ["FREE_TIER"]
  }

  amount {
    specified_amount {
      currency_code = var.budget_currency
      units         = tostring(var.budget_amount)
    }
  }

  dynamic "threshold_rules" {
    for_each = [0.01, 0.5, 1.0]
    content {
      threshold_percent = threshold_rules.value
    }
  }

  # No all_updates_rule: budgets email billing admins at each threshold by default.
}
