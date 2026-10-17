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

  depends_on = [time_sleep.api_propagation]
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

# Spend caps: Google pauses the service itself once GROSS cost (before any free-tier or
# trial credit) reaches the cap. Caps sit well below each service's free-tier value, so
# even with reporting lag the net bill for these services stays at zero.
locals {
  spend_caps = {
    cloud_run = {
      service_id = "152E-C115-5142" # Cloud Run
      amount     = var.cloud_run_spend_cap
    }
    cloud_functions = {
      service_id = "29E7-DA93-CA13" # Cloud Run functions
      amount     = var.cloud_functions_spend_cap
    }
  }
}

resource "terraform_data" "spend_cap" {
  for_each = local.spend_caps

  triggers_replace = {
    billing_account = var.billing_account_id
    display_name    = "spend-cap: ${each.key}"
    project_number  = data.google_project.this.number
    service_id      = each.value.service_id
    amount          = each.value.amount
    currency        = var.budget_currency
    quota_project   = var.project_id
  }

  provisioner "local-exec" {
    command = "python3 ${path.module}/scripts/spend_cap.py apply '${jsonencode(self.triggers_replace)}'"
  }

  provisioner "local-exec" {
    when    = destroy
    command = "python3 ${path.module}/scripts/spend_cap.py delete '${jsonencode(self.triggers_replace)}'"
  }

  depends_on = [time_sleep.api_propagation]
}
