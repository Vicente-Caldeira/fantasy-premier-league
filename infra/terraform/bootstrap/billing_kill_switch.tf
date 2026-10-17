# Budget -> Pub/Sub -> Cloud Run function that unlinks the billing account once actual
# spend exceeds the budget. Budget notifications lag real spend by hours, so this is a
# backstop on top of staying within Always Free limits, not a hard real-time cap.

resource "google_pubsub_topic" "budget" {
  name       = "billing-budget-alerts"
  depends_on = [google_project_service.this]
}

resource "google_billing_budget" "monthly" {
  billing_account = var.billing_account_id
  display_name    = "${var.project_id} monthly kill-switch"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_amount_usd)
    }
  }

  dynamic "threshold_rules" {
    for_each = [0.5, 0.9, 1.0]
    content {
      threshold_percent = threshold_rules.value
    }
  }

  all_updates_rule {
    pubsub_topic   = google_pubsub_topic.budget.id
    schema_version = "1.0"
  }
}

resource "google_service_account" "kill_switch" {
  account_id   = "billing-kill-switch"
  display_name = "Disables billing when the budget is exceeded"
}

resource "google_project_iam_member" "kill_switch_billing" {
  project = var.project_id
  role    = "roles/billing.projectManager"
  member  = "serviceAccount:${google_service_account.kill_switch.email}"
}

resource "google_storage_bucket" "functions_source" {
  name                        = "${var.project_id}-functions-source"
  location                    = upper(var.region)
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
}

data "archive_file" "kill_switch" {
  type        = "zip"
  source_dir  = "${path.module}/functions/billing_kill_switch"
  output_path = "${path.module}/.build/billing_kill_switch.zip"
}

resource "google_storage_bucket_object" "kill_switch" {
  name   = "billing_kill_switch-${data.archive_file.kill_switch.output_md5}.zip"
  bucket = google_storage_bucket.functions_source.name
  source = data.archive_file.kill_switch.output_path
}

resource "google_cloudfunctions2_function" "kill_switch" {
  name     = "billing-kill-switch"
  location = var.region

  build_config {
    runtime     = "python312"
    entry_point = "handle_budget_alert"
    source {
      storage_source {
        bucket = google_storage_bucket.functions_source.name
        object = google_storage_bucket_object.kill_switch.name
      }
    }
  }

  service_config {
    max_instance_count    = 1
    available_memory      = "256M"
    service_account_email = google_service_account.kill_switch.email
    environment_variables = {
      GCP_PROJECT_ID = var.project_id
    }
  }

  event_trigger {
    trigger_region = var.region
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.budget.id
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  depends_on = [google_project_service.this]
}
