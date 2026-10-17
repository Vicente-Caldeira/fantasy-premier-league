# Budget -> Pub/Sub -> Cloud Run function that unlinks the billing account once actual
# spend exceeds the budget. Budget notifications lag real spend by hours, so this is a
# backstop on top of staying within Always Free limits, not a hard real-time cap.

resource "google_pubsub_topic" "budget" {
  name       = "billing-budget-alerts"
  depends_on = [time_sleep.api_propagation]
}

resource "google_billing_budget" "monthly" {
  billing_account = var.billing_account_id
  display_name    = "${var.project_id} monthly kill-switch"

  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
    # Subtract only Always Free usage. Trial/promotional credits are NOT subtracted, so
    # spend they would silently absorb still trips the kill-switch.
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
    for_each = [0.5, 0.9, 1.0]
    content {
      threshold_percent = threshold_rules.value
    }
  }

  all_updates_rule {
    pubsub_topic                   = google_pubsub_topic.budget.id
    schema_version                 = "1.0"
    disable_default_iam_recipients = false
  }
}

resource "google_service_account" "kill_switch" {
  account_id   = "billing-kill-switch"
  display_name = "Disables billing when the budget is exceeded"
  depends_on   = [time_sleep.api_propagation]
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

# Function images live here rather than in the auto-created gcf-artifacts repo, so a
# cleanup policy keeps storage far below the 0.5 GB Artifact Registry free tier.
resource "google_artifact_registry_repository" "functions" {
  repository_id          = "functions"
  location               = var.region
  format                 = "DOCKER"
  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-latest"
    action = "KEEP"
    most_recent_versions {
      keep_count = 1
    }
  }

  cleanup_policies {
    id     = "delete-rest"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "3600s"
    }
  }

  depends_on = [time_sleep.api_propagation]
}

# Dedicated identities instead of the default compute service account, whose broad
# Editor grant may not exist (or may be revoked) on newer projects.
resource "google_service_account" "kill_switch_builder" {
  account_id   = "billing-kill-switch-builder"
  display_name = "Builds the billing kill-switch function"
  depends_on   = [time_sleep.api_propagation]
}

resource "google_project_iam_member" "kill_switch_builder" {
  for_each = toset([
    "roles/artifactregistry.writer",
    "roles/logging.logWriter",
    "roles/storage.objectViewer",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.kill_switch_builder.email}"
}

resource "google_service_account" "kill_switch_trigger" {
  account_id   = "billing-kill-switch-trigger"
  display_name = "Delivers budget alerts to the billing kill-switch"
  depends_on   = [time_sleep.api_propagation]
}

resource "google_project_iam_member" "kill_switch_trigger" {
  for_each = toset(["roles/run.invoker", "roles/eventarc.eventReceiver"])
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.kill_switch_trigger.email}"
}

data "archive_file" "kill_switch" {
  type        = "zip"
  source_dir  = "${path.module}/functions/billing_kill_switch"
  output_path = "${path.module}/.build/billing_kill_switch.zip"
  excludes    = ["test_main.py", "__pycache__"]
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
    runtime           = "python312"
    entry_point       = "handle_budget_alert"
    service_account   = google_service_account.kill_switch_builder.id
    docker_repository = google_artifact_registry_repository.functions.id
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
    timeout_seconds       = 60
    service_account_email = google_service_account.kill_switch.email
    environment_variables = {
      GCP_PROJECT_ID = var.project_id
    }
  }

  event_trigger {
    trigger_region        = var.region
    event_type            = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic          = google_pubsub_topic.budget.id
    retry_policy          = "RETRY_POLICY_RETRY"
    service_account_email = google_service_account.kill_switch_trigger.email
  }

  depends_on = [
    time_sleep.api_propagation,
    google_project_iam_member.kill_switch_builder,
    google_project_iam_member.kill_switch_trigger,
  ]
}
