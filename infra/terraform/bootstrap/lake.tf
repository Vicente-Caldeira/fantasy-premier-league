# Data lake for the daily snapshots. FPL prices, news and upcoming odds exist only as
# current state, so a day not captured is lost for good; the snapshot runs in GitHub
# Actions (public repo: free minutes) and writes here. ~1 MB and ~12 objects a day stay
# far inside the 5 GB / 5,000 writes a month Always Free allowance.

resource "google_storage_bucket" "lake" {
  name                        = "${var.project_id}-lake"
  location                    = upper(var.region)
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # Snapshots can't be re-taken, so keep deleted or overwritten objects for a week.
  soft_delete_policy {
    retention_duration_seconds = 604800
  }

  depends_on = [time_sleep.api_propagation]
}

resource "google_service_account" "lake_snapshotter" {
  account_id   = "lake-snapshotter"
  display_name = "Writes daily snapshots to the data lake"
  depends_on   = [time_sleep.api_propagation]
}

# Objects in this bucket only: no project-level grant.
resource "google_storage_bucket_iam_member" "lake_snapshotter" {
  bucket = google_storage_bucket.lake.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${google_service_account.lake_snapshotter.email}"
}

# Only workflow runs on main may impersonate it (the provider already pins the repo),
# so a pull request can't write to the lake.
resource "google_service_account_iam_member" "lake_snapshotter_wif" {
  service_account_id = google_service_account.lake_snapshotter.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.ref/refs/heads/main"
}
