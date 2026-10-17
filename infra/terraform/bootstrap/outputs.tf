output "tfstate_bucket" {
  value = google_storage_bucket.tfstate.name
}

output "workload_identity_provider" {
  description = "Set as the GCP_WORKLOAD_IDENTITY_PROVIDER GitHub Actions variable."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deployer_service_account" {
  description = "Set as the GCP_DEPLOYER_SA GitHub Actions variable."
  value       = google_service_account.deployer.email
}
