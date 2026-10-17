variable "project_id" {
  description = "GCP project that hosts the whole platform."
  type        = string
}

variable "billing_account_id" {
  description = "Billing account linked to the project (format: XXXXXX-XXXXXX-XXXXXX)."
  type        = string
}

variable "region" {
  description = "Always Free tier regions are us-central1, us-east1 and us-west1."
  type        = string
  default     = "us-central1"

  validation {
    condition     = contains(["us-central1", "us-east1", "us-west1"], var.region)
    error_message = "Use an Always Free region so storage stays within the free tier."
  }
}

variable "github_repository" {
  description = "owner/repo allowed to deploy via Workload Identity Federation."
  type        = string
}

variable "budget_amount" {
  description = "Monthly spend, in the billing account's currency, at which billing is disabled."
  type        = number
  default     = 1
}

variable "budget_currency" {
  description = "Must match the billing account's currency."
  type        = string
  default     = "EUR"
}

variable "bigquery_daily_query_limit_gib" {
  description = "Hard per-project cap on BigQuery bytes scanned per day. 30 GiB x 31 days stays under the 1 TiB/month free tier."
  type        = number
  default     = 30
}
