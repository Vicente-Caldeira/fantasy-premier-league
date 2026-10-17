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

variable "budget_amount_usd" {
  description = "Monthly spend at which billing is automatically disabled."
  type        = number
  default     = 1
}
