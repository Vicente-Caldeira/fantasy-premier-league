terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.6"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.12"
    }
  }

  # On a brand-new project, comment this out for the first apply (the bucket does not
  # exist yet), then run `terraform init -migrate-state`.
  backend "gcs" {
    bucket = "fplvicentecaldeira-tfstate"
    prefix = "bootstrap"
  }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}
