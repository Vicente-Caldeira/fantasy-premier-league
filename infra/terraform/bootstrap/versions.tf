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
  }

  # Bootstrap starts with local state. After the first apply, uncomment and run
  # `terraform init -migrate-state` to move state into the bucket it created.
  # backend "gcs" {
  #   bucket = "<project_id>-tfstate"
  #   prefix = "bootstrap"
  # }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true
}
