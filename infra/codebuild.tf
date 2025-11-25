# Terraform configuration for CodeBuild project
# Usage: terraform init && terraform apply

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-south-1"
}

variable "region" {
  default = "ap-south-1"
}

# ECR Repository
resource "aws_ecr_repository" "tts_service" {
  name                 = "tts-service"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  lifecycle_policy {
    policy = jsonencode({
      rules = [{
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "any"
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      }]
    })
  }
}

# IAM Role for CodeBuild
resource "aws_iam_role" "codebuild_role" {
  name = "tts-service-codebuild-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "codebuild.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

# IAM Policy for CodeBuild (ECR access)
resource "aws_iam_role_policy" "codebuild_policy" {
  role = aws_iam_role.codebuild_role.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = aws_ecr_repository.tts_service.arn
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:ap-south-1:*:log-group:/aws/codebuild/tts-service-build"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "arn:aws:s3:::codebuild-*/*"
      }
    ]
  })
}

# CodeBuild Project
resource "aws_codebuild_project" "tts_service" {
  name          = "tts-service-build"
  description   = "Build TTS service Docker image"
  build_timeout = 60 # minutes
  service_role  = aws_iam_role.codebuild_role.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type                = "BUILD_GENERAL1_LARGE" # 8 vCPU, 15GB RAM
    image                       = "aws/codebuild/standard:7.0"
    type                        = "LINUX_CONTAINER"
    image_pull_credentials_type = "CODEBUILD"
    privileged_mode            = true # Required for Docker-in-Docker

    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = data.aws_caller_identity.current.account_id
    }

    environment_variable {
      name  = "AWS_DEFAULT_REGION"
      value = "ap-south-1"
    }

    environment_variable {
      name  = "IMAGE_REPO_NAME"
      value = aws_ecr_repository.tts_service.name
    }

    environment_variable {
      name  = "IMAGE_TAG"
      value = "latest"
    }
  }

  source {
    type            = "GITHUB"
    location        = "https://github.com/YOUR_ORG/YOUR_REPO.git" # Update this
    git_clone_depth = 1

    git_submodules_config {
      fetch_submodules = false
    }
  }

  source_version = "main" # or your default branch

  logs_config {
    cloudwatch_logs {
      group_name  = "/aws/codebuild/tts-service-build"
      stream_name = ""
    }
  }

  # Cache Docker layers - use S3 for persistent cache across instances
  cache {
    type     = "S3"
    location = "codebuild-cache-${data.aws_caller_identity.current.account_id}-${var.region}"
    modes    = ["LOCAL_DOCKER_LAYER_CACHE", "LOCAL_SOURCE_CACHE"]
  }
}

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# Outputs
output "ecr_repository_url" {
  value = aws_ecr_repository.tts_service.repository_url
}

output "codebuild_project_name" {
  value = aws_codebuild_project.tts_service.name
}

output "build_command" {
  value = "aws codebuild start-build --project-name ${aws_codebuild_project.tts_service.name} --region ap-south-1"
}

