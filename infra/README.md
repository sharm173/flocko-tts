# Infrastructure Files

Infrastructure-as-code and CI/CD configuration files.

## Files

- **`codebuild.tf`** - Terraform configuration for CodeBuild project

**Note**: `buildspec.yml` is kept in the root directory as CodeBuild expects it there by default.

### Terraform

```bash
cd infra
terraform init
terraform plan
terraform apply
```

