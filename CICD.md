# CI/CD Pipeline & Deployment Automation

## 🚀 Complete CI/CD Implementation

### GitHub Actions Workflow

```yaml
# .github/workflows/ai-agent-pipeline.yml
name: AI Agent System CI/CD Pipeline

on:
  push:
    branches: [main, develop, release/*]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Nightly builds

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  PYTHON_VERSION: '3.11'

jobs:
  # ============================================================================
  # Code Quality & Security Checks
  # ============================================================================
  
  quality-checks:
    name: Code Quality & Security
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0  # Full history for better analysis
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cache/pip
          ~/.cache/pre-commit
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    
    - name: Install dependencies
      run: |
        pip install --upgrade pip
        pip install -r requirements/dev.txt
    
    - name: Run pre-commit hooks
      uses: pre-commit/action@v3.0.0
    
    - name: Run Black formatter
      run: black --check .
    
    - name: Run isort
      run: isort --check-only .
    
    - name: Run flake8
      run: flake8 . --config=.flake8
    
    - name: Run mypy type checking
      run: mypy . --config-file=mypy.ini
    
    - name: Security scan with Bandit
      run: bandit -r agents core api -f json -o bandit-report.json
    
    - name: SAST with Semgrep
      uses: returntocorp/semgrep-action@v1
      with:
        config: >-
          p/security-audit
          p/python
          p/owasp-top-ten
    
    - name: Dependency vulnerability scan
      run: |
        pip install safety
        safety check --json > safety-report.json
    
    - name: License compliance check
      run: |
        pip install pip-licenses
        pip-licenses --format=json > licenses.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
          licenses.json

  # ============================================================================
  # Unit & Integration Tests
  # ============================================================================
  
  test-suite:
    name: Test Suite - ${{ matrix.test-type }}
    runs-on: ${{ matrix.os }}
    needs: quality-checks
    
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        test-type: [unit, integration, contract]
        python-version: ['3.10', '3.11', '3.12']
        exclude:
          - os: windows-latest
            test-type: integration  # Skip integration tests on Windows
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install system dependencies
      if: runner.os == 'Linux'
      run: |
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr poppler-utils
    
    - name: Install Python dependencies
      run: |
        pip install --upgrade pip
        pip install -r requirements/test.txt
    
    - name: Run ${{ matrix.test-type }} tests
      env:
        DATABASE_URL: postgresql://postgres:test@localhost:5432/test
        REDIS_URL: redis://localhost:6379
      run: |
        pytest tests/${{ matrix.test-type }} \
          --cov=agents --cov=core --cov=api \
          --cov-report=xml \
          --cov-report=html \
          --junitxml=test-results-${{ matrix.test-type }}.xml \
          -v
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.os }}-${{ matrix.python-version }}-${{ matrix.test-type }}
        path: |
          test-results-*.xml
          htmlcov/
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: ${{ matrix.test-type }}
        name: ${{ matrix.os }}-${{ matrix.python-version }}-${{ matrix.test-type }}

  # ============================================================================
  # Performance & Load Tests
  # ============================================================================
  
  performance-tests:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: test-suite
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements/test.txt
        pip install locust
    
    - name: Start test environment
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30  # Wait for services
    
    - name: Run performance tests
      run: |
        pytest tests/performance -v --benchmark-only
    
    - name: Run load tests with Locust
      run: |
        locust -f tests/load/locustfile.py \
          --headless \
          --users 100 \
          --spawn-rate 10 \
          --run-time 5m \
          --html performance-report.html
    
    - name: Analyze performance results
      run: |
        python scripts/analyze_performance.py \
          --baseline performance-baseline.json \
          --current performance-report.json \
          --threshold 10  # Allow 10% regression
    
    - name: Upload performance report
      uses: actions/upload-artifact@v3
      with:
        name: performance-report
        path: performance-report.html

  # ============================================================================
  # Build & Package
  # ============================================================================
  
  build-images:
    name: Build Docker Images
    runs-on: ubuntu-latest
    needs: test-suite
    
    permissions:
      contents: read
      packages: write
    
    outputs:
      version: ${{ steps.meta.outputs.version }}
      tags: ${{ steps.meta.outputs.tags }}
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        platforms: linux/amd64,linux/arm64
        build-args: |
          BUILD_DATE=${{ github.event.head_commit.timestamp }}
          VCS_REF=${{ github.sha }}
          VERSION=${{ steps.meta.outputs.version }}

  # ============================================================================
  # Security Scanning
  # ============================================================================
  
  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: build-images
    
    steps:
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ needs.build-images.outputs.version }}
        format: 'sarif'
        output: 'trivy-results.sarif'
        severity: 'CRITICAL,HIGH'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Run Snyk container scan
      uses: snyk/actions/docker@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ needs.build-images.outputs.version }}
        args: --severity-threshold=high

  # ============================================================================
  # Chaos Engineering Tests
  # ============================================================================
  
  chaos-tests:
    name: Chaos Engineering Tests
    runs-on: ubuntu-latest
    needs: build-images
    if: github.event_name == 'schedule' || contains(github.event.head_commit.message, '[chaos]')
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up test cluster
      uses: helm/kind-action@v1.5.0
      with:
        cluster_name: chaos-test
        config: tests/chaos/kind-config.yaml
    
    - name: Install Chaos Mesh
      run: |
        helm repo add chaos-mesh https://charts.chaos-mesh.org
        helm install chaos-mesh chaos-mesh/chaos-mesh \
          --namespace=chaos-testing \
          --create-namespace \
          --set chaosDaemon.runtime=containerd \
          --set chaosDaemon.socketPath=/run/containerd/containerd.sock
    
    - name: Deploy AI Agent System
      run: |
        kubectl create namespace ai-agents
        helm install ai-agents ./helm/ai-agent-system \
          --namespace ai-agents \
          --set image.tag=${{ needs.build-images.outputs.version }}
    
    - name: Wait for deployment
      run: |
        kubectl wait --for=condition=ready pod -l app=ai-agent -n ai-agents --timeout=300s
    
    - name: Run chaos experiments
      run: |
        kubectl apply -f tests/chaos/experiments/
        sleep 300  # Run experiments for 5 minutes
    
    - name: Validate system resilience
      run: |
        python tests/chaos/validate_resilience.py \
          --namespace ai-agents \
          --expected-availability 99.0

  # ============================================================================
  # Deploy to Staging
  # ============================================================================
  
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [build-images, security-scan]
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --name ai-agents-staging --region us-east-1
    
    - name: Deploy with Helm
      run: |
        helm upgrade --install ai-agents ./helm/ai-agent-system \
          --namespace ai-agents-staging \
          --create-namespace \
          --set image.tag=${{ needs.build-images.outputs.version }} \
          --set environment=staging \
          --values helm/values.staging.yaml \
          --wait \
          --timeout 10m
    
    - name: Run smoke tests
      run: |
        kubectl run smoke-test \
          --image=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ needs.build-images.outputs.version }} \
          --rm -it --restart=Never \
          --command -- python -m tests.smoke.run_smoke_tests
    
    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        text: 'Staging deployment completed for version ${{ needs.build-images.outputs.version }}'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}

  # ============================================================================
  # Deploy to Production
  # ============================================================================
  
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build-images, security-scan, deploy-staging]
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure kubectl for production
      run: |
        # Configure for your production cluster
        echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Blue-Green Deployment
      run: |
        # Deploy to green environment
        helm upgrade --install ai-agents-green ./helm/ai-agent-system \
          --namespace ai-agents-prod \
          --set image.tag=${{ needs.build-images.outputs.version }} \
          --set environment=production \
          --set deployment.strategy=blue-green \
          --set deployment.slot=green \
          --values helm/values.production.yaml \
          --wait
        
        # Run production tests
        kubectl run prod-test \
          --image=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ needs.build-images.outputs.version }} \
          --rm -it --restart=Never \
          --command -- python -m tests.production.validate_deployment
        
        # Switch traffic to green
        kubectl patch service ai-agents-prod \
          -n ai-agents-prod \
          -p '{"spec":{"selector":{"slot":"green"}}}'
        
        # Wait and monitor
        sleep 300  # 5 minutes
        
        # If successful, remove blue deployment
        helm delete ai-agents-blue -n ai-agents-prod || true
    
    - name: Create release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: v${{ needs.build-images.outputs.version }}
        release_name: Release ${{ needs.build-images.outputs.version }}
        body: |
          AI Agent System Release ${{ needs.build-images.outputs.version }}
          
          ## Changes
          ${{ github.event.head_commit.message }}
          
          ## Docker Image
          `${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ needs.build-images.outputs.version }}`
        draft: false
        prerelease: false
```

### GitLab CI/CD Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - quality
  - test
  - build
  - security
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: ""
  REGISTRY: registry.gitlab.com
  IMAGE_NAME: ${CI_PROJECT_PATH}

# ============================================================================
# Templates
# ============================================================================

.python_template:
  image: python:3.11
  before_script:
    - pip install --upgrade pip
    - pip install -r requirements/dev.txt
  cache:
    paths:
      - .cache/pip
      - venv/

.docker_template:
  image: docker:24-dind
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY

# ============================================================================
# Quality Stage
# ============================================================================

code_quality:
  extends: .python_template
  stage: quality
  script:
    - black --check .
    - isort --check-only .
    - flake8 .
    - mypy .
    - bandit -r agents core api
  artifacts:
    reports:
      codequality: code-quality-report.json

sonarqube:
  stage: quality
  image: sonarsource/sonar-scanner-cli:latest
  script:
    - sonar-scanner
      -Dsonar.projectKey=ai-agent-system
      -Dsonar.sources=.
      -Dsonar.host.url=$SONAR_HOST_URL
      -Dsonar.login=$SONAR_TOKEN

# ============================================================================
# Test Stage
# ============================================================================

unit_tests:
  extends: .python_template
  stage: test
  services:
    - postgres:15
    - redis:7-alpine
  variables:
    POSTGRES_DB: test
    POSTGRES_PASSWORD: test
    DATABASE_URL: postgresql://postgres:test@postgres:5432/test
    REDIS_URL: redis://redis:6379
  script:
    - pytest tests/unit --cov --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
      junit: test-results.xml

integration_tests:
  extends: .python_template
  stage: test
  services:
    - postgres:15
    - redis:7-alpine
    - docker:24-dind
  script:
    - pytest tests/integration -v
  artifacts:
    reports:
      junit: integration-test-results.xml

performance_tests:
  extends: .python_template
  stage: test
  script:
    - pytest tests/performance --benchmark-only --benchmark-json=benchmark.json
    - python scripts/check_performance_regression.py benchmark.json
  artifacts:
    paths:
      - benchmark.json
  only:
    - merge_requests
    - main

# ============================================================================
# Build Stage
# ============================================================================

build_images:
  extends: .docker_template
  stage: build
  script:
    - docker build -t $REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA .
    - docker tag $REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA $REGISTRY/$IMAGE_NAME:$CI_COMMIT_REF_SLUG
    - docker push $REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA
    - docker push $REGISTRY/$IMAGE_NAME:$CI_COMMIT_REF_SLUG
    
    # Build multi-arch images
    - docker buildx create --use
    - docker buildx build
      --platform linux/amd64,linux/arm64
      --push
      -t $REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA-multiarch .

# ============================================================================
# Security Stage
# ============================================================================

container_scanning:
  stage: security
  image: registry.gitlab.com/gitlab-org/security-products/analyzers/klar:latest
  variables:
    CLAIR_DB_IMAGE: arminc/clair-db:latest
    CLAIR_DB_IMAGE_TAG: latest
    DOCKERFILE_PATH: Dockerfile
  script:
    - /analyzer run
  artifacts:
    reports:
      container_scanning: gl-container-scanning-report.json

dependency_scanning:
  stage: security
  image: registry.gitlab.com/gitlab-org/security-products/analyzers/gemnasium:latest
  script:
    - /analyzer run
  artifacts:
    reports:
      dependency_scanning: gl-dependency-scanning-report.json

# ============================================================================
# Deploy Stage
# ============================================================================

deploy_staging:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: https://staging.ai-agents.example.com
  script:
    - kubectl config use-context $KUBE_CONTEXT_STAGING
    - helm upgrade --install ai-agents ./helm/ai-agent-system
      --namespace ai-agents-staging
      --set image.tag=$CI_COMMIT_SHA
      --values helm/values.staging.yaml
      --wait
  only:
    - develop

deploy_production:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://ai-agents.example.com
  script:
    # Canary deployment
    - kubectl config use-context $KUBE_CONTEXT_PROD
    - helm upgrade --install ai-agents-canary ./helm/ai-agent-system
      --namespace ai-agents-prod
      --set image.tag=$CI_COMMIT_SHA
      --set deployment.strategy=canary
      --set deployment.canary.weight=10
      --values helm/values.production.yaml
    
    # Monitor canary
    - python scripts/monitor_canary.py --duration 600 --error-threshold 1
    
    # Promote to full deployment
    - helm upgrade ai-agents ./helm/ai-agent-system
      --namespace ai-agents-prod
      --set image.tag=$CI_COMMIT_SHA
      --values helm/values.production.yaml
      --wait
  only:
    - main
  when: manual
```

### Terraform Infrastructure as Code

```hcl
# infrastructure/terraform/main.tf

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
  
  backend "s3" {
    bucket = "ai-agents-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
    dynamodb_table = "terraform-state-lock"
  }
}

# EKS Cluster for AI Agents
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"
  
  cluster_name    = "ai-agents-${var.environment}"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Node groups
  eks_managed_node_groups = {
    # CPU nodes for general workloads
    general = {
      desired_size = 3
      min_size     = 3
      max_size     = 10
      
      instance_types = ["m6i.xlarge"]
      
      labels = {
        workload = "general"
      }
      
      taints = []
    }
    
    # GPU nodes for ML workloads
    gpu = {
      desired_size = 2
      min_size     = 1
      max_size     = 5
      
      instance_types = ["g4dn.xlarge"]
      
      labels = {
        workload = "gpu"
        "nvidia.com/gpu" = "true"
      }
      
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
      
      # Install NVIDIA drivers
      user_data = base64encode(templatefile("${path.module}/user_data/gpu_init.sh", {}))
    }
    
    # High memory nodes for caching
    memory = {
      desired_size = 2
      min_size     = 1
      max_size     = 4
      
      instance_types = ["r6i.2xlarge"]
      
      labels = {
        workload = "memory-intensive"
      }
    }
  }
  
  # Enable IRSA for pod IAM roles
  enable_irsa = true
  
  # Add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
}

# RDS for PostgreSQL
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"
  
  identifier = "ai-agents-${var.environment}"
  
  engine               = "postgres"
  engine_version       = "15.4"
  family              = "postgres15"
  major_engine_version = "15"
  instance_class      = "db.r6g.xlarge"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true
  
  db_name  = "aiagents"
  username = "aiagents"
  port     = 5432
  
  multi_az               = true
  vpc_security_group_ids = [module.security_group.security_group_id]
  subnet_ids            = module.vpc.database_subnets
  
  backup_retention_period = 30
  backup_window          = "03:00-06:00"
  maintenance_window     = "Mon:00:00-Mon:03:00"
  
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  
  deletion_protection = true
}

# ElastiCache for Redis
module "redis" {
  source = "terraform-aws-modules/elasticache/aws"
  
  cluster_id           = "ai-agents-${var.environment}"
  engine              = "redis"
  node_type           = "cache.r6g.xlarge"
  num_cache_nodes     = 3
  engine_version      = "7.0"
  port                = 6379
  
  subnet_ids         = module.vpc.private_subnets
  security_group_ids = [module.security_group.security_group_id]
  
  # Enable Redis cluster mode
  parameter_group_family = "redis7"
  
  # Enable automatic failover
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  # Enable backups
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
  
  # Enable encryption
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
}

# S3 buckets for model storage
resource "aws_s3_bucket" "models" {
  bucket = "ai-agents-models-${var.environment}"
  
  tags = {
    Environment = var.environment
    Purpose     = "ml-models"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

# Deploy AI Agent System
resource "helm_release" "ai_agents" {
  name       = "ai-agents"
  namespace  = "ai-agents"
  repository = "https://charts.ai-agents.io"
  chart      = "ai-agent-system"
  version    = var.chart_version
  
  values = [
    templatefile("${path.module}/helm-values/values.yaml", {
      environment     = var.environment
      image_tag      = var.image_tag
      database_url   = module.rds.db_instance_endpoint
      redis_url      = module.redis.configuration_endpoint_address
      model_bucket   = aws_s3_bucket.models.id
    })
  ]
  
  depends_on = [
    module.eks,
    module.rds,
    module.redis
  ]
}

# Monitoring stack
resource "helm_release" "monitoring" {
  name       = "monitoring"
  namespace  = "monitoring"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  
  values = [
    file("${path.module}/monitoring/prometheus-values.yaml")
  ]
  
  depends_on = [module.eks]
}
```

### ArgoCD Application Manifests

```yaml
# argocd/applications/ai-agent-system.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: ai-agent-system
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  
  source:
    repoURL: https://github.com/your-org/ai-agent-system
    targetRevision: HEAD
    path: k8s/overlays/production
    
    # Kustomize configuration
    kustomize:
      images:
        - name: ai-agent-system
          newTag: ${ARGOCD_APP_REVISION}
  
  destination:
    server: https://kubernetes.default.svc
    namespace: ai-agents
  
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    
    syncOptions:
    - CreateNamespace=true
    - PrunePropagationPolicy=foreground
    - PruneLast=true
    
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
  
  # Progressive delivery with Flagger
  progressiveSync:
    enabled: true
    canary:
      steps:
        - setWeight: 10
        - pause: {duration: 10m}
        - setWeight: 30
        - pause: {duration: 10m}
        - setWeight: 50
        - pause: {duration: 10m}
      
      analysis:
        templates:
        - templateName: success-rate
          args:
          - name: service-name
            value: ai-agent-system
        
        - templateName: latency
          args:
          - name: service-name
            value: ai-agent-system
      
      thresholds:
        successRate: 99
        latency: 500
```

### Monitoring & Observability

```yaml
# monitoring/grafana-dashboards/ai-agents-dashboard.json
{
  "dashboard": {
    "title": "AI Agent System - Production Dashboard",
    "panels": [
      {
        "title": "Agent Performance",
        "targets": [
          {
            "expr": "rate(agent_decisions_total[5m])",
            "legendFormat": "{{agent_id}} - Decisions/sec"
          }
        ]
      },
      {
        "title": "Document Processing Rate",
        "targets": [
          {
            "expr": "rate(documents_processed_total[5m])",
            "legendFormat": "Documents/sec"
          }
        ]
      },
      {
        "title": "Error Rate by Agent",
        "targets": [
          {
            "expr": "rate(agent_errors_total[5m]) / rate(agent_decisions_total[5m])",
            "legendFormat": "{{agent_id}} - Error Rate"
          }
        ]
      },
      {
        "title": "Learning Improvements",
        "targets": [
          {
            "expr": "agent_learning_improvement_ratio",
            "legendFormat": "{{agent_id}} - Improvement"
          }
        ]
      }
    ]
  }
}
```

## 🚀 Deployment Strategies

### Blue-Green Deployment
- Zero-downtime deployments
- Instant rollback capability
- Full environment testing before switch

### Canary Deployment
- Gradual rollout to subset of users
- Automated rollback on metric degradation
- A/B testing capabilities

### Feature Flags
```python
# Feature flag configuration
feature_flags = {
    "new_extraction_algorithm": {
        "enabled": True,
        "rollout_percentage": 10,
        "target_agents": ["extractor_001", "extractor_002"]
    },
    "advanced_reasoning": {
        "enabled": False,
        "whitelist_users": ["beta_tester_001"]
    }
}
```

## 📊 Success Metrics

- **Deployment Frequency**: Multiple times per day
- **Lead Time**: < 30 minutes from commit to production
- **MTTR**: < 5 minutes
- **Change Failure Rate**: < 1%
- **Test Coverage**: > 95%
- **Security Scan Pass Rate**: 100%