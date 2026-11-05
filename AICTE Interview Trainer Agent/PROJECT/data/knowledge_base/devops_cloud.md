# DevOps & Cloud Engineering Interview Guide

## Core Technical Areas

### Containerization & Orchestration
- Docker: Dockerfile best practices, multi-stage builds, layer caching
- Kubernetes: Pods, Deployments, Services, ConfigMaps, Secrets
- Helm charts for package management
- Container security scanning (Trivy, Aqua Security)
- Service mesh: Istio, Linkerd

### CI/CD Pipelines
- Jenkins: Declarative vs Scripted Pipelines, shared libraries
- GitHub Actions: Workflows, matrix builds, secrets management
- GitLab CI: Stages, environments, artifacts
- ArgoCD, Flux for GitOps
- Blue-Green deployments, Canary releases, Rolling updates

### Infrastructure as Code
- Terraform: Providers, modules, state management, remote backends
- Ansible: Playbooks, roles, inventory management
- Pulumi: Infrastructure using general-purpose programming languages
- AWS CloudFormation, Azure ARM templates, IBM Schematics

### Cloud Platforms
**IBM Cloud (required knowledge for IBM roles)**:
- IBM Kubernetes Service (IKS)
- IBM Cloud Functions (serverless)
- IBM Cloud Object Storage
- IBM Watson AI services integration
- IBM Cloud CLI and API usage
- IBM Schematics (Terraform-based IaC)

**AWS**:
- EC2, S3, RDS, Lambda, ECS, EKS
- VPC, Security Groups, IAM roles and policies
- CloudWatch for monitoring and alerting

**Azure**:
- Azure Kubernetes Service (AKS)
- Azure DevOps pipelines
- Azure Active Directory integration

### Monitoring & Observability
- Prometheus + Grafana stack
- ELK Stack: Elasticsearch, Logstash, Kibana
- Datadog, New Relic, Dynatrace
- Distributed tracing: Jaeger, Zipkin, OpenTelemetry
- SLOs, SLIs, SLAs — setting and measuring

### Networking & Security
- TCP/IP, DNS, HTTP/HTTPS, TLS/SSL
- Load balancers: Layer 4 vs Layer 7
- Firewalls, VPNs, NAT gateways
- Zero Trust network architecture
- Secret management: HashiCorp Vault, AWS Secrets Manager

## Common Interview Questions — DevOps Engineer

### Scenario Questions
1. "Walk me through your CI/CD pipeline design. How do you handle a failing deployment?"
2. "How would you set up monitoring and alerting for a microservices application?"
3. "Our Kubernetes cluster is running out of resources. How do you diagnose and resolve this?"
4. "Describe how you'd implement a zero-downtime deployment strategy."
5. "A critical service went down in production at 2 AM. Walk me through your incident response."

### Technical Questions
6. "What is the difference between a Kubernetes Deployment and a StatefulSet?"
7. "Explain Docker networking modes (bridge, host, overlay)."
8. "What is an HPA (Horizontal Pod Autoscaler) and when would you use it?"
9. "Describe the difference between mutable and immutable infrastructure."
10. "How do you handle secret rotation in a Kubernetes environment?"

### Architecture Questions
11. "Design a highly available, multi-region application architecture on IBM Cloud."
12. "How would you implement a GitOps workflow from scratch?"
13. "Explain your approach to infrastructure security hardening."

## Experience Level Expectations
- **Junior**: Basic Linux, Git, Docker, one cloud provider, Jenkins basics
- **Mid-Level**: Full CI/CD ownership, Kubernetes management, IaC with Terraform
- **Senior**: Architecture design, SRE practices, platform engineering, team mentoring
- **Lead**: Engineering excellence standards, FinOps, security policy, org-wide platform

## IBM-Specific DevOps Knowledge
- IBM Turbonomic for resource optimization
- IBM UrbanCode Deploy for enterprise release automation
- IBM Cloud Continuous Delivery (Toolchains)
- IBM Z/OS DevOps for mainframe environments
- RedHat OpenShift on IBM Cloud
