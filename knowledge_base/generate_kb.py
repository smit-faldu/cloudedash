"""
knowledge_base/generate_kb.py
==============================
Generates 18 sample Knowledge Base articles covering:
  - FAQs         (general product questions)
  - Troubleshooting (error resolution guides)
  - Billing      (plan & invoice questions)
  - API Docs     (integration & endpoint references)

Each article is written to:
    knowledge_base/articles/<article_id>.json

Run
---
    python -m knowledge_base.generate_kb

Output
------
    knowledge_base/articles/  (18 .json files)
    knowledge_base/articles/index.json  (manifest of all article IDs + metadata)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Article definitions
# ---------------------------------------------------------------------------

ARTICLES: list[dict] = [
    # -----------------------------------------------------------------------
    # CATEGORY: FAQ
    # -----------------------------------------------------------------------
    {
        "id": "FAQ-001",
        "category": "faq",
        "title": "What is CloudDash and what does it monitor?",
        "content": (
            "CloudDash is a B2B SaaS platform that provides real-time monitoring, "
            "intelligent alerting, and cost-optimization tools for cloud infrastructure. "
            "It supports AWS, Google Cloud Platform (GCP), and Microsoft Azure. "
            "CloudDash continuously collects metrics from your cloud resources — including "
            "EC2 instances, Lambda functions, S3 buckets, RDS databases, Kubernetes clusters, "
            "and serverless workloads — and surfaces anomalies, cost spikes, and SLA breaches "
            "in a unified dashboard.\n\n"
            "Key capabilities:\n"
            "- Real-time metrics with a 10-second polling interval on Growth and Scale plans.\n"
            "- Customisable alert rules with threshold, anomaly-detection, and forecast-based triggers.\n"
            "- Cost attribution by team, project tag, or resource group.\n"
            "- Automated remediation scripts for common issues (e.g., auto-stop idle instances).\n"
            "- SOC 2 Type II certified; all data is encrypted at rest (AES-256) and in transit (TLS 1.3)."
        ),
        "tags": ["overview", "features", "cloud providers", "monitoring"],
        "version": "2.4",
    },
    {
        "id": "FAQ-002",
        "category": "faq",
        "title": "Which cloud providers does CloudDash support?",
        "content": (
            "CloudDash currently supports the three major cloud providers:\n\n"
            "1. **Amazon Web Services (AWS)** – EC2, RDS, S3, Lambda, EKS, CloudFront, "
            "SQS, SNS, ElastiCache, and 140+ other services via CloudWatch integration.\n"
            "2. **Google Cloud Platform (GCP)** – Compute Engine, Cloud Run, GKE, BigQuery, "
            "Cloud SQL, Pub/Sub, and Cloud Storage via the GCP Monitoring API.\n"
            "3. **Microsoft Azure** – Virtual Machines, AKS, Azure Functions, Cosmos DB, "
            "Azure SQL, Blob Storage, and Service Bus via Azure Monitor.\n\n"
            "Multi-cloud views: You can combine metrics from all three providers into a single "
            "dashboard or alert rule. For example, you can trigger a PagerDuty alert when "
            "combined spend across AWS and GCP exceeds your monthly budget.\n\n"
            "On-premise / hybrid: Limited support via the CloudDash Agent (self-hosted binary). "
            "Refer to the Hybrid Deployment Guide for details."
        ),
        "tags": ["aws", "gcp", "azure", "cloud providers", "integration"],
        "version": "2.4",
    },
    {
        "id": "FAQ-003",
        "category": "faq",
        "title": "How do I invite team members and manage roles?",
        "content": (
            "CloudDash uses role-based access control (RBAC) with four built-in roles:\n\n"
            "| Role        | Description                                              |\n"
            "|-------------|----------------------------------------------------------|\n"
            "| Owner       | Full control, billing access, can delete the account.    |\n"
            "| Admin       | Manage integrations, users, and alert rules.             |\n"
            "| Editor      | Create/edit dashboards and alert rules; no billing access.|\n"
            "| Viewer      | Read-only access to dashboards and reports.              |\n\n"
            "**To invite a team member:**\n"
            "1. Go to Settings → Team Members → Invite Member.\n"
            "2. Enter the email address and select a role.\n"
            "3. Click Send Invite. The invitee receives an email with a secure link (expires in 48 hours).\n\n"
            "**SSO:** Growth and Scale plans support SAML 2.0 SSO (Okta, Google Workspace, Azure AD). "
            "Configure under Settings → Security → Single Sign-On.\n\n"
            "**Seat limits:** Starter: 3 seats. Growth: 15 seats. Scale: unlimited."
        ),
        "tags": ["rbac", "team", "users", "sso", "invite", "permissions"],
        "version": "2.4",
    },
    {
        "id": "FAQ-004",
        "category": "faq",
        "title": "What are the data retention policies for metrics and logs?",
        "content": (
            "CloudDash stores raw metrics, aggregated time-series data, and alert history "
            "according to the following retention schedule:\n\n"
            "| Plan        | Raw Metrics (1-min resolution) | Aggregated (5-min) | Alert History |\n"
            "|-------------|-------------------------------|---------------------|---------------|\n"
            "| Starter     | 7 days                        | 30 days             | 90 days       |\n"
            "| Growth      | 30 days                       | 6 months            | 1 year        |\n"
            "| Scale       | 90 days                       | 2 years             | 3 years       |\n"
            "| Enterprise  | Custom (up to 7 years)        | Custom              | Custom        |\n\n"
            "Log data (from the optional Log Aggregation add-on) is retained for 30 days on Growth "
            "and 90 days on Scale. Longer retention is available as a paid add-on ($0.10/GB/month).\n\n"
            "**Data export:** All data can be exported via the `/export` API endpoint in CSV, "
            "JSON, or Parquet format before retention expiry. See API-003 for the export endpoint docs."
        ),
        "tags": ["retention", "data", "metrics", "logs", "storage", "plan"],
        "version": "2.4",
    },
    {
        "id": "FAQ-005",
        "category": "faq",
        "title": "How does the cost optimisation feature work?",
        "content": (
            "CloudDash Cost Optimizer analyses your cloud spend and automatically identifies "
            "savings opportunities using three engines:\n\n"
            "1. **Idle Resource Detection** – Flags EC2, RDS, and Azure VM instances with CPU < 5% "
            "and network < 1 MB/s for 14+ consecutive days. CloudDash can auto-stop these with your approval.\n\n"
            "2. **Right-sizing Recommendations** – Compares your instances' actual peak utilization against "
            "available instance families and recommends downsizing (e.g., m5.2xlarge → m5.large) with "
            "projected monthly savings.\n\n"
            "3. **Reserved/Savings Plan Advisor** – Based on your 90-day usage patterns, recommends "
            "the optimal mix of 1-year Reserved Instances or Savings Plans to minimize on-demand costs. "
            "Typically saves 30–60% vs. on-demand.\n\n"
            "Cost Optimizer is enabled by default on Growth and Scale plans. On Starter, "
            "only the Idle Resource report is available (recommendations only, no auto-remediation)."
        ),
        "tags": ["cost", "optimization", "savings", "idle resources", "right-sizing"],
        "version": "2.4",
    },

    # -----------------------------------------------------------------------
    # CATEGORY: TROUBLESHOOTING
    # -----------------------------------------------------------------------
    {
        "id": "TS-001",
        "category": "troubleshooting",
        "title": "Error ERR-4012: Integration disconnected — AWS credentials expired",
        "content": (
            "**Error Code:** ERR-4012\n"
            "**Symptom:** The CloudDash dashboard shows a red 'Integration Error' banner. "
            "Metrics from AWS stop updating. The integration status page shows "
            "'AWS credentials expired or revoked'.\n\n"
            "**Root Cause:** CloudDash uses an IAM Role (cross-account assume-role) to access "
            "your AWS account. This error occurs when:\n"
            "  a) The IAM role was deleted or its trust policy was modified.\n"
            "  b) The external ID in the trust policy was changed.\n"
            "  c) The IAM policy attached to the role no longer grants required permissions.\n\n"
            "**Resolution Steps:**\n"
            "1. Go to Settings → Integrations → AWS → View Details.\n"
            "2. Click 'Re-authenticate'. CloudDash will display the IAM Role ARN and External ID.\n"
            "3. In your AWS Console, navigate to IAM → Roles → find the CloudDash role.\n"
            "4. Verify the Trust Relationship matches the CloudDash account ID (123456789012) "
            "   and the External ID shown in step 2.\n"
            "5. Verify the attached policy includes: cloudwatch:GetMetricData, "
            "   cloudwatch:ListMetrics, ec2:Describe*, rds:Describe*, s3:ListAllMyBuckets, "
            "   ce:GetCostAndUsage.\n"
            "6. Click 'Test Connection' in CloudDash. If it passes, metrics resume within 2 minutes.\n\n"
            "**If the issue persists:** Contact support with your Integration ID (visible on the "
            "integration details page) and the error timestamp."
        ),
        "tags": ["ERR-4012", "aws", "iam", "credentials", "integration error", "troubleshooting"],
        "version": "2.4",
    },
    {
        "id": "TS-002",
        "category": "troubleshooting",
        "title": "Error ERR-5001: Alert notifications not being delivered",
        "content": (
            "**Error Code:** ERR-5001\n"
            "**Symptom:** Alert rules are triggering (visible in Alert History) but "
            "notifications are not arriving via Slack, PagerDuty, or email.\n\n"
            "**Checklist — work through in order:**\n\n"
            "**1. Verify the notification channel is connected:**\n"
            "   - Settings → Notifications → check the channel status is 'Active'.\n"
            "   - Click 'Send Test' to trigger a test notification.\n\n"
            "**2. Slack-specific:**\n"
            "   - Confirm the CloudDash Slack App has been re-authorised after any Slack workspace "
            "     security policy changes.\n"
            "   - Verify the bot has permission to post in the target channel (not a private channel "
            "     the bot was not invited to).\n\n"
            "**3. PagerDuty-specific:**\n"
            "   - Confirm the Integration Key is still valid (PagerDuty → Services → Integrations).\n"
            "   - Check if PagerDuty has a maintenance window active that is suppressing alerts.\n\n"
            "**4. Email-specific:**\n"
            "   - Check spam/junk folders.\n"
            "   - Whitelist the sender domain: alerts@mail.clouddash.io.\n"
            "   - Enterprise customers: ensure your email gateway allows inbound SMTP from "
            "     149.72.0.0/14 (SendGrid IP range).\n\n"
            "**5. Alert rule misconfiguration:**\n"
            "   - Re-check the alert rule's 'Notify' section. Ensure the correct channel is selected "
            "     and 'Mute' is not enabled.\n"
            "   - Check for a 'Cooldown Period' setting that may be suppressing repeat notifications."
        ),
        "tags": ["ERR-5001", "alerts", "notifications", "slack", "pagerduty", "email", "troubleshooting"],
        "version": "2.4",
    },
    {
        "id": "TS-003",
        "category": "troubleshooting",
        "title": "Dashboard metrics show 'No Data' for last 24 hours",
        "content": (
            "**Symptom:** One or more dashboard panels display 'No Data' or a flat line despite "
            "resources being active in your cloud account.\n\n"
            "**Step 1 — Check integration health:**\n"
            "   Navigate to Settings → Integrations. If any integration shows a warning or error icon, "
            "   resolve that first (see ERR-4012 for AWS credential issues).\n\n"
            "**Step 2 — Verify the metric namespace:**\n"
            "   For AWS, some metrics (e.g., ECS container metrics) require the CloudWatch Agent "
            "   to be installed on the host and are not available via the standard CloudWatch API. "
            "   Check Settings → Integrations → AWS → Enhanced Monitoring.\n\n"
            "**Step 3 — Check time-zone alignment:**\n"
            "   CloudDash stores metrics in UTC. If your dashboard's time picker is set to a local "
            "   timezone with a large offset, you may be viewing a future time range. "
            "   Set the picker to 'Last 3 hours' to confirm data is arriving.\n\n"
            "**Step 4 — Check plan limits:**\n"
            "   Starter plan accounts are limited to 50 monitored resources. If you have exceeded "
            "   this limit, newer resources are silently excluded. Navigate to Settings → Usage to "
            "   view your resource count.\n\n"
            "**Step 5 — Wait for backfill:**\n"
            "   After reconnecting an integration, CloudDash backfills up to 72 hours of historical "
            "   data. This process can take 15–30 minutes for large accounts."
        ),
        "tags": ["no data", "dashboard", "metrics", "troubleshooting", "backfill", "integration"],
        "version": "2.4",
    },
    {
        "id": "TS-004",
        "category": "troubleshooting",
        "title": "Error ERR-3007: FAISS index out of memory on self-hosted agent",
        "content": (
            "**Error Code:** ERR-3007 (self-hosted CloudDash Agent only)\n"
            "**Symptom:** The CloudDash Agent process crashes with ERR-3007 and the log contains "
            "'Insufficient memory to build local metric index'.\n\n"
            "**Root Cause:** The CloudDash Agent uses an in-memory metric buffer for local "
            "aggregation before shipping data. This buffer grows proportionally to the number of "
            "monitored resources × metric dimensions × polling interval.\n\n"
            "**Resolution Options:**\n\n"
            "**Option A — Increase agent memory allocation:**\n"
            "   Edit /etc/clouddash-agent/agent.yaml:\n"
            "   ```\n"
            "   memory:\n"
            "     max_buffer_mb: 512   # increase from default 256\n"
            "   ```\n"
            "   Restart: sudo systemctl restart clouddash-agent\n\n"
            "**Option B — Reduce monitored resource scope:**\n"
            "   Use include/exclude filters in agent.yaml to monitor only critical resources:\n"
            "   ```\n"
            "   filters:\n"
            "     include_tags:\n"
            "       - Environment: production\n"
            "   ```\n\n"
            "**Option C — Increase polling interval:**\n"
            "   Change polling_interval_seconds from 10 to 60 to reduce buffer pressure.\n\n"
            "**Minimum recommended host specs:** 2 vCPU, 4 GB RAM for up to 500 resources."
        ),
        "tags": ["ERR-3007", "agent", "memory", "self-hosted", "troubleshooting", "OOM"],
        "version": "2.4",
    },
    {
        "id": "TS-005",
        "category": "troubleshooting",
        "title": "Alert rule fires continuously (alert storm / flapping)",
        "content": (
            "**Symptom:** An alert rule is triggering and resolving repeatedly in short cycles, "
            "flooding your notification channel with dozens of messages per hour.\n\n"
            "**Root Cause:** Alert flapping occurs when a metric oscillates around the threshold "
            "value. Without hysteresis (a recovery threshold), the alert flips between FIRING "
            "and RESOLVED on every evaluation cycle.\n\n"
            "**Resolution:**\n\n"
            "**1. Add a recovery threshold (hysteresis):**\n"
            "   In the alert rule editor, set a separate 'Resolve Condition' that is more lenient "
            "   than the fire condition. Example:\n"
            "   - Fire when: CPU > 85% for 5 minutes\n"
            "   - Resolve when: CPU < 70% for 10 minutes\n\n"
            "**2. Increase the evaluation window:**\n"
            "   Change 'Evaluate every: 1 minute' to '5 minutes'. "
            "   This smooths out brief spikes.\n\n"
            "**3. Enable the built-in flap detection:**\n"
            "   Alert rule → Advanced Options → Enable Flap Detection. "
            "   This suppresses notifications if the alert state changes more than 4 times "
            "   in a 20-minute window.\n\n"
            "**4. Set a cooldown period:**\n"
            "   Alert rule → Notifications → Cooldown: 30 minutes. "
            "   Prevents repeat notifications until the cooldown expires even if the alert "
            "   continues to fire."
        ),
        "tags": ["alert storm", "flapping", "alert rule", "hysteresis", "cooldown", "troubleshooting"],
        "version": "2.4",
    },

    # -----------------------------------------------------------------------
    # CATEGORY: BILLING
    # -----------------------------------------------------------------------
    {
        "id": "BIL-001",
        "category": "billing",
        "title": "CloudDash pricing plans overview",
        "content": (
            "CloudDash offers four subscription tiers designed for teams of all sizes:\n\n"
            "| Plan       | Price           | Resources | Seats | Key Features                            |\n"
            "|------------|-----------------|-----------|-------|-----------------------------------------|\n"
            "| Starter    | $49 / month     | Up to 50  | 3     | Basic monitoring, 7-day metric history  |\n"
            "| Growth     | $149 / month    | Up to 250 | 15    | SSO, anomaly detection, 30-day history  |\n"
            "| Scale      | $499 / month    | Unlimited | Unlimited | Custom alerts, 90-day raw history   |\n"
            "| Enterprise | Custom pricing  | Unlimited | Unlimited | SLA, dedicated support, private cloud |\n\n"
            "**Annual billing:** 20% discount on all plans when paid annually (Growth and above).\n\n"
            "**Add-ons (available on Growth and above):**\n"
            "- Log Aggregation: $29/month per 10 GB/day ingestion.\n"
            "- Extended Retention: $0.10/GB/month beyond plan limits.\n"
            "- Premium Support: $299/month — 1-hour SLA response time.\n\n"
            "**Free trial:** 14-day free trial on all plans, no credit card required. "
            "Trial accounts default to Growth plan features. After trial, the account "
            "downgrades to Starter unless upgraded."
        ),
        "tags": ["pricing", "plans", "billing", "starter", "growth", "scale", "enterprise"],
        "version": "2.4",
    },
    {
        "id": "BIL-002",
        "category": "billing",
        "title": "How to upgrade or downgrade your subscription plan",
        "content": (
            "**Upgrading your plan:**\n"
            "Plan upgrades take effect immediately. You are billed a pro-rated amount "
            "for the remainder of the current billing cycle.\n\n"
            "Steps:\n"
            "1. Go to Settings → Billing → Change Plan.\n"
            "2. Select the new plan and click 'Upgrade Now'.\n"
            "3. Confirm the pro-rated charge shown in the preview.\n"
            "4. The upgrade activates within 60 seconds.\n\n"
            "**Downgrading your plan:**\n"
            "Plan downgrades take effect at the start of the next billing cycle. "
            "You will retain current plan features until then.\n\n"
            "Important considerations before downgrading:\n"
            "- If your current resource count exceeds the lower plan's limit, "
            "  CloudDash will automatically stop monitoring the excess resources "
            "  (oldest integrations are paused first).\n"
            "- SSO and advanced security features are disabled immediately on downgrade to Starter.\n"
            "- Historical data beyond the new plan's retention period is queued for deletion "
            "  after 30 days (giving you time to export).\n\n"
            "**Cancellation:**\n"
            "You can cancel at any time from Settings → Billing → Cancel Subscription. "
            "Your account remains active until the end of the paid billing period. "
            "Data is retained for 90 days post-cancellation, after which it is permanently deleted."
        ),
        "tags": ["upgrade", "downgrade", "plan change", "cancel", "billing", "subscription"],
        "version": "2.4",
    },
    {
        "id": "BIL-003",
        "category": "billing",
        "title": "Understanding your CloudDash invoice",
        "content": (
            "CloudDash invoices are generated on the 1st of each month (for monthly plans) "
            "or annually on the renewal date (for annual plans).\n\n"
            "**Invoice line items:**\n"
            "- **Base subscription:** Your plan's monthly/annual fee.\n"
            "- **Pro-rated charges:** Adjustments for mid-cycle upgrades.\n"
            "- **Add-on services:** Log Aggregation, Extended Retention, Premium Support.\n"
            "- **Overage charges:** If your resource count exceeded plan limits during the period "
            "  (only applicable on metered Enterprise plans).\n\n"
            "**Accessing your invoices:**\n"
            "Settings → Billing → Invoice History. All invoices are downloadable as PDF.\n\n"
            "**Payment methods accepted:**\n"
            "Visa, Mastercard, American Express, and ACH bank transfer (Enterprise only).\n\n"
            "**Failed payment handling:**\n"
            "If a payment fails, CloudDash sends an email notification and retries after "
            "3, 7, and 14 days. After 3 failed retries, the account is suspended "
            "(dashboards become read-only; data collection pauses). "
            "Account access is restored immediately upon successful payment.\n\n"
            "**VAT/Tax:** CloudDash adds applicable VAT/GST for EU and AU customers automatically. "
            "To add a VAT ID, go to Settings → Billing → Tax Information."
        ),
        "tags": ["invoice", "billing", "payment", "failed payment", "vat", "tax"],
        "version": "2.4",
    },
    {
        "id": "BIL-004",
        "category": "billing",
        "title": "Refund and dispute policy",
        "content": (
            "**Refund eligibility:**\n"
            "CloudDash operates a no-refund policy on subscription fees once a billing period "
            "has started, except in the following circumstances:\n\n"
            "1. **Service outage SLA breach:** If CloudDash's uptime falls below the "
            "   guaranteed SLA (99.9% monthly on Growth; 99.95% on Scale/Enterprise), "
            "   you are entitled to a service credit equal to 10× the downtime as a percentage "
            "   of your monthly bill.\n"
            "2. **Duplicate charge:** If you were charged twice for the same billing period "
            "   due to a CloudDash system error, a full refund of the duplicate charge is issued "
            "   within 5 business days.\n"
            "3. **Annual plan cancellation within 14 days:** If you cancel an annual plan "
            "   within 14 days of the renewal date, a pro-rated refund for unused months is issued.\n\n"
            "**How to request a refund:**\n"
            "Refunds cannot be processed automatically — they require review by the Billing Team. "
            "Submit a refund request via the Support portal, including:\n"
            "- Your account email and customer ID (format: CLD-XXXXX)\n"
            "- The invoice number in question\n"
            "- The reason for the refund request\n\n"
            "Refund requests are reviewed within 3 business days. "
            "Approved refunds are credited to the original payment method within 5–10 business days."
        ),
        "tags": ["refund", "dispute", "billing policy", "sla credit", "duplicate charge"],
        "version": "2.4",
    },

    # -----------------------------------------------------------------------
    # CATEGORY: API DOCS
    # -----------------------------------------------------------------------
    {
        "id": "API-001",
        "category": "api_docs",
        "title": "Authentication — API Keys and OAuth 2.0",
        "content": (
            "CloudDash offers two authentication mechanisms for API access:\n\n"
            "**1. API Keys (recommended for server-to-server)**\n"
            "Generate API keys from Settings → Developer → API Keys.\n"
            "Pass the key in every request header:\n"
            "```\n"
            "Authorization: Bearer cd_live_xxxxxxxxxxxxxxxxxxxx\n"
            "```\n"
            "Key types:\n"
            "- **Read-only:** Can GET metrics, dashboards, alerts. Cannot modify resources.\n"
            "- **Read-write:** Full API access. Rotate every 90 days.\n"
            "- **Admin:** Includes user management and billing endpoints. Restrict carefully.\n\n"
            "Key rate limits:\n"
            "- Starter: 100 requests/minute\n"
            "- Growth: 500 requests/minute\n"
            "- Scale/Enterprise: 2,000 requests/minute (higher on request)\n\n"
            "**2. OAuth 2.0 (recommended for user-facing integrations)**\n"
            "CloudDash supports the Authorization Code flow with PKCE.\n"
            "- Authorization endpoint: https://auth.clouddash.io/oauth/authorize\n"
            "- Token endpoint: https://auth.clouddash.io/oauth/token\n"
            "- Available scopes: metrics:read, alerts:write, billing:read, admin:write\n\n"
            "Access tokens expire after 1 hour. Use the refresh token to obtain a new "
            "access token (refresh tokens expire after 30 days or on first use)."
        ),
        "tags": ["api", "authentication", "api key", "oauth", "bearer token", "rate limiting"],
        "version": "2.4",
    },
    {
        "id": "API-002",
        "category": "api_docs",
        "title": "Metrics Query API — GET /v2/metrics",
        "content": (
            "Retrieve time-series metric data for monitored resources.\n\n"
            "**Endpoint:** GET https://api.clouddash.io/v2/metrics\n\n"
            "**Request Parameters:**\n"
            "| Parameter      | Type     | Required | Description                                    |\n"
            "|----------------|----------|----------|------------------------------------------------|\n"
            "| resource_id    | string   | Yes      | CloudDash resource ID (e.g., aws-ec2-i-abc123) |\n"
            "| metric_name    | string   | Yes      | Metric name (e.g., CPUUtilization)             |\n"
            "| start_time     | ISO 8601 | Yes      | Start of time range (UTC)                      |\n"
            "| end_time       | ISO 8601 | Yes      | End of time range (UTC)                        |\n"
            "| resolution     | string   | No       | '1m', '5m', '1h', '1d' (default: '5m')        |\n"
            "| aggregation    | string   | No       | 'avg', 'max', 'min', 'sum', 'p99' (default: 'avg') |\n\n"
            "**Example Request:**\n"
            "```bash\n"
            "curl -H 'Authorization: Bearer cd_live_xxx' \\\n"
            "  'https://api.clouddash.io/v2/metrics?resource_id=aws-ec2-i-0abc123&"
            "metric_name=CPUUtilization&start_time=2024-01-01T00:00:00Z&end_time=2024-01-01T01:00:00Z'\n"
            "```\n\n"
            "**Response (200 OK):**\n"
            "```json\n"
            "{\n"
            "  \"resource_id\": \"aws-ec2-i-0abc123\",\n"
            "  \"metric_name\": \"CPUUtilization\",\n"
            "  \"unit\": \"Percent\",\n"
            "  \"datapoints\": [\n"
            "    {\"timestamp\": \"2024-01-01T00:00:00Z\", \"value\": 42.3},\n"
            "    {\"timestamp\": \"2024-01-01T00:05:00Z\", \"value\": 38.1}\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "**Error Codes:**\n"
            "- 400: Invalid parameters (check start_time < end_time, valid metric_name).\n"
            "- 404: Resource not found or not monitored by this account.\n"
            "- 429: Rate limit exceeded. Retry after the number of seconds in Retry-After header."
        ),
        "tags": ["api", "metrics", "GET", "time-series", "CPUUtilization", "query"],
        "version": "2.4",
    },
    {
        "id": "API-003",
        "category": "api_docs",
        "title": "Alerts API — Create and manage alert rules via API",
        "content": (
            "Programmatically create, update, and delete alert rules.\n\n"
            "**Base URL:** https://api.clouddash.io/v2/alerts\n\n"
            "**Create an alert rule — POST /v2/alerts**\n"
            "```json\n"
            "{\n"
            "  \"name\": \"High CPU on production servers\",\n"
            "  \"condition\": {\n"
            "    \"metric\": \"CPUUtilization\",\n"
            "    \"operator\": \"gt\",\n"
            "    \"threshold\": 85,\n"
            "    \"for_minutes\": 5\n"
            "  },\n"
            "  \"resource_filter\": {\n"
            "    \"tags\": {\"Environment\": \"production\"}\n"
            "  },\n"
            "  \"severity\": \"high\",\n"
            "  \"notify\": [\n"
            "    {\"type\": \"slack\", \"channel_id\": \"C01234ABCDE\"},\n"
            "    {\"type\": \"pagerduty\", \"integration_key\": \"abc123\"}\n"
            "  ],\n"
            "  \"cooldown_minutes\": 30\n"
            "}\n"
            "```\n\n"
            "**Response (201 Created):** Returns the full alert rule object with its id field.\n\n"
            "**Manage rules:**\n"
            "- GET /v2/alerts — list all rules (paginated, max 100 per page)\n"
            "- GET /v2/alerts/{id} — get a specific rule\n"
            "- PATCH /v2/alerts/{id} — update fields (partial update supported)\n"
            "- DELETE /v2/alerts/{id} — delete a rule\n"
            "- POST /v2/alerts/{id}/mute — mute for a duration ({\"minutes\": 60})\n\n"
            "**Webhook notifications:**\n"
            "Set notify type to 'webhook' with a url field. CloudDash will POST a JSON payload "
            "to your endpoint when the alert fires. Payloads are signed with HMAC-SHA256 "
            "using your webhook secret — verify the X-CloudDash-Signature header."
        ),
        "tags": ["api", "alerts", "POST", "webhook", "create alert", "manage alerts"],
        "version": "2.4",
    },
    {
        "id": "API-004",
        "category": "api_docs",
        "title": "Data Export API — Bulk export metrics and logs",
        "content": (
            "Export historical metric data or log entries in bulk for external analysis.\n\n"
            "**Endpoint:** POST https://api.clouddash.io/v2/export\n\n"
            "**Request Body:**\n"
            "```json\n"
            "{\n"
            "  \"type\": \"metrics\",\n"
            "  \"resource_ids\": [\"aws-ec2-i-0abc123\", \"aws-rds-db-xyz\"],\n"
            "  \"metrics\": [\"CPUUtilization\", \"NetworkIn\", \"DiskWriteOps\"],\n"
            "  \"start_time\": \"2024-01-01T00:00:00Z\",\n"
            "  \"end_time\": \"2024-01-31T23:59:59Z\",\n"
            "  \"format\": \"parquet\",\n"
            "  \"resolution\": \"5m\",\n"
            "  \"destination\": {\n"
            "    \"type\": \"s3\",\n"
            "    \"bucket\": \"my-export-bucket\",\n"
            "    \"prefix\": \"clouddash-exports/jan-2024/\"\n"
            "  }\n"
            "}\n"
            "```\n\n"
            "**Export formats:** json, csv, parquet\n"
            "**Destinations:** s3, gcs, azure_blob, or presigned_url (for direct download)\n\n"
            "**Async processing:** Large exports return HTTP 202 Accepted with an export_job_id. "
            "Poll GET /v2/export/{export_job_id} for status (pending → processing → completed → failed). "
            "Completed exports include a download URL or confirm delivery to the specified destination.\n\n"
            "**Limits:**\n"
            "- Max time range per export: 90 days (Growth), 2 years (Scale/Enterprise)\n"
            "- Max resources per export: 1,000\n"
            "- Concurrent exports: 3 (Growth), 10 (Scale/Enterprise)"
        ),
        "tags": ["api", "export", "bulk", "parquet", "s3", "data export", "async"],
        "version": "2.4",
    },
]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def generate_articles(output_dir: Path | None = None) -> Path:
    """
    Write all articles to *output_dir* as individual JSON files,
    plus an ``index.json`` manifest.

    Returns the path to the output directory.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "articles"

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    for article in ARTICLES:
        article_path = output_dir / f"{article['id']}.json"
        article_path.write_text(json.dumps(article, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote %s", article_path.name)

        manifest.append(
            {
                "id": article["id"],
                "category": article["category"],
                "title": article["title"],
                "tags": article["tags"],
                "version": article["version"],
                "file": f"{article['id']}.json",
            }
        )

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote index.json with %d articles", len(manifest))

    return output_dir


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = generate_articles()
    print(f"OK  Generated {len(ARTICLES)} KB articles in: {out.resolve()}")
