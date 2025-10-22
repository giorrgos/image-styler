# Prometheus Queries Reference

This document contains the essential Prometheus queries to monitor your Image Styler application.

## Quick Access URLs

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Application**: http://localhost:8502

---

## Core Metrics (4 Essential Queries)

### 1. API Success Rate (percentage)
```promql
sum(rate(image_styler_api_calls_total{status="success"}[5m])) / sum(rate(image_styler_api_calls_total[5m])) * 100
```
**What it shows**: Percentage of successful API calls over the last 5 minutes  
**Good value**: > 95%  
**Alert threshold**: < 90%

---

### 2. Total API Calls
```promql
sum(image_styler_api_calls_total)
```
**What it shows**: Total number of API calls made since the application started  
**Use case**: Track overall usage and volume

---

### 3. Average API Call Duration (seconds)
```promql
rate(image_styler_api_call_duration_seconds_sum[5m]) / rate(image_styler_api_call_duration_seconds_count[5m])
```
**What it shows**: Average time taken for API calls over the last 5 minutes  
**Good value**: < 15 seconds  
**Alert threshold**: > 30 seconds

---

### 4. Total Errors
```promql
sum(image_styler_api_errors_total)
```
**What it shows**: Total number of errors encountered since the application started  
**Good value**: 0  
**Alert threshold**: > 0 (investigate any errors)

---

## How to Use These Queries

### In Prometheus UI (http://localhost:9090):
1. Copy any query above
2. Paste it in the query box at the top
3. Click **"Execute"** button
4. Switch between **Table** and **Graph** views to visualize data
5. Adjust the time range using the date picker in the top-right

### In Grafana Dashboard (http://localhost:3000):
1. Open Grafana at http://localhost:3000
2. Click **"Dashboards"** in the left menu
3. Select **"Image Styler - Overview"**
4. All 4 metrics are pre-configured and auto-refreshing

### In Grafana Explore (http://localhost:3000):
1. Click **"Explore"** (compass icon) in the left menu
2. Select **"Prometheus"** as the datasource
3. Paste any query from above
4. Click **"Run query"**
5. Experiment with different time ranges and visualization options

---

## Understanding Your Metrics

### API Success Rate
- **100%** = Perfect! All API calls succeeded
- **95-99%** = Good, some occasional failures
- **< 95%** = Investigation needed, too many failures

### Total API Calls
- Shows cumulative usage over time
- Useful for understanding application adoption
- Resets when the application restarts

### Average API Call Duration
- **< 10s** = Excellent performance
- **10-20s** = Good, typical for image generation
- **20-30s** = Acceptable but slow
- **> 30s** = Poor, investigate bottlenecks

### Total Errors
- **0** = Perfect! No errors
- **1-5** = A few errors, monitor the trend
- **> 5** = Something is wrong, check logs immediately

---

## Quick Reference Card

Copy this to keep handy:

```
┌─────────────────────────────────────────────────────────────────┐
│                  IMAGE STYLER METRICS                           │
├─────────────────────────────────────────────────────────────────┤
│ 1. API Success Rate:                                            │
│    sum(rate(image_styler_api_calls_total{status="success"}[5m]))│
│    / sum(rate(image_styler_api_calls_total[5m])) * 100         │
│                                                                  │
│ 2. Total API Calls:                                             │
│    sum(image_styler_api_calls_total)                            │
│                                                                  │
│ 3. Average API Call Duration:                                   │
│    rate(image_styler_api_call_duration_seconds_sum[5m])         │
│    / rate(image_styler_api_call_duration_seconds_count[5m])     │
│                                                                  │
│ 4. Total Errors:                                                │
│    sum(image_styler_api_errors_total)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### No data showing in Prometheus?
1. **Check if app is running**: http://localhost:8502
2. **Verify telemetry logs**: Look for "Telemetry enabled" in app output
3. **Generate some metrics**: Upload and process an image
4. **Check metrics endpoint**: 
   ```bash
   curl http://localhost:8889/metrics | grep image_styler
   ```
5. **Verify Prometheus targets**: http://localhost:9090/targets (should show "UP")

### Dashboard not showing in Grafana?
1. **Restart Grafana**: 
   ```bash
   docker-compose restart grafana
   ```
2. **Check Grafana logs**: 
   ```bash
   docker-compose logs grafana
   ```
3. **Manually refresh**: Go to Dashboards → Browse → Image Styler - Overview

### Metrics are stale or not updating?
1. **Check time range**: Make sure you're viewing recent data (last 1 hour)
2. **Process a new image**: This will generate fresh metrics
3. **Verify auto-refresh**: Dashboard should refresh every 10 seconds
4. **Check otel-collector**: 
   ```bash
   docker-compose logs otel-collector
   ```

---

## Additional Resources

- **Prometheus Documentation**: https://prometheus.io/docs/
- **Grafana Documentation**: https://grafana.com/docs/
- **OpenTelemetry**: https://opentelemetry.io/docs/
- **PromQL Tutorial**: https://prometheus.io/docs/prometheus/latest/querying/basics/

