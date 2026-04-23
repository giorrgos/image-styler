# Image Styler - Simple Metrics Guide

## 📊 The 4 Essential Metrics

### 1. **API Success Rate**
- **Query**: `sum(rate(image_styler_api_calls_total{status="success"}[5m])) / sum(rate(image_styler_api_calls_total[5m])) * 100`
- **What it shows**: Percentage of successful API calls (0-100%)
- **Good value**: > 95%
- **Dashboard**: Green gauge showing success percentage

### 2. **Total API Calls**
- **Query**: `sum(image_styler_api_calls_total)`
- **What it shows**: Total number of API calls since app started
- **Use case**: Track overall usage
- **Dashboard**: Blue stat panel with call count

### 3. **Average API Call Duration**
- **Query**: `rate(image_styler_api_call_duration_seconds_sum[5m]) / rate(image_styler_api_call_duration_seconds_count[5m])`
- **What it shows**: Average time for API calls in seconds
- **Good value**: < 15 seconds
- **Dashboard**: Time series graph with threshold lines (green < 20s, yellow < 30s, red > 30s)

### 4. **Total Errors**
- **Query**: `sum(image_styler_api_errors_total)`
- **What it shows**: Total number of errors since app started
- **Good value**: 0
- **Dashboard**: Stat panel with color coding (green = 0, yellow = 1-4, red ≥ 5)

---

## 🚀 Quick Start

### View Dashboard
1. Open **http://localhost:3000**
2. Click **"Dashboards"** → **"Image Styler - Overview"**
3. See all 4 metrics in a clean 2x2 grid
4. Auto-refreshes every 10 seconds

### Use Prometheus Directly
1. Open **http://localhost:9090**
2. Copy any query from above
3. Paste and click **"Execute"**

---

## 📈 Dashboard Layout

```
┌─────────────────────────────────────────────┐
│   API Success Rate   │   Total API Calls    │
│      (Gauge)         │      (Stat)          │
├─────────────────────────────────────────────┤
│ Average API Duration │   Total Errors       │
│   (Time Series)      │      (Stat)          │
└─────────────────────────────────────────────┘
```

---

## 💡 Tips

- **Time Range**: Adjust in top-right (default: last 1 hour)
- **Auto-Refresh**: Updates every 10 seconds
- **Full Screen**: Click panel title → View → Full Screen
- **Share**: Click share icon on any panel to export

---

## 🔍 Understanding Your Metrics

| Metric | Great | Good | Warning | Critical |
|--------|-------|------|---------|----------|
| Success Rate | 100% | 95-99% | 90-95% | < 90% |
| Duration | < 10s | 10-20s | 20-30s | > 30s |
| Errors | 0 | 0 | 1-4 | ≥ 5 |

---

## 📚 More Information

For detailed queries and troubleshooting, see:
- **PROMETHEUS_QUERIES.md** - Full documentation
- **README.md** - Complete setup guide
