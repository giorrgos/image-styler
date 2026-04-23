# Image Styler

A Streamlit web application that allows you to upload images and apply various artistic styles to transform them using AI models from Replicate. Features comprehensive observability with OpenTelemetry, Prometheus, Tempo, and Grafana.

## Features

- 📸 **Image Upload**: Support for common image formats (JPG, PNG, JPEG)
- 🔄 **EXIF Orientation**: Automatically fixes image orientation
- 🎨 **AI Style Transfer**: Apply artistic styles using AI models from Replicate (FLUX Kontext Pro, Qwen Image Edit Plus)
- 🔀 **Model Toggle**: Switch between image models directly in the UI
- 🖼️ **Real-time Preview**: See both original and styled images side by side
- 📊 **Full Observability**: Distributed tracing, metrics, and logging with Grafana stack
- � **Performance Monitoring**: Track API call durations, error rates, and request flows
- 🌐 **Web Interface**: Easy-to-use browser-based application

## Architecture

The application is organized into clean, modular components:

```
image-styler/
├── app.py                    # Streamlit UI
├── image_generator.py        # Replicate API interactions
├── image_utils.py            # Image processing utilities
├── logger_config.py          # Logging configuration
├── telemetry_config.py       # OpenTelemetry setup
├── docker-compose.yml        # Observability stack
└── config/                   # Configuration files
    ├── otel-collector-config.yml
    ├── prometheus.yml
    ├── tempo.yml
    └── grafana/
        ├── datasources.yml
        ├── dashboards.yml
        └── dashboards/
            └── image-styler-dashboard.json
```

## Prerequisites

- **Python 3.12+**
- **uv** - Python package manager (install from [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/))
- **Docker & Docker Compose** (for observability stack)
- **Replicate API Token** - Get one at [https://replicate.com](https://replicate.com)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/giorrgos/image-styler.git
cd image-styler
```

### 2. Set Up Python Environment

```bash
# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Required: Your Replicate API token
REPLICATE_API_TOKEN=your_replicate_api_token_here

# Optional: OpenTelemetry collector endpoint (defaults to localhost:4317)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

## Usage

### Option 1: Run with Full Observability Stack (Recommended)

This starts the complete Grafana observability stack with tracing, metrics, and visualization.

#### Start the Observability Stack

```bash
# Start Grafana, Tempo, Prometheus, and OpenTelemetry Collector
docker-compose up -d

# Verify all services are running
docker-compose ps
```

You should see 4 services running:
- `otel-collector` - Receives telemetry from the app
- `tempo` - Stores distributed traces
- `prometheus` - Stores metrics
- `grafana` - Visualization dashboard

#### Start the Application

```bash
uv run streamlit run app.py
```

#### Access the Services

- **Application**: [http://localhost:8502](http://localhost:8502) (or check terminal output for actual port)
- **Grafana Dashboard**: [http://localhost:3000/d/image-styler-overview](http://localhost:3000/d/image-styler-overview) (auto-login enabled)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)

#### Using Grafana Dashboard

1. **Open the Dashboard** at [http://localhost:3000/d/image-styler-overview](http://localhost:3000/d/image-styler-overview)
   - Or: Click **"Dashboards"** in left sidebar → **"Image Styler - Overview"**

2. **View the 4 Essential Metrics**:
   - **API Success Rate** - Percentage of successful API calls (gauge, target: >95%)
   - **Total API Calls** - Cumulative count of all API calls
   - **Average API Call Duration** - Time series showing API latency
   - **Total Errors** - Error count (color-coded: green=0, yellow=1-4, red≥5)

3. **Dashboard Features**:
   - Auto-refreshes every 10 seconds
   - Time range selector in top-right (default: last 1 hour)
   - Click any panel title → **View** → **Full Screen** for detailed view

4. **Explore Traces**:
   - Click **"Explore"** (compass icon) → Select **"Tempo"** datasource
   - Search for `image_styling_request` to see complete request traces
   - View span details: fix_orientation, replicate_api_call

5. **Custom Prometheus Queries**:
   - Click **"Explore"** → Select **"Prometheus"** datasource
   - See `PROMETHEUS_QUERIES.md` or `METRICS_QUICK_START.md` for query examples

#### Stop the Observability Stack

```bash
# Stop services but keep data
docker-compose stop

# Stop and remove services (keeps volumes with data)
docker-compose down

# Stop and remove everything including data
docker-compose down -v
```

---

### Option 2: Run Without Observability (Simple Mode)

If you don't have Docker or don't need observability, the app will run fine without it.

```bash
# Just start the application
uv run streamlit run app.py
```

The app will detect that telemetry services aren't available and run without instrumentation.

---

## Observability

### How the Stack Fits Together

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit App                           │
│  Generates traces + metrics via OpenTelemetry SDK           │
└────────────────────┬────────────────────────────────────────┘
                     │ OTLP gRPC (localhost:4317)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               OpenTelemetry Collector                       │
│  Receives all telemetry and fans it out:                    │
│    traces  ──► OTLP export ──► Tempo                       │
│    metrics ──► Prometheus exporter on :8889                 │
└────────────────────┬───────────────────┬────────────────────┘
                     │                   │
          OTLP gRPC  │                   │ Prometheus scrapes :8889
                     ▼                   ▼
              ┌──────────┐        ┌─────────────┐
              │  Tempo   │        │ Prometheus  │
              │ (traces) │        │  (metrics)  │
              └────┬─────┘        └──────┬──────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
                       ┌─────────┐
                       │ Grafana │
                       │ queries │
                       │  both   │
                       └─────────┘
```

Each image generation request produces:
- **One trace** with three spans: `image_styling_request` → `fix_orientation` + `replicate_api_call`
- **Metrics increments**: `api_calls_total`, `api_call_duration_seconds`, `images_processed_total`

---

### OpenTelemetry Collector

The collector is the central hub. It receives everything the app sends and routes it:
- **Traces** → forwarded via OTLP to Tempo for storage
- **Metrics** → exposed as a Prometheus scrape endpoint on `:8889`; Prometheus pulls from there every 15 seconds

The collector also adds the `image_styler_` namespace prefix to all metric names, which is why Prometheus metrics appear as `image_styler_api_calls_total` rather than just `api_calls_total`.

Check collector logs if data is missing:
```bash
docker-compose logs -f otel-collector
```

---

### Tempo (Distributed Traces)

Tempo stores request traces. Each trace shows the full lifecycle of one image generation — how long orientation fixing took, how long the Replicate API call took, and whether it succeeded.

**Access traces in Grafana:**
1. Go to [http://localhost:3000](http://localhost:3000) → **Explore** (compass icon)
2. Select **Tempo** as the datasource
3. Set **Query type** to `Search`, filter by service name `image-styler`
4. Click a trace to expand the span waterfall

**Span attributes recorded per request:**

| Span | Attributes |
|------|-----------|
| `image_styling_request` | `filename`, `file_size_mb`, `file_type`, `has_prompt`, `model` |
| `fix_orientation` | `orientation_fixed`, `image_width`, `image_height` |
| `replicate_api_call` | `model`, `prompt_length`, `success`, `duration_seconds`, `error_message` |

---

### Prometheus (Metrics)

Prometheus stores time-series metrics. Access the UI at [http://localhost:9090](http://localhost:9090) to run ad-hoc queries, or check [http://localhost:9090/targets](http://localhost:9090/targets) to confirm the collector scrape target is `UP`.

**Metrics emitted by the app** (all prefixed `image_styler_` by the collector):

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `image_styler_api_calls_total` | Counter | `model`, `status` (`success`/`failure`) | Completed API calls |
| `image_styler_api_errors_total` | Counter | `model`, `error_type` | API errors by type |
| `image_styler_api_call_duration_seconds` | Histogram | `model` | Replicate API response time |
| `image_styler_images_processed_total` | Counter | `model`, `status` | Images processed end-to-end |
| `image_styler_request_duration_seconds` | Histogram | `model`, `status` | Full request duration |

**Sample queries** (paste into [http://localhost:9090](http://localhost:9090) or Grafana Explore → Prometheus):

```promql
# Success rate over the last 5 minutes (%)
sum(rate(image_styler_api_calls_total{status="success"}[5m]))
  / sum(rate(image_styler_api_calls_total[5m])) * 100

# Average Replicate API response time (seconds)
rate(image_styler_api_call_duration_seconds_sum[5m])
  / rate(image_styler_api_call_duration_seconds_count[5m])

# Call volume split by model
sum by (model) (rate(image_styler_api_calls_total[5m]))

# Error breakdown by type
sum by (error_type) (image_styler_api_errors_total)

# 95th percentile API duration
histogram_quantile(0.95, rate(image_styler_api_call_duration_seconds_bucket[5m]))
```

For the full query reference see `PROMETHEUS_QUERIES.md`.

---

### Grafana (Visualisation)

Grafana is pre-provisioned with both datasources (Prometheus and Tempo) and the Image Styler dashboard. No login required.

**Pre-built dashboard** — [http://localhost:3000/d/image-styler-overview](http://localhost:3000/d/image-styler-overview):

```
┌──────────────────────┬──────────────────────┐
│   API Success Rate   │   Total API Calls    │
│   (gauge, target     │   (stat counter)     │
│    ≥ 95%)            │                      │
├──────────────────────┼──────────────────────┤
│ Avg API Duration     │   Total Errors       │
│ (time series graph)  │   (stat counter)     │
└──────────────────────┴──────────────────────┘
```

**To explore traces from Grafana:**
1. **Explore** → select **Tempo** → query by service `image-styler`
2. Click any trace row to open the span waterfall

**To run custom metric queries from Grafana:**
1. **Explore** → select **Prometheus** → paste any PromQL query above

---

## Development

### Project Structure

- `app.py` - Streamlit UI and request orchestration with telemetry
- `image_generator.py` - Replicate API client with instrumentation
- `image_utils.py` - Image processing (EXIF, format conversion)
- `logger_config.py` - Centralized logging configuration
- `telemetry_config.py` - OpenTelemetry setup (traces & metrics)
- `METRICS_QUICK_START.md` - Quick reference for the 4 essential metrics
- `PROMETHEUS_QUERIES.md` - Complete Prometheus query documentation


---

## Requirements

See `pyproject.toml` for complete list. Key dependencies:

- **Application**: streamlit, replicate, pillow, python-dotenv
- **Observability**: opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp
- **Metrics**: prometheus-client

---

## License

[Add your license here]

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- AI models powered by [Replicate](https://replicate.com/)
- Image processing with PIL/Pillow
- Observability with [OpenTelemetry](https://opentelemetry.io/), [Grafana](https://grafana.com/), [Prometheus](https://prometheus.io/), and [Tempo](https://grafana.com/oss/tempo/)

---

## Quick Reference

### Common Commands

```bash
# Start everything
docker-compose up -d && uv run streamlit run app.py

# View logs
docker-compose logs -f
docker-compose logs -f otel-collector

# Restart a service
docker-compose restart grafana

# Check service status
docker-compose ps

# Stop everything
docker-compose down
```

### Useful URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Application | http://localhost:8502 | Streamlit app (check terminal for actual port) |
| Grafana Dashboard | http://localhost:3000/d/image-styler-overview | Pre-built metrics dashboard |
| Grafana Explore | http://localhost:3000/explore | Ad-hoc traces and metric queries |
| Prometheus | http://localhost:9090 | Raw metric queries |
| Prometheus Targets | http://localhost:9090/targets | Verify scrape targets are UP |
| OTEL Collector Metrics | http://localhost:8889/metrics | Raw metrics before Prometheus scrapes |

### Documentation Files

- **README.md** - Complete setup and usage guide (this file)
- **METRICS_QUICK_START.md** - Quick reference for the 4 essential metrics
- **PROMETHEUS_QUERIES.md** - Detailed Prometheus query documentation with examples
