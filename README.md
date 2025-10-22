# Image Styler

A Streamlit web application that allows you to upload images and apply various artistic styles to transform them using AI models from Replicate. Features comprehensive observability with OpenTelemetry, Prometheus, Tempo, and Grafana.

## Features

- 📸 **Image Upload**: Support for common image formats (JPG, PNG, JPEG)
- 🔄 **EXIF Orientation**: Automatically fixes image orientation
- 🎨 **AI Style Transfer**: Apply artistic styles using FLUX models from Replicate
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
        └── datasources.yml
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
- **Grafana Dashboard**: [http://localhost:3000/d/image-styler-overview](http://localhost:3000/d/image-styler-overview)
- **Grafana**: [http://localhost:3000](http://localhost:3000) (auto-login enabled)
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

## Observability Details

### Traces (Tempo)

The application creates the following trace spans for each request:

```
image_styling_request (root)
├── fix_orientation
└── replicate_api_call
```

**Span Attributes**:
- `filename`, `file_size_mb`, `file_type`
- `model`, `prompt_length`
- `orientation_fixed`, `success`
- `duration_seconds`, `error_message` (if failed)

### Metrics (Prometheus)

**The 4 Essential Metrics** (displayed in Grafana dashboard):

1. **API Success Rate**
   - Query: `sum(rate(image_styler_api_calls_total{status="success"}[5m])) / sum(rate(image_styler_api_calls_total[5m])) * 100`
   - Shows: Percentage of successful API calls
   - Target: > 95%

2. **Total API Calls**
   - Query: `sum(image_styler_api_calls_total)`
   - Shows: Total number of API calls since app started
   - Use: Track overall usage

3. **Average API Call Duration**
   - Query: `rate(image_styler_api_call_duration_seconds_sum[5m]) / rate(image_styler_api_call_duration_seconds_count[5m])`
   - Shows: Average time for API calls in seconds
   - Target: < 15 seconds

4. **Total Errors**
   - Query: `sum(image_styler_api_errors_total)`
   - Shows: Total number of errors since app started
   - Target: 0

**Additional Metrics Available**:
- `images_processed_total` - Total images processed by model and status
- `api_calls_total` - Total API calls by model and status
- `api_errors_total` - Total API errors by error type
- `api_call_duration_seconds` - API response time histogram
- `request_duration_seconds` - End-to-end request duration histogram

**For More Queries**: See `PROMETHEUS_QUERIES.md` and `METRICS_QUICK_START.md` for detailed examples and usage.

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

- **Application**: http://localhost:8502 (check terminal for actual port)
- **Grafana Dashboard**: http://localhost:3000/d/image-styler-overview
- **Grafana Home**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Prometheus Targets**: http://localhost:9090/targets
- **OTEL Collector Metrics**: http://localhost:8889/metrics

### Documentation Files

- **README.md** - Complete setup and usage guide (this file)
- **METRICS_QUICK_START.md** - Quick reference for the 4 essential metrics
- **PROMETHEUS_QUERIES.md** - Detailed Prometheus query documentation with examples
