# Intelehealth AI - Treatment Recommendation (TTx) Server

A FastAPI-based server that provides AI-powered treatment recommendations using DSPy and Google Gemini/Groq LLMs.

## Features

- 🤖 **Multiple LLM Support**: Gemini 2.5 Flash, Groq Llama 4 Maverick
- 🔄 **Configurable Transformation**: Choose different LLMs for response transformation
- 🎯 **Response Field Filtering**: Select specific fields to include in API responses
- 📊 **MLflow Integration**: Track experiments with SQLite (local) or MySQL (production)
- ⚡ **Async Support**: Fast, non-blocking API endpoints
- 🛡️ **Comprehensive Error Handling**: Proper HTTP status codes for all scenarios

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [MLflow Setup](#mlflow-setup)
- [API Usage](#api-usage)
- [Client Usage](#client-usage)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Error Handling](#error-handling)

## Prerequisites

- Python 3.10 or higher
- uv (Python package installer)
- API keys for:
  - Google Gemini API (`GEMINI_API_KEY`)
  - Groq API (`GROQ_API_KEY`)
- MySQL (optional, for production MLflow tracking)

## Installation

1. **Clone the repository**:
```bash
cd /path/to/intelehealth-ai-ttx
```

2. **Sync dependencies using uv**:
```bash
uv sync
```

This will install all required dependencies listed in `pyproject.toml`.

## Configuration

### Environment Variables

Create a `.env` file in the `ops/` directory:

```bash
# ops/.env

# Required API Keys
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Environment Configuration
ENVIRONMENT=local  # Options: local, production

# MLflow Configuration (Optional)
MLFLOW_TRACKING_URI=  # Leave empty for auto-configuration
ENABLE_MLFLOW_AUTOLOG=false  # Set to true to enable DSPy autologging

# MySQL Configuration (Production Only)
MYSQL_USER=mluser
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=mlflow_db
```

### MLflow Backend Selection

The server automatically selects the appropriate MLflow backend based on configuration:

| Configuration | Backend Used |
|--------------|-------------|
| `MLFLOW_TRACKING_URI` set | Uses specified URI |
| `ENVIRONMENT=production` + MySQL configured | Uses MySQL |
| `ENVIRONMENT=local` or MySQL not configured | Uses SQLite (`mlflow.db`) |

## Running the Server

### Start the TTx Server

```bash
# Using uv with auto-reload (recommended for development)
uv run uvicorn ttx_server:app --reload

# Or specify host and port
uv run uvicorn ttx_server:app --host 0.0.0.0 --port 8000 --reload

# Production (without reload)
uv run uvicorn ttx_server:app --host 0.0.0.0 --port 8000 --workers 4
```

The server will start at `http://127.0.0.1:8000` by default.

### Verify Server is Running

```bash
curl http://127.0.0.1:8000/health-status
```

Expected response:
```json
{
  "status": "AVAILABLE",
  "description": "Service status for TTx server"
}
```

## MLflow Setup

### Local Development (SQLite)

No additional setup required. MLflow will automatically create a `mlflow.db` file in the project root.

```bash
# View MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Access the UI at: `http://localhost:5000`

### Production (MySQL)

1. **Install MySQL** (if not already installed)

2. **Create MLflow database**:
```sql
CREATE DATABASE mlflow_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'mluser'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON mlflow_db.* TO 'mluser'@'localhost';
FLUSH PRIVILEGES;
```

3. **Configure environment variables**:
```bash
# In ops/.env
ENVIRONMENT=production
MYSQL_USER=mluser
MYSQL_PASSWORD=your_password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=mlflow_db
```

4. **Start MLflow UI**:
```bash
uv run mlflow ui --backend-store-uri "mysql+pymysql://mluser:your_password@localhost:3306/mlflow_db"
```

## API Usage

### Endpoint: `POST /ttx/v1`

Generate treatment recommendations for a patient case.

#### Request Body

```json
{
  "case": "Patient case description with symptoms, vitals, history...",
  "diagnosis": "Primary diagnosis",
  "model_name": "gemini-2.5-flash",
  "tracker": "unique_request_id",
  "transformation_model": "groq-llama",
  "response_fields": ["medications", "medical_advice"]
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `case` | string | Yes | - | Detailed patient case information |
| `diagnosis` | string | Yes | - | Primary diagnosis |
| `model_name` | string | Yes | - | LLM for TTx prediction (`gemini-2.5-flash`, `llama-4-maverick`) |
| `tracker` | string | Yes | - | Unique identifier for tracking |
| `transformation_model` | string | No | `groq-llama` | LLM for response transformation (`groq-llama`, `gemini-2.5-flash`) |
| `response_fields` | array | No | null (all fields) | List of fields to include in response |

#### Available Response Fields

- `medications` - List of prescribed medications
- `medical_advice` - General medical advice
- `tests_to_be_done` - Recommended diagnostic tests
- `follow_up` - Follow-up schedule and reasons
- `referral` - Referral information

**Note**: `success` and `error` fields are always included.

#### Example Request - All Fields

```bash
curl -X POST http://127.0.0.1:8000/ttx/v1 \
  -H "Content-Type: application/json" \
  -d '{
    "case": "Gender: Female, Age: 70 years, Chief_complaint: Cold, Sneezing, Headache...",
    "diagnosis": "Acute Pharyngitis",
    "model_name": "gemini-2.5-flash",
    "tracker": "test_001"
  }'
```

#### Example Request - Filtered Fields

```bash
curl -X POST http://127.0.0.1:8000/ttx/v1 \
  -H "Content-Type: application/json" \
  -d '{
    "case": "Gender: Female, Age: 70 years, Chief_complaint: Cold, Sneezing, Headache...",
    "diagnosis": "Acute Pharyngitis",
    "model_name": "gemini-2.5-flash",
    "tracker": "test_002",
    "response_fields": ["medications", "medical_advice"]
  }'
```

#### Response Format

```json
{
  "status": "success",
  "data": {
    "success": true,
    "medications": [
      {
        "name": "Paracetamol 500 mg Oral Tablet",
        "dosage": "1 tablet",
        "frequency": "Thrice daily",
        "duration": "3",
        "duration_unit": "days",
        "instructions": "After food",
        "confidence": "high"
      }
    ],
    "medical_advice": [
      "Rest and stay hydrated.",
      "Monitor temperature and symptoms."
    ],
    "tests_to_be_done": [
      {
        "test_name": "Complete Blood Count (CBC)",
        "test_reason": "To assess for signs of infection"
      }
    ],
    "follow_up": [
      {
        "follow_up_required": true,
        "follow_up_duration": "3 days",
        "reason_for_follow_up": "To assess symptom improvement"
      }
    ],
    "referral": [
      {
        "referral_required": false,
        "referral_to": "",
        "referral_facility": "",
        "remark": ""
      }
    ],
    "error": ""
  }
}
```

## Client Usage

### Running the Test Client

The project includes a Python client (`ttx_client.py`) for testing the API.

```bash
uv run python ttx_client.py
```

### Using the Client Library

```python
from ttx_client import get_treatment_recommendations_v1

# Get all fields
result = get_treatment_recommendations_v1(
    patient_case="Patient case description...",
    diagnosis="Acute Pharyngitis",
    model="gemini-2.5-flash"
)

# Get only specific fields
result = get_treatment_recommendations_v1(
    patient_case="Patient case description...",
    diagnosis="Acute Pharyngitis",
    model="gemini-2.5-flash",
    response_fields=["medications", "medical_advice"]
)

if result["success"]:
    print("Medications:", result["data"]["medications"])
    print("Advice:", result["data"]["medical_advice"])
else:
    print("Error:", result["error"])
```

### Custom API Integration

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/ttx/v1",
    json={
        "case": "Your patient case...",
        "diagnosis": "Diagnosis here",
        "model_name": "gemini-2.5-flash",
        "tracker": "unique_id",
        "transformation_model": "groq-llama",  # Optional
        "response_fields": ["medications", "medical_advice"]  # Optional
    }
)

if response.status_code == 200:
    data = response.json()
    print(data["data"]["medications"])
else:
    print(f"Error {response.status_code}: {response.text}")
```

## Environment Variables

### Complete Reference

```bash
# ============================================
# Required API Keys
# ============================================
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

# ============================================
# Environment Configuration
# ============================================
# Options: local, production
# Default: local
ENVIRONMENT=local

# ============================================
# MLflow Configuration
# ============================================

# Explicit tracking URI (overrides automatic selection)
# Examples:
#   - sqlite:///mlflow.db
#   - mysql+pymysql://user:pass@host:port/db
#   - http://mlflow-server:5000
MLFLOW_TRACKING_URI=

# Enable DSPy autologging (captures detailed traces)
# Options: true, false
# Default: false
ENABLE_MLFLOW_AUTOLOG=false

# ============================================
# MySQL Configuration (Production)
# ============================================
# Only used when ENVIRONMENT=production
MYSQL_USER=mluser
MYSQL_PASSWORD=your_secure_password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=mlflow_db
```

## API Reference

### Health Check

```
GET /health-status
```

**Response:**
```json
{
  "status": "AVAILABLE",
  "description": "Service status for TTx server"
}
```

### Treatment Recommendations

```
POST /ttx/v1
```

See [API Usage](#api-usage) section for detailed documentation.

## Error Handling

The API returns appropriate HTTP status codes:

| Status Code | Description | Example |
|------------|-------------|---------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid model name, invalid transformation model, invalid input format |
| 429 | Too Many Requests | API rate limit exceeded (Groq or Gemini) |
| 500 | Internal Server Error | Unexpected server error |
| 503 | Service Unavailable | API not configured, LLM service down, connection errors |

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

**API Key Not Configured (503)**:
```json
{
  "detail": "Groq transformation service not configured."
}
```

**Rate Limit Exceeded (429)**:
```json
{
  "detail": "API quota exceeded. Please try again later."
}
```

**Invalid Model Name (400)**:
```json
{
  "detail": "Invalid model_name: gemini-3.0. Supported models: gemini-2.5-flash, llama-4-maverick"
}
```

**Service Unavailable (503)**:
```json
{
  "detail": "Prediction service temporarily unavailable. Please try again later."
}
```

## Architecture

### Request Flow

```
Client Request
    ↓
FastAPI Endpoint (/ttx/v1)
    ↓
[1] Dynamic LLM Selection (based on model_name)
    ↓
[2] DSPy Program Execution (TTxv2Module or TTxv3Module)
    ↓
[3] Raw LLM Response
    ↓
[4] Response Transformation (Groq/Llama or Gemini)
    ↓
[5] Field Filtering (if response_fields specified)
    ↓
[6] MLflow Logging (metrics, params, artifacts)
    ↓
JSON Response to Client
```

### Supported Models

#### Main Prediction Models
- `gemini-2.5-flash` - Google Gemini 2.5 Flash (High quality, fast inference)
- `llama-4-maverick` - Groq Llama 4 Maverick 17B 128E (Extended context, medical reasoning)

#### Transformation Models
- `groq-llama` - Groq with Llama 4 Scout (default, fast structured output)
- `gemini-2.5-flash` - Google Gemini 2.5 Flash (alternative)

## Logging

Logs are written to:
- **Console**: Structured JSON logs
- **File**: `logs/ttx.log` (rotating, max 10MB per file, 5 backup files)

### Log Format

```json
{
  "timestamp": "2025-12-16T15:09:57.110472Z",
  "level": "INFO",
  "message": "TTx prediction successful. Total latency: 6.02 seconds",
  "logger_name": "ttx_logger",
  "tracker": "test",
  "model_name": "gemini-2.5-flash",
  "dspy_latency": 4.26,
  "transformation_latency": 1.39,
  "total_latency": 6.02
}
```

## Development

### Project Structure

```
intelehealth-ai-ttx/
├── modules/           # DSPy modules (TTxv2Module, TTxv3Module)
├── signatures/        # DSPy signatures
├── utils/            # Utility functions (metric_utils.py)
├── ops/              # Operations and configuration (.env)
├── outputs/          # Model weights and configurations
├── logs/             # Application logs
├── ttx_server.py     # Main FastAPI server
├── ttx_client.py     # Test client
├── pyproject.toml    # Project dependencies
└── README.md         # This file
```

### Running Tests

```bash
# Test with default configuration
uv run python ttx_client.py

# Test with custom case
# Edit ttx_client.py to modify patient_test_case and diagnosis
```

## Troubleshooting

### Server Won't Start

**Issue**: `ModuleNotFoundError` or dependency errors

**Solution**:
```bash
uv sync
```

### MLflow Database Issues

**Issue**: `alembic.runtime.migration` errors

**Solution**:
```bash
# Delete and recreate the database
rm mlflow.db
# Restart the server - it will recreate the database
```

### API Key Errors

**Issue**: `503: Service not configured`

**Solution**: Verify API keys are set in `ops/.env`:
```bash
echo $GEMINI_API_KEY
echo $GROQ_API_KEY
```

### Rate Limit Errors

**Issue**: `429: Too Many Requests`

**Solution**:
- Wait before retrying
- Consider using a different transformation model
- Check your API quota limits

## License

Copyright © 2025 Intelehealth. All rights reserved.

## Support

For issues and questions:
- Create an issue in the project repository
- Contact the development team

---

**Version**: 1.0.0
**Last Updated**: December 2025
