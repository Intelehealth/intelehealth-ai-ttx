# Artifacts Directory - TTx Server

This directory contains all model artifacts, configurations, and prompts for the TTx (Treatment Recommendation) server.

## Directory Structure

```
artifacts/
├── config/
│   └── model_registry.yaml      # Central registry for all models
├── models/
│   ├── 30_05_2025_13_44_ttx_v3_gemini2_5_cot_nas_v2_combined_medications_referral_tests_followup.json
│   └── 25_08_2025_16_45_ttx_v3_groq_llama_4_maverick_17b_128e_instruct_cot_nas_v2_combined_medications_referral_tests_followup.json
└── prompts/
    ├── ttx-base-system-prompt.txt
    └── transformation-prompt-template.txt
```

## Purpose

### config/
Contains configuration files that map model names to their artifacts:
- **model_registry.yaml**: Central registry defining available models, their weights files, LM loaders, and transformation models

### models/
Stores trained model weights (DSPy modules) for treatment recommendations:
- **Gemini 2.5 Flash weights**: TTxv3Module trained with Google Gemini 2.5 Flash
- **Llama 4 Maverick weights**: TTxv3Module trained with Groq Llama 4 Maverick 17B

### prompts/
Contains prompt templates used by the system:
- **ttx-base-system-prompt.txt**: Base system prompt for treatment recommendations
- **transformation-prompt-template.txt**: Template for transforming LLM outputs to structured JSON

## Supported Models

### Main Prediction Models

#### gemini-2.5-flash
- **Provider**: Google
- **Module**: TTxv3Module
- **Weights**: `artifacts/models/30_05_2025_13_44_ttx_v3_gemini2_5_cot_nas_v2_combined_medications_referral_tests_followup.json`
- **LM Loader**: `load_gemini2_5_lm()`
- **Max Tokens**: 10,000
- **Description**: High-quality, fast inference for treatment recommendations

#### llama-4-maverick
- **Provider**: Groq
- **Model**: meta-llama/llama-4-maverick-17b-128e-instruct
- **Module**: TTxv3Module
- **Weights**: `artifacts/models/25_08_2025_16_45_ttx_v3_groq_llama_4_maverick_17b_128e_instruct_cot_nas_v2_combined_medications_referral_tests_followup.json`
- **LM Loader**: `load_groq_llama_4_maverick()`
- **Max Tokens**: 8,192
- **Description**: Extended context window, medical reasoning specialist

### Transformation Models

#### groq-llama (default)
- **Model**: meta-llama/llama-4-scout-17b-16e-instruct
- **Provider**: Groq
- **LM Loader**: `load_groq_llama_scout()`
- **Purpose**: Fast structured JSON output transformation
- **Max Tokens**: 8,192

#### gemini-2.5-flash
- **Provider**: Google
- **LM Loader**: `load_gemini2_5_lm()`
- **Purpose**: Alternative transformation model

## Usage

### Loading Models in Code

```python
from utils.ttx_utils import load_gemini2_5_lm, load_groq_llama_4_maverick
from modules.TTxv3Module import TTxv3Module

# Load Gemini 2.5 Flash
load_gemini2_5_lm()
cot = TTxv3Module()
cot.load("artifacts/models/30_05_2025_13_44_ttx_v3_gemini2_5_cot_nas_v2_combined_medications_referral_tests_followup.json")

# Load Llama 4 Maverick
load_groq_llama_4_maverick()
cot = TTxv3Module()
cot.load("artifacts/models/25_08_2025_16_45_ttx_v3_groq_llama_4_maverick_17b_128e_instruct_cot_nas_v2_combined_medications_referral_tests_followup.json")
```

### API Request

```bash
curl -X POST http://127.0.0.1:8000/ttx/v1 \
  -H "Content-Type: application/json" \
  -d '{
    "case": "Patient case description...",
    "diagnosis": "Acute Pharyngitis",
    "model_name": "llama-4-maverick",
    "tracker": "unique_id",
    "transformation_model": "groq-llama"
  }'
```

## Adding New Models

To add a new model:

1. **Add model weights**: Place the trained model JSON file in `artifacts/models/`
2. **Create LM loader**: Add a new loader function in `utils/ttx_utils.py`
3. **Update registry**: Add model configuration to `artifacts/config/model_registry.yaml`
4. **Update server**: Add model case in `ttx_server.py` endpoint

## Best Practices

1. **Naming Convention**: Keep descriptive filenames with date, version, model name
2. **Version Control**: Do not commit large model files to git (use .gitignore)
3. **Documentation**: Update this README when adding new models
4. **Testing**: Always test new models before deploying to production

## Technical Notes

### DSPy Configuration
- **Gemini**: Uses `dspy.Google` with `max_output_tokens=10000`
- **Groq**: Uses `dspy.LM` with `max_tokens=8192` (Groq API limit)

### Model Format
All model weight files are JSON-serialized DSPy modules that can be loaded with:
```python
module.load("path/to/weights.json")
```

---

**Last Updated**: December 2025
**Version**: 1.0.0
