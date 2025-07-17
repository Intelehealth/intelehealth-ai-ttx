import dspy
import time
import json
from utils.metric_utils import load_gemini_lm_prod, load_open_ai_lm, load_gemini_lm, load_gemini2_lm, load_gemini2_5_lm
from dotenv import load_dotenv
from modules.TTxModule import TTxModule
from modules.TTxv2Module import TTxv2Module
from modules.TTxv3Module import TTxv3Module
import google.generativeai as genai
from groq import AsyncGroq
import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prompt_config import prompt_config
from ttx_client import process_medications
import mlflow
import re
from fastapi.responses import JSONResponse

# Custom JSON Formatter for structured logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
        }
        if hasattr(record, 'extra_data'):
            log_record.update(record.extra_data)
        return json.dumps(log_record)

class CustomLogger:
    def __init__(self, name='ttx_logger', log_file='ttx.log', max_bytes=10485760, backup_count=5):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Get log directory from environment variable, default to 'logs' for local development
        log_dir = os.getenv('LOG_DIR', 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, log_file)

        file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        console_handler = logging.StreamHandler()

        formatter = JSONFormatter()
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _log(self, level, message, *args, **kwargs):
        extra_data = kwargs.pop('extra_data', None)
        if extra_data:
            kwargs['extra'] = {'extra_data': extra_data}
        self.logger.log(level, message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self._log(logging.INFO, message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._log(logging.ERROR, message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._log(logging.WARNING, message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self._log(logging.DEBUG, message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self._log(logging.CRITICAL, message, *args, **kwargs)

logger = CustomLogger()

load_dotenv(
    "ops/.env"
)

# MLflow setup for tracking DSPy calls
MLFLOW_DISABLED = os.getenv("MLFLOW_DISABLED", "false").lower() == "true"

if not MLFLOW_DISABLED:
    logger.info("Setting up MLflow for DSPy tracking...")
    
    # Control MLflow backend via environment variable
    USE_LOCAL_MLFLOW = os.getenv("USE_LOCAL_MLFLOW", "false").lower() == "true"
    
    if USE_LOCAL_MLFLOW:
        # Use local SQLite database for development
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        logger.info("Using local SQLite database for MLflow tracking")
    else:
        # Use MySQL database backend for production
        # Get MySQL credentials from environment variables
        mysql_user = os.getenv("MLFLOW_MYSQL_USER")
        mysql_password = os.getenv("MLFLOW_MYSQL_PASSWORD")
        mysql_host = os.getenv("MLFLOW_MYSQL_HOST", "localhost")
        mysql_port = os.getenv("MLFLOW_MYSQL_PORT", "3306")
        mysql_db = os.getenv("MLFLOW_MYSQL_DB", "mlflow_db")
        
        # Validate required credentials are present
        if not mysql_user or not mysql_password:
            logger.error("MySQL credentials not found in environment variables. Please set MLFLOW_MYSQL_USER and MLFLOW_MYSQL_PASSWORD in ops/.env")
            raise ValueError("Missing required MySQL credentials for MLflow tracking")
        
        # URL-encode the password to handle special characters like @
        mysql_uri = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
        mlflow.set_tracking_uri(mysql_uri)
        logger.info("Using MySQL database for MLflow tracking")
    
    mlflow.set_experiment("ttx-server-tracking")
    mlflow.dspy.autolog(
        log_traces=True,
        log_traces_from_compile=True,
        log_traces_from_eval=True,
        log_compiles=True,
        log_evals=True,
        silent=False
    )
    logger.info("MLflow DSPy autologging configured successfully!")
else:
    logger.info("MLflow tracking is disabled via MLFLOW_DISABLED environment variable")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = None

# Initialize Groq client only if API key is provided
if GROQ_API_KEY:
    try:
        groq_client = AsyncGroq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        groq_client = None
else:
    logger.warning("GROQ_API_KEY not found in environment variables. Groq client will not be available.")

async def transform_ttx_output(llm_output: dict) -> dict:
    """Transform the TTx LLM output into a structured format using an LLM.
    
    Args:
        llm_output (dict): The raw output from the TTx model
        
    Returns:
        dict: A structured response with the following format:
        {
            "success": bool,
            "medications": [
                {
                    "name": "medication in route-form-strength format",
                    "dosage": "dosage information",
                    "frequency": "frequency of administration",
                    "duration": "duration of treatment",
                    "instructions": "instructions if any for this medication",
                    "confidence": "confidence level of the medication"
                },
                ...
            ],
            "medical_advice": [
                "Rest and stay hydrated. Drink plenty of warm fluids.",
                "Monitor temperature and symptoms.",
                "If symptoms worsen or new symptoms develop, seek further medical advice.",
                ...
            ],
            "tests_to_be_done": [{
                "test_name": "test_name",
                "test_reason": "test_reason"
                },
                ...
            ],
            "follow_up": [{
                "follow_up_required": "true/false",
                "follow_up_duration": "number along with the unit of time like days/weeks/month",
                "reason_for_follow_up": "short rationale about the follow up"
                },
                ...
            ],
            "referral": [{
                "referral_required": "true/false",
                "referral_to": "referral_to",
                "referral_facility": "referral_facility",
                "remark": "remark"
                },
                ...
            ],
            "error": "error message if success is false"
        }
    """
    transform_prompt = f"""Transform the following input text into a single, valid JSON object.

**Transform the following input text:**
```
{str(llm_output)}
```

**Into this exact JSON structure:**
```json
{{
    "success": true,
    "medications": [
        {{
            "name": "medication in route-form-strength format",
            "dosage": "dosage information",
            "frequency": "frequency of administration",
            "duration": "duration of treatment (number, or empty string if not applicable/null)",
            "duration_unit": "unit of time like days/weeks/month (singular or plural), or empty string if duration is empty or unit is not applicable/valid",
            "instructions": "instructions if any for this medication",
            "confidence": "confidence level of the medication"
        }}
    ],
    "medical_advice": [
         "Rest and stay hydrated. Drink plenty of warm fluids.",
         "Monitor temperature and symptoms."
    ],
    "tests_to_be_done": [
        {{
            "test_name": "test_name",
            "test_reason": "test_reason"
        }}
    ],
    "follow_up": [
        {{
            "follow_up_required": "true/false",
            "follow_up_duration": "number along with the unit of time like days/weeks/month",
            "reason_for_follow_up": "short rationale about the follow up"
        }}
    ],
    "referral": [
        {{
            "referral_required": "true/false",
            "referral_to": "referral_to",
            "referral_facility": "referral_facility",
            "remark": "remark"
        }}
    ],
    "error": "error message if success is false"
}}
```

**Guidelines for Transformation:**
1.  `success`: Set to `true` if valid medication recommendations are found, otherwise `false`.
2.  `medications`:
    - `name`: Format as "MedicationName Strength Route Form" (e.g., "Paracetamol 500 mg Oral Tablet").
    - `dosage`: The recommended dosage (e.g., "1 tablet", "10 ml").
    - `frequency`: Use standard terms (e.g., "Once daily", "Twice daily").
    - `duration`: A number as a string. Use an empty string `""` if not applicable.
    - `duration_unit`: Use "days", "weeks", "months". Use an empty string `""` if not applicable.
    - `instructions`: Include timing and food instructions (e.g., "After food", "At bedtime").
3.  `medical_advice`: **CRITICAL**: This MUST be an array of strings. If the input for it is empty, return an empty array `[]`.
4.  `follow_up`:
    - Parse the input string to extract `follow_up_required` (string "true" or "false"), `next_followup_duration` (number), `next_followup_units` (string), and `next_followup_reason`.
    - Populate the output fields accordingly. `follow_up_required` in the output should be a boolean.
5.  `referral`:
    - `referral_required` in the output should be a boolean.
6.  If `tests_to_be_done`, `follow_up`, or `referral` are `None` or empty in the input, return an empty array `[]` for the corresponding field in the output JSON.
7.  If the input indicates insufficient information, set `success` to `false` and provide an error message in the `error` field for insufficient information.
"""

    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that transforms medical LLM output into a structured JSON format. Your output must be only the JSON object, without any markdown formatting like ```json or any other explanatory text."
                },
                {
                    "role": "user",
                    "content": transform_prompt,
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        response_text = chat_completion.choices[0].message.content
        
        # Log the transformation request to MLflow (if enabled)
        if not MLFLOW_DISABLED:
            mlflow.log_param("transformation_model", "meta-llama/llama-4-scout-17b-16e-instruct")
            mlflow.log_param("transformation_prompt_length", len(transform_prompt))
            
            # Log the transformation prompt as an artifact
            mlflow.log_text(transform_prompt, "transformation_prompt.txt")
            mlflow.log_text(str(response_text), "raw_transformation_response.txt")
            
            # Log transformation success metric
            mlflow.log_metric("transformation_success", 1)
        logger.info("Transformation response: -----")
        logger.info(response_text)
        logger.info("-----")
        
        # Parse the JSON response directly since Groq returns JSON format
        transformed = json.loads(response_text)
        
        # Ensure medical_advice and adverse_effects are always arrays
        if "medical_advice" not in transformed or not isinstance(transformed["medical_advice"], list):
            transformed["medical_advice"] = []
        # if "adverse_effects" not in transformed or not isinstance(transformed["adverse_effects"], list):
        #     transformed["adverse_effects"] = []
            
        # Handle empty or null medical_advice
        if transformed["medical_advice"] is None or transformed["medical_advice"] == "":
            transformed["medical_advice"] = []
        
        # Ensure medical_advice maintains the correct format with numbered keys
        # The format should be an array containing a single object with numbered keys
            
        return transformed
    except Exception as e:
        logger.error(f"Error transforming TTx output: {e}")
        # Fallback to a basic error response if transformation fails
        return {
            "success": False,
            "medications": [],
            "medical_advice": [],
            "tests_to_be_done": [],
            "follow_up": [],
            "referral": [],
            "error": "Error processing treatment recommendations. Please try again."
        }

load_gemini2_lm()

app = FastAPI(
    title="Treatment Recommendation Server",
    description="A simple API serving a DSPy Chain of Thought program for TTx",
    version="1.0.0"
)

class BaseTTxRequest(BaseModel):
    case: str
    diagnosis: str
    model_name: str
    tracker: str


@app.post("/ttx/v1")
async def ttx_v1(request_body: BaseTTxRequest):
    start_time = time.time()
    log_extra = {
        'extra_data': {
            'tracker': request_body.tracker,
            'model_name': request_body.model_name,
            'diagnosis': request_body.diagnosis
        }
    }
    logger.info("Starting TTx prediction request...", extra_data=log_extra)

    mlflow_context = None
    if not MLFLOW_DISABLED:
        try:
            mlflow_context = mlflow.start_run(run_name=f"ttx_prediction_{int(time.time())}")
            mlflow_context.__enter__()
            mlflow.log_param("model_name", request_body.model_name)
            mlflow.log_param("case_length", len(request_body.case))
            mlflow.log_param("diagnosis", request_body.diagnosis)
            mlflow.log_param("tracker", request_body.tracker)
            mlflow.log_text(request_body.case, "input_case.txt")
            mlflow.log_text(request_body.diagnosis, "input_diagnosis.txt")
            logger.info("MLflow run started successfully.", extra_data=log_extra)
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}", extra_data=log_extra)
            # Decide if you want to raise an exception or just log the error
            # For now, we'll just log it and continue without MLflow for this request
            mlflow_context = None

    try:
        logger.info("Loading TTx module...", extra_data=log_extra)
        cot = None
        if request_body.model_name == "gemini-2.0-flash":
            cot = TTxv2Module()
            cot.load("outputs/" + "24_04_2025_12_11_ttx_v2_gemini_cot_nas_v2_combined_llm_judge.json")
        elif request_body.model_name == "gemini-2.5-flash":
            cot = TTxv3Module()
            cot.load("outputs/" + "30_05_2025_13_44_ttx_v3_gemini2_5_cot_nas_v2_combined_medications_referral_tests_followup.json")
        else:
            logger.error(f"Invalid model name: {request_body.model_name}", extra_data=log_extra)
            raise HTTPException(status_code=400, detail=f"Invalid model name for TTx v1: {request_body.model_name}")
        logger.info("TTx module loaded successfully.", extra_data=log_extra)

        dspy_program = dspy.asyncify(cot)
        logger.info("About to call DSPy program for TTx...", extra_data=log_extra)
        
        dspy_start_time = time.time()
        result = await dspy_program(case=request_body.case, diagnosis=request_body.diagnosis)
        dspy_latency = time.time() - dspy_start_time
        
        log_extra['extra_data']['dspy_latency'] = dspy_latency
        logger.info(f"DSPy program for TTx completed in {dspy_latency:.2f} seconds", extra_data=log_extra)
        
        if mlflow_context:
            mlflow.log_metric("dspy_latency", dspy_latency)
            mlflow.log_text(str(result), "raw_dspy_output.txt")

        if hasattr(result, 'output') and hasattr(result.output, 'treatment') and result.output.treatment == "NA":
            logger.warning("No treatment possible for this case.", extra_data=log_extra)
            response_data = {
                "status": "success",
                "data": "The Input provided does not have enough clinical details for AI based assessment."
            }
            if mlflow_context:
                mlflow.log_text(str(response_data), "final_response.txt")
            return JSONResponse(content=response_data)

        logger.info("Transforming TTx output...", extra_data=log_extra)
        transform_start_time = time.time()
        transformed_output = await transform_ttx_output(result.toDict())
        transform_latency = time.time() - transform_start_time
        
        log_extra['extra_data']['transformation_latency'] = transform_latency
        logger.info(f"TTx transformation completed in {transform_latency:.2f} seconds", extra_data=log_extra)
        
        if mlflow_context:
            mlflow.log_metric("transformation_latency", transform_latency)
            mlflow.log_text(json.dumps(transformed_output, indent=2), "transformed_output.json")

        response_data = {
            "status": "success",
            "data": transformed_output
        }
        
        if mlflow_context:
            mlflow.log_text(json.dumps(response_data, indent=2), "final_response.json")
            mlflow.log_metric("prediction_success", 1)

        total_latency = time.time() - start_time
        if mlflow_context:
            mlflow.log_metric("total_latency", total_latency)
            
        log_extra['extra_data']['total_latency'] = total_latency
        logger.info(f"TTx prediction successful. Total latency: {total_latency:.2f} seconds", extra_data=log_extra)
        
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error in TTx prediction: {e}", exc_info=True, extra_data=log_extra)
        if mlflow_context:
            mlflow.log_param("error", str(e))
            mlflow.log_metric("prediction_success", 0)
        raise HTTPException(status_code=500, detail="Internal Server Error. Please try again later.")

    finally:
        if mlflow_context:
            mlflow_context.__exit__(None, None, None)
            logger.info("MLflow run finished.", extra_data=log_extra)


@app.get("/health-status")
async def health_status():
    return {
        "status": "AVAILABLE",
        "description": "Service status for TTx server"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
