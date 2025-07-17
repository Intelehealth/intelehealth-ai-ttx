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


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

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

        log_dir = 'logs'
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
logger.info("Setting up MLflow for DSPy tracking...")
# Set tracking URI to MySQL database backend
# URL-encode the password to handle special characters like @
# mlflow.set_tracking_uri("mysql+pymysql://mluser:noidea#2@0.0.0.0:3306/mlflow_db")
# mlflow.set_experiment("ttx-server-tracking")
# mlflow.dspy.autolog(
#     log_traces=True,
#     log_traces_from_compile=True,
#     log_traces_from_eval=True,
#     log_compiles=True,
#     log_evals=True,
#     silent=False
#)
logger.info("MLflow DSPy autologging configured successfully!")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


async def transform_ttx_output(llm_output: dict) -> dict:
    """Transform the TTx LLM output into a structured format using an LLM.
    
    Args:
        llm_output (dspy.Prediction): The raw output from the TTx model
        
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
    if not llm_output:
        logger.error("TTx output is None, cannot transform.")
        return {
            "success": False,
            "medications": [],
            "medical_advice": [],
            "tests_to_be_done": [],
            "follow_up": [],
            "referral": [],
            "error": "Error processing treatment recommendations: Empty output from model."
        }
        
    # Construct a clean, readable string from the llm_output object
    input_text = f"""
    Rationale: {llm_output.get('rationale', '')}
    Medication Recommendations: {llm_output.get('medication_recommendations', '')}
    Medical Advice: {llm_output.get('medical_advice', '')}
    Tests to be done: {llm_output.get('tests_to_be_done', '')}
    Follow up: {llm_output.get('follow_up', '')}
    Referral: {llm_output.get('referral', '')}
    """
    
    transform_prompt = f"""Transform the following input text into a single, valid JSON object.

**Transform the following input text:**
```
{input_text}
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

    if not groq_client:
        logger.error("GROQ_API_KEY not found. Cannot use Groq for transformation.")
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    try:
        print(transform_prompt)
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

        print(" transformation response ")
        print(response_text)
        
        # Log the transformation request to MLflow (if enabled)
        # if not MLFLOW_DISABLED:
        #     mlflow.log_param("transformation_model", "meta-llama/llama-4-scout-17b-16e-instruct")
        #     mlflow.log_param("transformation_prompt_length", len(transform_prompt))
            
        #     # Log the transformation prompt as an artifact
        #     mlflow.log_text(transform_prompt, "transformation_prompt.txt")
        #     mlflow.log_text(str(response_text), "raw_transformation_response.txt")
            
        #     # Log transformation success metric
        #     mlflow.log_metric("transformation_success", 1)
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

async def transform_ttx_output_old(llm_output: dict) -> dict:
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
    transform_prompt = f"""You are a service that transforms raw text into a single, valid JSON object.
Your entire response MUST be only the JSON object, starting with `{{` and ending with `}}`.
Do not include any other text, explanations, or markdown formatting like ````json`.

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
        response = genai.models.generate_content(
            model="gemini-2.5-flash",
            contents=transform_prompt,
        )
        # Log the transformation request to MLflow
        mlflow.log_param("transformation_model", "gemini-2.5-flash-preview-04-17")
        mlflow.log_param("transformation_prompt_length", len(transform_prompt))
        
        # Log the transformation prompt as an artifact
        mlflow.log_text(transform_prompt, "transformation_prompt.txt")
        mlflow.log_text(str(response), "raw_transformation_response.txt")
        
        # Log transformation success metric
        mlflow.log_metric("transformation_success", 1)
        logger.info("Transformation response: -----")
        logger.info(response)
        logger.info("-----")
        
        # Extract JSON from the markdown-formatted response
        response_text = response.text
        json_str = None
        
        # Use regex to find JSON between ```json and ``` or just a plain JSON object
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', response_text)
        
        if json_match:
            # If the first group (markdown) is found, use it; otherwise, use the second group (plain JSON)
            if json_match.group(1):
                json_str = json_match.group(1).strip()
            elif json_match.group(2):
                json_str = json_match.group(2).strip()

        if not json_str:
            raise ValueError("No JSON object found in the LLM response.")
            
        # Parse the extracted JSON
        transformed = json.loads(json_str)
        
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

    # Start MLflow run to track this prediction
    with mlflow.start_run(run_name=f"ttx_prediction_{int(time.time())}"):
        # Log request parameters
        mlflow.log_param("model_name", request_body.model_name)
        mlflow.log_param("case_length", len(request_body.case))
        mlflow.log_param("diagnosis", request_body.diagnosis)
        mlflow.log_param("tracker", request_body.tracker)

        # Log case and diagnosis
        mlflow.log_text(request_body.case, "input_case.txt")
        mlflow.log_text(request_body.diagnosis, "input_diagnosis.txt")

        cot = None
        if request_body.model_name == "gemini-2.0-flash":
            cot = TTxv2Module()
            cot.load("outputs/" + "24_04_2025_12_11_ttx_v2_gemini_cot_nas_v2_combined_llm_judge.json")
        elif request_body.model_name == "gemini-2.5-flash":
            cot = TTxv3Module()
            cot.load("outputs/" + "30_05_2025_13_44_ttx_v3_gemini2_5_cot_nas_v2_combined_medications_referral_tests_followup.json")
        else:
            raise HTTPException(status_code=400, detail="Invalid model name for TTx v2")

        dspy_program = dspy.asyncify(cot)

        try:
            logger.info("About to call DSPy program for TTx...", extra=log_extra)

            dspy_start_time = time.time()
            result = await dspy_program(case=request_body.case, diagnosis=request_body.diagnosis)
            dspy_latency = time.time() - dspy_start_time
            
            log_extra['extra_data']['dspy_latency'] = dspy_latency
            logger.info(f"DSPy program for TTx completed in {dspy_latency:.2f} seconds", extra=log_extra)
            mlflow.log_metric("dspy_latency", dspy_latency)
            
            # Log the raw result
            mlflow.log_text(str(result), "raw_dspy_output.txt")

            if hasattr(result, 'output') and hasattr(result.output, 'treatment') and result.output.treatment == "NA":
                logger.warning("No treatment possible for this case.", extra=log_extra)
                response_data = {
                    "status": "success",
                    "data": "The Input provided does not have enough clinical details for AI based assessment."
                }
                mlflow.log_text(str(response_data), "final_response.txt")
                return response_data
            
            transform_start_time = time.time()
            print(result.toDict())
            transformed_output = await transform_ttx_output(result.output.toDict())
            transform_latency = time.time() - transform_start_time
            log_extra['extra_data']['transformation_latency'] = transform_latency
            logger.info(f"TTx transformation completed in {transform_latency:.2f} seconds", extra=log_extra)
            mlflow.log_metric("transformation_latency", transform_latency)
            
            # Log the transformed output
            mlflow.log_text(json.dumps(transformed_output, indent=2), "transformed_output.json")
            
            response_data = {
                "status": "success",
                "data": transformed_output
            }
            
            # Log final response
            mlflow.log_text(json.dumps(response_data, indent=2), "final_response.json")
            mlflow.log_metric("prediction_success", 1)
            
            total_latency = time.time() - start_time
            mlflow.log_metric("total_latency", total_latency)
            log_extra['extra_data']['total_latency'] = total_latency
            logger.info(f"TTx prediction successful. Total latency: {total_latency:.2f} seconds", extra=log_extra)
            
            return response_data

        except Exception as e:
            logger.error(f"Error in TTx prediction: {e}", extra=log_extra)
            mlflow.log_param("error", str(e))
            mlflow.log_metric("prediction_success", 0)
            logger.error(
                f"Error processing TTx request: {e}",
                extra_data={"tracker": request_body.tracker, "error": str(e)}
            )
            raise HTTPException(status_code=500, detail="Internal Server Error. Please try again later.")


@app.get("/health-status")
async def health_status():
    return {
        "status": "AVAILABLE",
        "description": "Service status for TTx server"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 