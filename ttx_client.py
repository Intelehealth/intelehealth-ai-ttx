import requests
import json
import re

patient_test_case = """
Gender: Female

 Age: 70 years

 Chief_complaint: ► **Cold, Sneezing** :  
• 1 Days.  
• Precipitating factors - Cold weather, Wind.  
• Prior treatment sought - None.  
► **Headache** :  
• Duration - 1 Days.  
• Site - Localized - कपाळावरती डोकं दुखतंय .  
• Severity - Moderate.  
• Onset - Acute onset (Patient can recall exact time when it started).  
• Character of headache - Throbbing.  
• Radiation - pain does not radiate.  
• Timing - Day.  
• Exacerbating factors - bending.  
• Prior treatment sought - None.  
► **Leg, Knee or Hip Pain** :  
• Site - Right leg, Hip, Thigh, Knee, Site of knee pain - Front, Back,
Lateral/medial. Swelling - No, Calf, Left leg, Hip, Thigh, Knee, Site of knee
pain - Front, Back, Lateral/medial. Swelling - No, Calf, Hip.  
• Duration - 6 Days.  
• Pain characteristics - Sharp shooting.  
• Onset - Gradual.  
• Progress - Static (Not changed).  
• Pain does not radiate.  
• Aggravating

 Physical_examination: **General exams:**  
• Eyes: Jaundice-no jaundice seen, [picture taken].  
• Eyes: Pallor-normal pallor, [picture taken].  
• Arm-Pinch skin* - pinch test normal.  
• Nail abnormality-nails normal, [picture taken].  
• Nail anemia-Nails are not pale, [picture taken].  
• Ankle-no pedal oedema, [picture taken].  
**Joint:**  
• non-tender.  
• no deformity around joint.  
• full range of movement is seen.  
• joint is not swollen.  
• no pain during movement.  
• no redness around joint.  
**Back:**  
• tenderness observed.  
**Head:**  
• No injury.

 Patient_medical_history: • Pregnancy status - Not pregnant.  
• Allergies - No known allergies.  
• Alcohol use - No.  
• Smoking history - Patient denied/has no h/o smoking.  
• Medical History - None.  
• Drug history - No recent medication.  

 Family_history: •Do you have a family history of any of the following? : None.  

 Vitals:- 

Sbp: 140.0

 Dbp: 90.0

 Pulse: 83.0

 Temperature: 36.78 'C

 Weight: 44.75 Kg

 Height: 152.0 cm

 BMI: 19.37

 RR: 21.0

 SPO2: 97.0

 HB: Null

 Sugar_random: Null

 Blood_group: Null

 Sugar_pp: Null

 Sugar_after_meal: Null
"""

diagnosis = "Acute Pharyngitis"

def process_medications(medication_text):
    """
    Process the medication recommendations text into a structured array
    
    Args:
        medication_text (str): Text containing medication recommendations
        
    Returns:
        list: Array of structured medication objects with the following fields:
            - drug_name: Name of the medication
            - strength: Dosage strength (e.g., "500 mg")
            - route: Administration route (e.g., "oral")
            - form: Medication form (e.g., "tablet")
            - dose: Amount to take (e.g., "1 tablet")
            - frequency: How often to take (e.g., "Thrice daily (TID)")
            - duration_number: Number of days/weeks/months
            - duration_unit: Unit of duration (e.g., "Days", "Weeks")
            - instruction: Additional instructions (e.g., "After food")
            - confidence: High/Moderate/Low confidence level
            - rationale: Reason for the medication
    """
    medications = []
    
    # Split text into individual medication entries (each starting with a number)
    med_entries = []
    current_entry = ""
    
    print("PROCESSING V2 medications")
    # Split by numbered entries
    for line in medication_text.strip().split(". "):
        if line.strip() and line.strip()[0].isdigit():
            if current_entry:
                med_entries.append(current_entry)
            current_entry = line
        elif current_entry:
            current_entry += ". " + line
    
    if current_entry:
        med_entries.append(current_entry)
    
    # Process each entry
    for entry in med_entries:
        med = {}
        
        # Extract drug name - everything before the first number
        name_match = re.match(r'^\d+\.\s*([^(]+)', entry)
        if name_match:
            med["drug_name"] = name_match.group(1).strip()
        else:
            continue  # Skip if we can't find a name
        
        # Extract strength
        strength_match = re.search(r'(\d+(?:\.\d+)?\s*(?:mg|g|ml|IU|%))', entry)
        med["strength"] = strength_match.group(1) if strength_match else ""
        
        # Extract route and form
        route = []
        form = []
        
        if "Oral" in entry:
            route.append("oral")
        if "Nasal" in entry:
            route.append("nasal")
        if "Skin" in entry:
            route.append("topical")
            
        if "Tablet" in entry:
            form.append("tablet")
        if "Capsule" in entry:
            form.append("capsule")
        if "Drops" in entry:
            form.append("drops")
        if "Lotion" in entry:
            form.append("lotion")
            
        med["route"] = ", ".join(route) if route else ""
        med["form"] = ", ".join(form) if form else ""
        
        # Extract dose
        dose_patterns = [
            r'(\d+\s+tablet)',
            r'(\d+\s+drops)',
            r'(\d+\s+capsule)',
            r'(Sufficient\s+Quantity)'
        ]
        
        for pattern in dose_patterns:
            dose_match = re.search(pattern, entry)
            if dose_match:
                med["dose"] = dose_match.group(1)
                break
        else:
            med["dose"] = ""
        
        # Extract frequency
        freq_match = re.search(r'((?:Thrice|Twice|Once)\s+daily\s*(?:\([^)]+\))?)', entry)
        med["frequency"] = freq_match.group(1) if freq_match else ""
        
        # Extract duration
        duration_match = re.search(r'(\d+)\s+(Days|Day|Week|Weeks|Month|Months)', entry)
        if duration_match:
            med["duration_number"] = duration_match.group(1)
            med["duration_unit"] = duration_match.group(2)
        else:
            med["duration_number"] = ""
            med["duration_unit"] = ""
        
        # Extract instruction
        instruction_match = re.search(r'(After\s+food|At\s+bedtime|Apply\s+to\s+the\s+affected\s+area|Nostrils)', entry)
        med["instruction"] = instruction_match.group(1) if instruction_match else ""
        
        # Extract confidence
        if "Likelihood: High" in entry:
            med["confidence"] = "High"
        elif "Likelihood: Moderate" in entry:
            med["confidence"] = "Moderate"
        elif "Likelihood: Low" in entry:
            med["confidence"] = "Low"
        else:
            med["confidence"] = "Unknown"
        
        # Extract rationale
        if "Rationale:" in entry:
            rationale = entry.split("Rationale:")[1].strip()
            med["rationale"] = rationale
        else:
            med["rationale"] = ""
        
        medications.append(med)
    
    return medications

def get_treatment_recommendations_v1(patient_case, diagnosis, api_url="http://127.0.0.1:8000/ttx/v1", model="gemini-2.5-flash"):
    """
    Get treatment recommendations from the TTX API v1
    
    Args:
        patient_case (str): Patient case description
        diagnosis (str): Diagnosis for the patient
        api_url (str): URL for the TTX API v1
        model (str): Model name to use for recommendations
        
    Returns:
        dict: Dictionary with keys:
            - success (bool): Whether the API call succeeded
            - data (dict): The treatment recommendations
            - error (str, optional): Error message if the call failed
    """
    try:
        response = requests.post(
            api_url,
            json={"model_name": model, "case": patient_case, "diagnosis": diagnosis, "tracker": "test"}
        )
        
        if response.status_code == 200:
            response_data = response.json()
            print("------------RESULT --------------------")
            print(response_data)
            print("--------------------------------")
            
            # Check for the new response structure
            if response_data.get("status") == "success" and "data" in response_data:
                return {"success": True, "data": response_data["data"]}
            else:
                # Fallback for original structure or errors
                return {"success": False, "error": response_data.get("detail", "Unknown error")}
        else:
            return {"success": False, "error": f"API error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"success": False, "error": f"Exception: {str(e)}"}

def get_treatment_recommendations_v2(patient_case, diagnosis, api_url="http://127.0.0.1:8000/ttx/v2", model="gemini-2.0-flash"):
    """
    Get treatment recommendations from the TTX API v2
    
    Args:
        patient_case (str): Patient case description
        diagnosis (str): Diagnosis for the patient
        api_url (str): URL for the TTX API v2
        model (str): Model name to use for recommendations
        
    Returns:
        dict: Dictionary with keys:
            - success (bool): Whether the API call succeeded
            - medications (list): Structured medication recommendations
            - medical_advice (str): Medical advice for the patient
            - error (str, optional): Error message if the call failed
    """
    try:
        response = requests.post(
            api_url,
            json={"model_name": model, "case": patient_case, "diagnosis": diagnosis}
        )
        
        if response.status_code == 200:
            response_data = response.json()
            print("--------------------------------")
            print(response_data)
            print("--------------------------------")
            # Parse the response, checking for the expected successful structure (like the provided JSON example) or known error formats
            if response_data.get("data") == "The Input provided does not have enough clinical details for AI based treatment recommendation.":
                return {
                    "success": False, # Indicate failure due to insufficient data
                    "error": response_data.get("data"),
                    "medications": [],
                    "medical_advice": ""
                }

            # Check if the API call was successful based on the structure expected from the server
            # (assuming similar structure to v1 before processing)
            if "data" in response_data and isinstance(response_data["data"], dict) and "output" in response_data["data"]:
                output = response_data["data"]["output"]

                # Check if the necessary keys exist in the output dictionary
                if isinstance(output, dict) and "medication_recommendations" in output and "medical_advice" in output:
                    med_text = output.get("medication_recommendations", "")
                    medical_advice = output.get("medical_advice", "")

                    # Process the raw medication text into a structured list
                    # Ensure process_medications can handle empty or non-string input gracefully if needed
                    processed_medications = []
                    if isinstance(med_text, str) and med_text.strip():
                         processed_medications = process_medications(med_text)
                    elif isinstance(med_text, list): # Handle if server pre-processes (unlikely based on v1)
                         processed_medications = med_text # Assume it's already structured

                    result = {
                        "success": True,
                        "medications": processed_medications,
                        "medical_advice": medical_advice
                    }
                    return result
                else:
                    # Handle cases where 'output' exists but lacks expected keys or is not a dict
                     error_msg = "Missing 'medication_recommendations' or 'medical_advice' in API response output"
                     if not isinstance(output, dict):
                         error_msg = "API response 'output' field is not a dictionary"
                     return {
                        "success": False,
                        "error": error_msg,
                        "medications": [],
                        "medical_advice": ""
                    }
            # Handle cases where the primary response structure is unexpected
            # Also check for the direct success/failure structure provided in the prompt example,
            # in case the v2 API returns the final processed format directly.
            elif response_data.get("success") is True and "medications" in response_data and "medical_advice" in response_data:
                 # Assume medications are already processed if this structure is returned
                 return {
                    "success": True,
                    "medications": response_data.get("medications", []),
                    "medical_advice": response_data.get("medical_advice", "")
                 }
            elif response_data.get("success") is False:
                 # Handle explicit failure indicated by the API
                 return {
                    "success": False,
                    "error": response_data.get("error", "API indicated failure"),
                    "medications": [],
                    "medical_advice": ""
                 }
            else:
                # Fallback for other unexpected formats
                return {
                    "success": False,
                    "error": "Invalid or unexpected v2 response format",
                    "medications": [],
                    "medical_advice": ""
                }
        else:
            return {
                "success": False,
                "error": f"API error: {response.status_code} - {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Exception: {str(e)}"
        }

def get_treatment_recommendations(patient_case, diagnosis, api_url="http://127.0.0.1:8000/ttx", model="gemini-2.5-flash", version="v1", tracker="test"):
    """
    Get treatment recommendations from the TTX API
    
    Args:
        patient_case (str): Patient case description
        diagnosis (str): Diagnosis for the patient
        api_url (str): Base URL for the TTX API
        model (str): Model name to use for recommendations
        version (str): API version to use ('v1' or 'v2')
        
    Returns:
        dict: Dictionary with treatment recommendations
    """
    # if version == "v2":
    #     return get_treatment_recommendations_v2(patient_case, diagnosis, api_url, model)
    # else:
    return get_treatment_recommendations_v1(patient_case, diagnosis, api_url, model)

if __name__ == "__main__":
    # Example patient case
    patient_test_case = """
    Gender: Female
    
     Age: 70 years
    
     Chief_complaint: ► **Cold, Sneezing** :  
    • 1 Days.  
    • Precipitating factors - Cold weather, Wind.  
    • Prior treatment sought - None.  
    ► **Headache** :  
    • Duration - 1 Days.  
    • Site - Localized - कपाळावरती डोकं दुखतंय .  
    • Severity - Moderate.  
    • Onset - Acute onset (Patient can recall exact time when it started).  
    • Character of headache - Throbbing.  
    • Radiation - pain does not radiate.  
    • Timing - Day.  
    • Exacerbating factors - bending.  
    • Prior treatment sought - None.  
    ► **Leg, Knee or Hip Pain** :  
    • Site - Right leg, Hip, Thigh, Knee, Site of knee pain - Front, Back,
    Lateral/medial. Swelling - No, Calf, Left leg, Hip, Thigh, Knee, Site of knee
    pain - Front, Back, Lateral/medial. Swelling - No, Calf, Hip.  
    • Duration - 6 Days.  
    • Pain characteristics - Sharp shooting.  
    • Onset - Gradual.  
    • Progress - Static (Not changed).  
    • Pain does not radiate.  
    • Aggravating
    
     Physical_examination: **General exams:**  
    • Eyes: Jaundice-no jaundice seen, [picture taken].  
    • Eyes: Pallor-normal pallor, [picture taken].  
    • Arm-Pinch skin* - pinch test normal.  
    • Nail abnormality-nails normal, [picture taken].  
    • Nail anemia-Nails are not pale, [picture taken].  
    • Ankle-no pedal oedema, [picture taken].  
    **Joint:**  
    • non-tender.  
    • no deformity around joint.  
    • full range of movement is seen.  
    • joint is not swollen.  
    • no pain during movement.  
    • no redness around joint.  
    **Back:**  
    • tenderness observed.  
    **Head:**  
    • No injury.
    
     Patient_medical_history: • Pregnancy status - Not pregnant.  
    • Allergies - No known allergies.  
    • Alcohol use - No.  
    • Smoking history - Patient denied/has no h/o smoking.  
    • Medical History - None.  
    • Drug history - No recent medication.  
    
     Family_history: •Do you have a family history of any of the following? : None.  
    
     Vitals:- 
    
    Sbp: 140.0
    
     Dbp: 90.0
    
     Pulse: 83.0
    
     Temperature: 36.78 'C
    
     Weight: 44.75 Kg
    
     Height: 152.0 cm
    
     BMI: 19.37
    
     RR: 21.0
    
     SPO2: 97.0
    
     HB: Null
    
     Sugar_random: Null
    
     Blood_group: Null
    
     Sugar_pp: Null
    
     Sugar_after_meal: Null
    """
    
    diagnosis = "Acute Pharyngitis"
    
    # Get recommendations from v1
    print("Getting treatment recommendations from v1...")
    result_v1 = get_treatment_recommendations_v1(patient_test_case, diagnosis)
    
    print("--------------------------------")
    print(result_v1)
    print("--------------------------------")
    # # Get recommendations from v2
    # print("\nGetting treatment recommendations from v2...")
    # result_v2 = get_treatment_recommendations_v2(patient_test_case, diagnosis)
    
    # Print v1 results
    if result_v1 and result_v1.get("success"):
        print("\nV1 Results:")
        print(json.dumps(result_v1.get("data", {}), indent=2))
    elif result_v1:
        print(f"\nV1 Error: {result_v1.get('error', 'Unknown error')}")
    else:
        print("\nV1 Error: Received no response from the server.")
    
    # Print v2 results
    # if result_v2["success"]:
    #     print("\nV2 Results:")
    #     print(json.dumps(result_v2, indent=2))
    # else:
    #     print(f"\nV2 Error: {result_v2.get('error', 'Unknown error')}")



