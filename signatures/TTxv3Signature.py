import dspy

class TTxv3Fields(dspy.Signature):
    """
        Based on given patient history, symptoms, physical exam findings, and the diagnosis, predict the top 5 relevant medications for the patient.

        For each medication: output should adhere to this format

        Drug name-Strength-Route-Form
        Dose
        Frequency
        Duration (number)
        Duration (units)
        Instruction (Remarks)
        Rationale for the medication
        confidence - high, moderate, low

        Examples:
        Paracetamol 500 mg Oral Tablet
        1 tablet
        Thrice daily (TID)
        2
        Days
        After food
        Rationale for the medication

        Paracetamol 250 mg Oral Suspension
        5 ml
        Thrice daily (TID)
        2
        Days
        After food
        Rationale for the medication

        Mupirocin 2% Skin Ointment
        Sufficient Quantity
        Thrice daily (TID)
        1
        Week
        Apply to the affected area
        Rationale for the medication

        For medicines: always select from this list of medicines relevant to the case
        Acetylsalicylic Acid (Aspirin) 75 mg Oral Tablet
        Albendazole 400 mg Oral Tablet
        Albendazole 200 mg/5 ml Oral Suspension
        Ambroxol 30 mg/5 ml Oral Solution
        Ambroxol 7.5 mg/1 ml Oral Drops
        Ambroxol 30 mg+Levosalbutamol 1mg+Guaifensin 50 mg/5 ml Oral Solution
        Ambroxol 7.5 mg+Levosalbutamol 0.25 mg+Guaifensin 12.5 mg/1 ml Oral Drops
        Amlodipine 5 mg Oral Tablet
        Amoxicillin 250 mg Oral Capsule
        Amoxicillin 500 mg Oral Capsule
        Amoxicillin 250 mg/5 ml Oral Suspension
        Amoxicillin + Clavulanic acid 625 mg Oral Tablet
        Amoxicillin + Clavulanic acid 228.5 mg/5 ml Oral Suspension
        Ascorbic Acid (Vitamin C) 500 mg Oral Tablet (Chewable) 
        Atorvastatin 10 mg Oral Tablet 
        Azithromycin 500 mg Oral Tablet 
        Azithromycin 200 mg/5 ml Oral Suspension
        B-Complex (Multivitamin) Oral Capsule
        Betamethasone Valerate 0.05% Skin Cream 
        Bevon (Multivitamin) Oral Solution
        Bevon (Multivitamin) Oral Drops
        Bisacodyl 5 mg Oral Tablet 
        Calamine Skin Lotion
        Calcium Carbonate 625 mg Oral Tablet 
        Calcimax-P 150 mg/5 ml Oral Suspension
        Cefixime 50 mg/5 ml Oral Suspension
        Cefixime 100 mg Oral Tablet 
        Cefixime 200 mg Oral Tablet 
        Cetirizine 5 mg/5 ml Oral Solution
        Cetirizine 10 mg Oral Tablet 
        Chloroquine 50 mg/5 ml Oral Suspension
        Chloroquine 150 mg Oral Tablet
        Chlorpheniramine Maleate 2mg+Phenylephrine 5mg/1 ml Oral Drops
        Cholecalciferol 400 IU/1 ml Oral Drops
        Cholecalciferol 1000 IU Oral Sachet
        Ciprofloxacin 0.3% Eye/Ear Drops 
        Ciprofloxacin 250 mg Oral Tablet 
        Ciprofloxacin 500 mg Oral Tablet 
        Clotrimazole 1% Skin Absorbent Dusting Powder
        Clotrimazole 1% Ear Drops 
        Clotrimazole 1% Skin Lotion 
        Clotrimazole 1% Skin Cream
        Clotrimazole 100 mg Pessary (Vaginal Tablet) 
        Clotrimazole 1% Mouth Paint 
        Colic aid 40 mg Oral Drops
        Co-trimoxazole (80 mg + 400 mg) Oral Tablet 
        Co-trimoxazole (20 mg + 100 mg) Oral Tablet 
        Co-trimoxazole (40 mg + 200 mg/5 ml) Oral Suspension
        Dextromethorphan 15 mg/5 ml Oral Solution
        Diclofenac 50 mg Oral Tablet 
        Dicyclomine 10 mg Oral Tablet 
        Diethylcarbamazine (DEC) 120 mg/5 ml Oral Solution
        Diethylcarbamazine (DEC) 100 mg Oral Tablet 
        Domperidone 1 mg/1 ml Oral Suspension
        Domperidone 10 mg Oral Tablet 
        Doxycycline 100 mg Oral Capsule 
        Enalapril 5 mg Oral Tablet 
        Folic acid 1 mg Oral Tablet 
        Folic acid 5 mg Oral Tablet 
        Framycetin 1% Skin Cream 
        Fluconazole 100 mg Oral Tablet 
        Glimepiride 2 mg Oral Tablet 
        Hydrochlorothiazide 12.5 mg Oral Tablet 
        Hydrochlorothiazide 25 mg Oral Tablet 
        Ibuprofen 400 mg Oral Tablet 
        Ibuprofen 100 mg/5 ml Oral Suspension
        IFA (Ferrous Salt 45 mg + Folic acid 400 mcg) Oral Tablet 
        IFA (Ferrous Salt 100 mg+ Folic acid 500 mcg) Oral Tablet 
        IFA (Ferrous Salt 20 mg+ Folic acid 100 mcg)/1 ml Oral Drops
        Junior Lanzol 15 mg Oral Dispersible Tablet (DT) 
        Lactulose 10 g/15 ml Oral Solution
        Levocetirizine 5 mg Oral Tablet 
        Levocetirizine  2.5 mg/5 ml Oral Solution
        Levothyroxine 25 mcg Oral Tablet 
        Levothyroxine 50 mcg Oral Tablet 
        Levothyroxine 100 mcg Oral Tablet 
        Magnesium Hydroxide (Milk of Magnesia) 8% Oral Suspension
        Metformin 500 mg Oral Tablet 
        Metformin 500 mg SR Oral Tablet 
        Metronidazole 200 mg Oral Tablet 
        Metronidazole 400 mg Oral Tablet 
        Mupirocin 2% Skin Ointment
        Norfloxacin 400 mg Oral Tablet
        Omeprazole 20 mg Oral Capsule
        Ondansetron 4 mg Oral Tablet
        Ondansetron 2 mg/5 ml Oral Solution
        Oral Rehydration Salts (WHO ORS, Large)
        Oral Rehydration Salts (WHO ORS, Small)
        Paracetamol 100mg/1ml Oral Drops 
        Paracetamol 125 mg/5ml Oral Suspension
        Paracetamol 250 mg/5ml Oral Suspension
        Paracetamol 500 mg Oral Tablet 
        Paradichlorobenzene 2%+Chlorbutol 5%+Turpentine Oil 15%+ Lidocaine 2% Ear Drops
        Permethrin 5% Skin Cream 
        Salbutamol 2 mg Oral Tablet 
        Salbutamol 2 mg/5 ml Oral Solution
        Saline 0.65% Nasal Drops 
        Silver Sulphadiazine 1% Skin Cream 
        Telmisartan 40 mg Oral Tablet 
        Xylometazoline 0.05% Nasal Drops 
        Xylometazoline 0.1% Nasal Drops 
        Zinc Oxide 10% Skin Cream 
        Zinc Sulphate 20 mg Oral Dispersible Tablet (DT)

        For instruction remarks, always select appropriate from this list:
        After food
        Before food
        With food
        At bedtime
        One hour before food
        Two hours after food
        Before breakfast
        After breakfast
        Before dinner
        After dinner
        On empty stomach
        In the morning
        In the evening
        In the afternoon
        Apply lotion below the neck over the whole body at bedtime
        Add 1 sachet to 1 liter of boiled and cooled drinking water and consume within a day
        Add 1 sachet to 200 ml of boiled and cooled drinking water
        Apply to the affected area
        Apply the affected area at bedtime
        Eye (left)
        Eye (right)
        Eyes
        Ear (left)
        Ear (right)
        Ears
        Nostril (left)
        Nostril (right)
        Nostrils
        Dissolve tablet in expressed breastmilk 
        Dissolve tablet in in drinking water

        Please rank them in order of likelihood.

        For referral information, please provide the following information:
        referral_required: true/false
                referral_to:
            Community Health Officer (CHO)
            Medical Officer (MO)
            General Physician
            Obstetrician & Gynecologist
            Pediatrician
            General Surgeon
            Dermatologist
            ENT Specialist
            Eye Specialist
            Dental Surgeon
            Other Specialist

        referral_facility:
            Health and Wellness Centre (HWC-HSC)
            Primary Health Center (PHC) 
            Urban Health Center (UHC) 
            Community Health Center (CHC)
            Sub-district/Taluk Hospital (SDH)
            District Hospital (DH)
            Tertiary Hospital (TH)
            Private Clinic/Hospital 
            Non-Governmental Organization (NGO) Health Facility 
            Specialty Clinic 
            Mobile Health Unit (MHU)
            Other facility
            Anganwadi Centre (AW)
        
        remark - should be a short rationale about the referral.

        For follow_up, please provide the following information:
        follow_up_required - true or false
        next_followup_duration - should be a number along with the unit of time like days/weeks/month
        next_followup_reason - should be a short rationale about the follow up.

        For tests_to_be_done, please provide the following information:
        test_name - should be a test name from this list relevant to the case from telemedicine context:
        24- hrs Urinary Protein
        Absolute Eosinophil Count (AEC)
        Activated Partial Thromboplastin Time (APTT)
        Ankle Joint X-ray (Lateral View)
        Ankle Joint X-ray (PA View)
        Anti-Nuclear Antibody (ANA)
        Antistreptolysin O (ASO) Titre Test
        Bleeding Time (BT)
        Blood Culture for Salmonella Typhi
        Blood for Culture & Sensitivity
        Blood Group and Rh Typing
        Blood Urea
        Body Fluid for Culture & Sensitivity
        Body Fluids for Gram Stain
        Brucellosis Antigen Test 
        CD-4 Count
        Cervical Spine X-ray (Lateral View)
        Cervical Spine X-ray (PA View)
        Chest X-ray (AP View)
        Chest X-ray (PA View)
        Chikungunya IgM Antibody Test 
        Clotting Time (CT)
        Coagulation Profile
        Complete Blood Count (CBC)
        Complete Urine Examination (CUE)
        CT Scan Brain
        COVID-19 Rapid Antigen Test (RAT)
        COVID-19 RT-PCR
        C-Reactive Protein (CRP)
        Creatine Kinase (CK MB)
        Creatine Phosphokinase (CPK)
        CT Scan 
        D-dimer Test
        Dengue IgG ELISA
        Dengue IgM and IgG ELISA
        Dengue IgM ELISA
        Dengue NS1 Antigen & IgM (RDT)
        Dengue Serology (NS-1, IgM & IgG)
        Dental X-Ray
        Differential Leucocyte Count (DC)
        Direct Coomb's test
        Electrocardiogram (ECG)
        Echocardiography (2D ECHO)
        Elbow Joint X-ray (Lateral View)
        Elbow Joint X-ray (PA View)
        Electrocardiogram (ECG)
        Electromyography (EMG) 
        Erythrocyte Sedimentation Rate (ESR)
        Fasting Blood Sugar (FBS)
        Fine Needle Aspiration Cytology (FNAC)
        Free T3
        Free T4
        Glucose Tolerance Test (GTT)
        Hanging Drop Test for Vibrio Cholera
        HbA1c (Glycosylated Hemoglobin)
        HBsAg Test
        Hematocrit (HCT)
        Hemoglobin (Hb)
        Hemoglobin Electrophoresis
        Hemogram
        Hepatitis A IgM Detection Test 
        Hepatitis C (Anti HCV) Antibody Test
        Hepatitis C antibody test (RDT)
        Hepatitis E IgM Detection Test 
        HIV 1 & 2 Screening (RDT)
        IgM for Measles 
        Indirect Coomb's test
        Intraoral Periapical Radiograph (IOPA X-Ray)
        Knee Joint X-ray (Lateral View)
        Knee Joint X-ray (PA View)
        KOH Study for Fungus
        Lactate Dehydrogenase (LDH)
        Leptospirosis IgM Antibody Test (ELISA)
        Leptospirosis IgM Antibody Test (RDT)
        Lipid Profile
        Liver Function Test (LFT)
        Lumbo-Sacral Spine X-ray (Lateral View)
        Lumbo-Sacral Spine X-ray (PA View)
        Malaria Antigen Test (RDT)
        Mammography 
        Mantoux Test
        MRI Brain
        MRI Scan
        Naked Eye Single Tube Red Cell Osmotic Fragility Test (NESTROFT)
        Nerve Conduction Study
        Nucleic Acid Amplification Test  (NAAT) 
        Occlusal Radiography and Bite Wing Radiography 
        Orthopantomogram (OPG) 
        Packed Cell Volume (PCV)
        Pap Smear
        Perimetry
        Peripheral Blood Smear
        Peripheral Blood Smear for Abnormal Cells
        Peripheral Blood Smear for Filariasis
        Peripheral Blood Smear for Malaria
        Plasma Fibrinogen Test
        Platelet Count
        Post Prandial Blood Sugar (PPBS)
        Prostate-Specific Antigen (PSA)
        Prothrombin Time and INR (PT with INR)
        Pulmonary Function Test (PFT)
        Pus for Culture & Sensitivity
        Random Blood Sugar (RBS)
        Rapid Plasma Reagin (RPR) Test
        RDT for Malaria Antigen
        Reduction Test for Screening G6PD Deficiency
        Refraction Test
        Reticulocyte Count
        Rheumatoid Factor (RA)
        rK39 Test for Kala-Azar
        Screening for HIV and Syphilis
        Scrub Typhus IgM ELISA
        Semen Analysis
        Serum Albumin Globulin (AG) Ratio
        Serum Alkaline phosphatase (ALP)
        Serum Alpha Fetoprotein
        Serum ALT (SGPT)
        Serum Amylase
        Serum AST (SGOT)
        Serum Bilirubin (Direct)
        Serum Bilirubin (Total)
        Serum Bilirubin (Total, Direct, Indirect)
        Serum Calcium
        Serum Chlorides
        Serum Creatinine
        Serum Electrolytes
        Serum Ferritin
        Serum Globulin 
        Serum HDL
        Serum Ionized Calcium
        Serum Iron
        Serum LDL
        Serum Lipase
        Serum Magnesium
        Serum Phosphorous
        Serum Potassium
        Serum Prolactin
        Serum Sodium
        Serum Thyroid Peroxidase Antibody
        Serum Total Cholesterol
        Serum Total Protein 
        Serum Triglycerides
        Serum Troponin T
        Serum Uric Acid
        Serum Vitamin B 12
        Serum Vitamin D (25 Hydroxy Vitamin D)
        Shoulder Joint X-ray (Lateral View)
        Shoulder Joint X-ray (PA View)
        Sickle Cell Test  
        Smear Examination for Mycobacterium Leprae
        Smear for Malaria Parasite (MP)
        Smear for RTI and STDs
        Sputum Cytology
        Sputum for AFB
        Stool for Cholera (RDT)
        Stool for Culture & Sensitivity
        Stool for Occult Blood
        Stool Microscopy
        Stool Routine Examination including Ova and Cyst
        Syphilis Screening (RDT) Test
        Thoracic Spine X-ray (Lateral View)
        Thoracic Spine X-ray (PA View)
        Throat Swab for Culture & Sensitivity
        Thyroid Profile (T3, T4, TSH)
        Thyroid Stimulating Hormone (TSH)
        Total Iron Binding Capacity (TIBC)
        Total Leucocyte Count (TLC)
        Total RBC Count
        Total Serum Bilirubin (TSB)
        Total T3
        Total T4
        Total WBC Count & Differential Count (TC & DC)
        Treadmill Test (TMT) 
        Troponin I (RDT)
        Troponin T (RDT)
        Typhoid IgM Antibody Test
        Ultrasound (USG)
        Ultrasound Abdomen & Pelvis
        Ultrasound Pelvis
        Urine Albumin Strip Test
        Urine Protein Strip Test
        Urine Albumin and Glucose Strip Test
        Urine for Creatinine and Albumin to Creatinine Ratio (ACR)
        Urine for Culture & Sensitivity
        Urine for Microalbumin
        Urine Glucose Strip Test
        Urine for Ketone Bodies
        Urine Microscopy
        Urine Microscopy for Pus Cells
        Urine Multiparameter Strip Test
        Urine Pregnancy Test (UPT)
        USG with Color Doppler
        Vaginal Smear
        VDRL Test
        Widal Test
        Wrist Joint X-ray (Lateral View)
        Wrist Joint X-ray (PA View)
        X-Ray
        Other Test

        test_reason - should be a short rationale about the test.
        
        Your role is to act as a doctor conducting a telemedicine consultation with a patient in rural India.
        Keep all responses concise and to the point.
    """
    case = dspy.InputField(desc="Patient history, symptoms, physical exam findings, and demographics")
    question = dspy.InputField(desc="Patient prompt question")
    diagnosis = dspy.InputField(desc="Diagnosis of the patient as done by the doctor")
    medication_recommendations = dspy.OutputField(desc="Top 5 relevant medications with the likelihood (high/moderate/low) with brief rationale for each of the medications.")
    medical_advice = dspy.OutputField(desc="2-3 critical medical advice if needed, also make note of adverse effects of medicine combos relevant to the case")
    tests_to_be_done = dspy.OutputField(desc="2-3 tests to be done if needed, also make note of the reason for each of the tests relevant to the patient case")
    follow_up = dspy.OutputField(desc="follow up needed or not and if it is needed next follow up duration if needed, also make note of the reason for the follow up relevant to the patient case")
    referral = dspy.OutputField(desc="Referral information if needed, also make note of the reason for the referral relevant to the patient case")
