"""
TTx Server Utilities
Provides LLM configuration functions for treatment recommendation models.
"""

import os
import dspy
from dotenv import load_dotenv

# Load environment variables
load_dotenv("ops/.env")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def load_gemini2_5_lm():
    """
    Configure DSPy to use Google Gemini 2.5 Flash model.

    This model is used for:
    - Main treatment recommendation generation (TTxv3Module)
    - Optional transformation of LLM outputs to structured JSON
    """
    gemini = dspy.Google(
        model="models/gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        max_output_tokens=10000
    )
    dspy.settings.configure(lm=gemini, top_k=5)


def load_groq_llama_4_maverick():
    """
    Configure DSPy to use Groq Llama 4 Maverick 17B model.

    This model is used for:
    - Main treatment recommendation generation (TTxv3Module)
    - High-quality medical reasoning with extended context
    """
    llama = dspy.LM(
        model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        api_key=GROQ_API_KEY,
        max_tokens=8192
    )
    dspy.configure(lm=llama, top_k=5)


def load_groq_llama_scout():
    """
    Configure DSPy to use Groq Llama 4 Scout 17B model.

    This model is used for:
    - Response transformation to structured JSON
    - Fast structured output generation
    """
    llama = dspy.LM(
        model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=GROQ_API_KEY,
        max_tokens=8192
    )
    dspy.configure(lm=llama, top_k=5)
