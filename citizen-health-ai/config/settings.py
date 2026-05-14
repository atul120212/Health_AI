"""Central configuration for Citizen Health AI platform."""
import os
from dotenv import load_dotenv

load_dotenv()

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-opus-4-7"

# Sarvam AI (Tamil/Kannada TTS/STT)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")

# HMIS Middleware (placeholder)
HMIS_BASE_URL = os.getenv("HMIS_BASE_URL", "http://localhost:8080/hmis")
HMIS_API_KEY = os.getenv("HMIS_API_KEY", "")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Supported languages
SUPPORTED_LANGUAGES = {
    "ta": "Tamil",
    "kn": "Kannada",
    "en": "English",
}

# Ayushman Bharat / CMCHIS thresholds
AYUSHMAN_ANNUAL_INCOME_LIMIT = 500000  # ₹5 lakh per year
CMCHIS_ANNUAL_INCOME_LIMIT = 720000    # ₹7.2 lakh per year

# Disease surveillance
OUTBREAK_ALERT_THRESHOLD = 2.0   # z-score threshold for outbreak detection
SURVEILLANCE_LOOKBACK_DAYS = 90  # days of historical data for baseline
EARLY_WARNING_WEEKS = 3          # weeks ahead for early warning
