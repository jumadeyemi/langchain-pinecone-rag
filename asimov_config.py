import os

# Read Asimov settings from environment (preferred) with sensible fallbacks
ASIMOV_API_KEY = os.environ.get("ASIMOV_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = (
	os.environ.get("BASE_URL")
	or os.environ.get("ASIMOV_BASE_URL")
	or os.environ.get("OPENAI_API_BASE")
)

# Export into OpenAI-compatible environment variables so existing code works
if ASIMOV_API_KEY:
	os.environ["OPENAI_API_KEY"] = ASIMOV_API_KEY

if BASE_URL:
	os.environ["OPENAI_API_BASE"] = BASE_URL

