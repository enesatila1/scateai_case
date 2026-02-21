import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    api_key = os.getenv("api_key", "")
    port = int(os.getenv("PORT", 8000))
    mureka_api_url = "https://api.mureka.ai/v1"

app_config = AppConfig()
