import requests
from loguru import logger

class SafeSynthesizerValidator:
    def __init__(self, nvcf_url: str, api_key: str):
        self.nvcf_url = nvcf_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def run_safety_job(self, synthetic_conversation: dict) -> bool:
        payload = {
            "content": synthetic_conversation,
            "policies": ["no_phi", "no_medical_advice", "grounded_only"]
        }
        response = requests.post(f"{self.nvcf_url}/v1/safety/evaluate", headers=self.headers, json=payload)
        if response.status_code == 200:
            results = response.json()
            return results.get("is_safe", False)
        else:
            raise Exception(f"Safety evaluation failed: {response.text}")