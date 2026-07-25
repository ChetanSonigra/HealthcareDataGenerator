import json
import requests
from loguru import logger

class DataSynthesizer:
    def __init__(self, api_key: str, endpoint_url: str):
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_prompt(self, case_id: str, state: str, plan_type: str, topic: str) -> str:
        return (
            f"Generate a multi-turn customer support conversation for synthetic case {case_id}. "
            f"The customer profile is a {plan_type} member in {state}. "
            f"The user needs help understanding where to verify a {topic} question. "
            "The assistant must use only public guidance and must not make a coverage decision. "
            "Ensure the resolution emphasizes guidance and human verification."
        )

    def generate_synthetic_data(self, prompt: str) -> dict:
        payload = {
            "model": "nemotron-4-340b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        response = requests.post(self.endpoint_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()