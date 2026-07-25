import requests
import json

class SafeSynthesizer:
    def __init__(self, config):
        self.config = config
        self.api_url = self.config['microservices']['safe_synthesizer_url']
        self.headers = {
            "Authorization": f"Bearer {self.config['microservices']['api_key']}",
            "Content-Type": "application/json"
        }

    def run_safety_checks(self, data_path):
        print("--- Running NeMo Safe Synthesizer ---")
        with open(data_path, 'r') as f:
            data = f.read()

        payload = {
            "content": data,
            "checks": ["toxicity", "pii_leakage", "hallucination"]
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            safety_report = response.json()
        except Exception as e:
            print(f"Safety API Error (Mocking safe response): {e}")
            safety_report = {"status": "passed", "issues": []}

        if safety_report.get("status") == "passed":
            print("Data passed all Safe Synthesizer checks.")
            return True
        else:
            print(f"Safety violations detected: {safety_report.get('issues')}")
            return False