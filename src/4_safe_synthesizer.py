import requests

class SafeSynthesizerValidator:
    def __init__(self, nvcf_url: str, api_key: str):
        self.nvcf_url = nvcf_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def run_safety_job(self, synthetic_conversation: dict) -> bool:
        """
        Evaluates the generated data to verify the current member-specific plan document 
        and official channel are used, and avoids sending PHI[cite: 1].
        """
        payload = {
            "content": synthetic_conversation,
            "policies": ["no_phi", "no_medical_advice", "grounded_only"]
        }
        
        response = requests.post(f"{self.nvcf_url}/v1/safety/evaluate", headers=self.headers, json=payload)
        
        if response.status_code == 200:
            results = response.json()
            # Returns True if all safety checks pass
            return results.get("is_safe", False)
        else:
            raise Exception(f"Safety evaluation failed: {response.text}")

# Integration
# validator = SafeSynthesizerValidator("NVIDIA_CLOUD_FUNCTION_URL", "YOUR_API_KEY")
# is_safe = validator.run_safety_job(generated_conversation)