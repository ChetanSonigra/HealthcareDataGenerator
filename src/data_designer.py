import json

class DataDesigner:
    def __init__(self, config):
        self.config = config

    def generate_prompt(self):
        print("--- Generating Prompt via Data Designer (Zero-Shot) ---")
        
        # We only load the manifest context. No sample data is loaded.
        with open(self.config['pipeline']['manifest_path'], 'r') as f:
            manifest_data = json.load(f)

        sources = manifest_data.get("sources", [])
        context_docs = "\n".join([f"- {s['title']} ({s['category']})" for s in sources[:5]])

        # Explicitly define the desired JSON schema inside the prompt
        prompt = f"""
        You are an expert synthetic data generator for healthcare customer support.
        Using the following reference documents for grounding:
        {context_docs}

        Generate 10 highly realistic synthetic customer support cases. 
        Focus on 'benefits_eligibility' and 'claims_billing' for Medicare Advantage and Medicare Part D.

        Please output the data strictly as a JSON array of objects with the following keys:
        - "conversation_id": A unique alphanumeric string hash.
        - "case_reference": A unique case ID (e.g., SYNCASE000001).
        - "use_case": The category of the case.
        - "customer_profile": An object containing "age_band", "state", "plan_type", and "channel".
        - "turns": An array of dialogue objects containing "turn_index", "role" (customer/assistant), and "content" with the actual dialogue.
        - "synthetic_only": true
        - "resolution": "guidance_and_human_verification"
        - "disclaimer": "Synthetic customer-support training conversation."

        Requirements:
        1. Do not include actual Protected Health Information (PHI).
        2. Ensure the response is valid, parseable JSON only. No markdown formatting blocks.
        """
        print("Prompt successfully generated without example template.")
        return prompt