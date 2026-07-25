import json

class DataDesigner:
    def __init__(self, config):
        self.config = config

    def generate_prompt(self):
        print("--- Generating Prompt via Data Designer ---")
        
        # Load sample data structure robustly (handles both standard JSON arrays and JSONL)
        with open(self.config['pipeline']['generated_data_path'], 'r') as f:
            try:
                data = json.load(f)
                # If it's a list, grab the first case as the schema template
                sample_data = data[0] if isinstance(data, list) else data
            except json.JSONDecodeError:
                # Fallback if the file happens to be formatted as JSON Lines
                f.seek(0)
                sample_data = json.loads(f.readline())
                
        with open(self.config['pipeline']['manifest_path'], 'r') as f:
            manifest_data = json.load(f)

        sources = manifest_data.get("sources", [])
        context_docs = "\n".join([f"- {s['title']} ({s['category']})" for s in sources[:5]])

        prompt = f"""
        You are an expert synthetic data generator for healthcare customer support.
        Using the following reference documents for grounding:
        {context_docs}

        Generate 10 highly realistic synthetic customer support cases. 
        Each case must strictly adhere to the following JSON schema and include a fictional Medicare scenario:
        {json.dumps(sample_data, indent=2)}
        
        Requirements:
        1. Maintain synthetic flags: "synthetic_only": true.
        2. Resolve to "guidance_and_human_verification".
        3. Do not include actual Protected Health Information (PHI).
        4. Focus on 'benefits_eligibility' and 'claims_billing'.
        """
        print("Prompt successfully generated.")
        return prompt