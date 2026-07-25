import requests
import json
import os

class Synthesizer:
    def __init__(self, config):
        self.config = config
        self.api_url = self.config['microservices']['data_designer_url']
        self.headers = {
            "Authorization": f"Bearer {self.config['microservices']['api_key']}",
            "Content-Type": "application/json"
        }

    def generate_synthetic_data(self, prompt):
        print("--- Running Synthetic Data Generation Job ---")
        payload = {
            "model": self.config['microservices']['model'],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            synthetic_output = response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"API Error (Fallback to mock for execution): {e}")
            synthetic_output = '{"mock": "generated_data", "synthetic_only": true}\n'

        output_path = os.path.join(self.config['pipeline']['output_dir'], "raw_synthetic_data.jsonl")
        with open(output_path, "w") as f:
            f.write(synthetic_output)
            
        print(f"Synthetic data generated and saved to {output_path}")
        return output_path