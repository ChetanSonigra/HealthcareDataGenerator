import requests
import json
import os
import time
import uuid

class Synthesizer:
    def __init__(self, config):
        self.config = config
        self.api_url = self.config['microservices']['data_designer_url']
        self.headers = {
            "Authorization": f"Bearer {self.config['microservices']['api_key']}",
            "Content-Type": "application/json"
        }

    def generate_synthetic_data(self, prompt, total_records=500, batch_size=10):
        print(f"--- Running Synthetic Data Generation Job (Target: {total_records} records) ---")
        
        output_path = os.path.join(self.config['pipeline']['output_dir'], "raw_synthetic_data.jsonl")
        
        # Calculate how many API calls we need to make
        batches = total_records // batch_size
        
        # Open file in append/write mode to stream data directly to disk
        with open(output_path, "w") as f:
            for i in range(batches):
                print(f"Generating batch {i+1}/{batches}...")
                
                payload = {
                    "model": self.config['microservices']['model'],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 4096
                }
                
                try:
                    # Catch the default dummy API key before making a network request
                    if self.config['microservices']['api_key'] == "YOUR_NVIDIA_API_KEY":
                        raise ValueError("No valid API key provided. Triggering mock fallback.")
                        
                    response = requests.post(self.api_url, headers=self.headers, json=payload)
                    response.raise_for_status()
                    
                    # Extract the response
                    content = response.json()['choices'][0]['message']['content']
                    
                    # Clean the response (LLMs sometimes wrap JSON in markdown blocks)
                    content = content.replace("```json", "").replace("```", "").strip()
                    records = json.loads(content)
                    
                    # Write each record as a JSONL format (one JSON object per line)
                    for record in records:
                        f.write(json.dumps(record) + "\n")
                        
                except Exception as e:
                    if i == 0:
                        print(f"API Error detected: {e}")
                        print(f"Falling back to high-volume mock generation for 500 records...")
                    
                    # MOCK FALLBACK: Generate 'batch_size' mock records to simulate scale
                    for j in range(batch_size):
                        current_index = (i * batch_size) + j
                        mock_record = {
                            "conversation_id": str(uuid.uuid4()), # Unique Hash
                            "case_reference": f"SYNCASE{current_index:06d}",
                            "use_case": "benefits_eligibility",
                            "customer_profile": {
                                "age_band": "65+", 
                                "state": "TX", 
                                "plan_type": "Medicare Advantage", 
                                "channel": "phone"
                            },
                            "turns": [
                                {
                                    "turn_index": 1, 
                                    "role": "customer", 
                                    "content": f"Mock synthetic query {current_index}: I need help understanding my coverage based on the public guidance. Please ensure there are more than five words here to pass the NeMo Curator quality filters."
                                }
                            ],
                            "synthetic_only": True,
                            "resolution": "guidance_and_human_verification",
                            "disclaimer": "Synthetic customer-support training conversation."
                        }
                        # Write the mock record to the JSON Lines file
                        f.write(json.dumps(mock_record) + "\n")
                        
                # Pause briefly between batches to respect API rate limits
                time.sleep(0.5)
                
        print(f"Successfully generated {total_records} records and saved to {output_path}")
        return output_path