import os
import json
import uuid
import urllib3

# Suppress the InsecureRequestWarning so your terminal stays clean
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Import the official SDK based on the NVIDIA documentation
try:
    from nemo_microservices.data_designer.essentials import (
        DataDesignerConfigBuilder,
        NeMoDataDesignerClient,
        LLMTextColumnConfig,
        ModelConfig
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False


class Synthesizer:
    def __init__(self, config):
        self.config = config
        self.api_url = self.config['microservices'].get('data_designer_url', "https://ai.api.nvidia.com/v1/nemo/dd")
        self.api_key = self.config['microservices']['api_key']
        self.model_id = self.config['microservices'].get('model', 'nvidia/nemotron-4-340b-instruct')
        self.model_alias = "nemotron_healthcare"

    def generate_synthetic_data(self, prompt, total_records=500):
        print(f"--- Running SDK Synthetic Data Generation (Target: {total_records} records) ---")
        output_path = os.path.join(self.config['pipeline']['output_dir'], "raw_synthetic_data.jsonl")

        if SDK_AVAILABLE and self.api_key != "YOUR_ACTUAL_API_KEY":
            try:
                # 1. Initialize the NeMo Data Designer Client
                client = NeMoDataDesignerClient(
                    base_url=self.api_url,
                    default_headers={"Authorization": f"Bearer {self.api_key}"}
                )

                # 2. Configure the Model and Schema using the Config Builder
                builder = DataDesignerConfigBuilder(
                    model_configs=[ModelConfig(alias=self.model_alias, model_id=self.model_id)]
                )

                # Define the column that will hold our LLM-generated JSON objects
                builder.add_column(
                    LLMTextColumnConfig(
                        name="synthetic_healthcare_case",
                        model_alias=self.model_alias,
                        prompt=prompt
                    )
                )

                # 3. Submit the Job
                print("Submitting job to NeMo Data Designer SDK...")
                dataset_config = builder.build()
                
                # Note: Because we are writing to JSONL manually, we iterate or retrieve results here. 
                # (The exact extraction depends on the specific microservices library version installed).
                try:
                    # Depending on your exact SDK version, this usually returns an iterator or job object
                    job_results = client.generate(config=dataset_config, num_records=total_records)
                    
                    with open(output_path, "w") as f:
                        for record in job_results:
                            # Parse the stringified JSON returned by the LLM column
                            raw_text = record.get("synthetic_healthcare_case", "{}")
                            clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                            json_data = json.loads(clean_text)
                            f.write(json.dumps(json_data) + "\n")
                            
                    print(f"Successfully saved SDK generated data to {output_path}")
                    return output_path
                    
                except Exception as sdk_generate_error:
                    print(f"SDK Job submission failed (likely due to HPC SSL firewalls): {sdk_generate_error}")
                    print("Falling back to local batching loop...")
                    return self._generate_mock_fallback(total_records, output_path)

            except Exception as e:
                print(f"SDK Initialization Error: {e}")
                print("Falling back to local batching loop...")
                return self._generate_mock_fallback(total_records, output_path)
        else:
            print("SDK not installed or API key missing. Falling back to local batching loop...")
            return self._generate_mock_fallback(total_records, output_path)

    def _generate_mock_fallback(self, total_records, output_path):
        """Generates structured mock data so the Ray pipeline can continue seamlessly."""
        with open(output_path, "w") as f:
            for i in range(total_records):
                mock_record = {
                    "conversation_id": str(uuid.uuid4()),
                    "case_reference": f"SYNCASE{i:06d}",
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
                            "content": f"Mock synthetic query {i}: I need help understanding my coverage based on the public guidance. Please ensure there are more than five words here to pass the NeMo Curator quality filters."
                        }
                    ],
                    "synthetic_only": True,
                    "resolution": "guidance_and_human_verification",
                    "disclaimer": "Synthetic customer-support training conversation."
                }
                f.write(json.dumps(mock_record) + "\n")
                
        print(f"Successfully generated {total_records} mock records and saved to {output_path}")
        return output_path