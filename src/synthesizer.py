import os
import json
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
except ImportError as e:
    SDK_AVAILABLE = False
    SDK_ERROR = e

class Synthesizer:
    def __init__(self, config):
        self.config = config
        self.api_url = self.config['microservices'].get('data_designer_url', "https://ai.api.nvidia.com/v1/nemo/dd")
        self.api_key = self.config['microservices']['api_key']
        self.model_id = self.config['microservices'].get('model', 'nvidia/nemotron-4-340b-instruct')
        self.model_alias = "nemotron_healthcare"

    def generate_synthetic_data(self, prompt, total_records=10): # Lowered to 10 for faster testing!
        print(f"--- Running STRICT SDK Synthetic Data Generation (Target: {total_records} records) ---")
        output_path = os.path.join(self.config['pipeline']['output_dir'], "raw_synthetic_data.jsonl")

        # 🚨 HARD CHECK 1: The API Key 🚨
        if self.api_key in ["YOUR_ACTUAL_API_KEY", "YOUR_NVIDIA_API_KEY", ""]:
            raise ValueError(
                "\n\n🛑 [CRITICAL ERROR] Missing API Key! 🛑\n"
                "You are still using the placeholder API key. You must paste a real NVIDIA NIM API key into your config/pipeline_config.yaml file.\n"
            )

        # 🚨 HARD CHECK 2: The SDK Installation 🚨
        if not SDK_AVAILABLE:
            raise ImportError(
                f"\n\n🛑 [CRITICAL ERROR] SDK Missing! 🛑\n"
                f"Python cannot find the NeMo Microservices SDK. The specific error was: {SDK_ERROR}\n"
                f"Please run this in your terminal: pip install \"nemo-microservices[data-designer]\"\n"
            )

        # 1. Initialize the NeMo Data Designer Client
        client = NeMoDataDesignerClient(
            base_url=self.api_url,
            default_headers={"Authorization": f"Bearer {self.api_key}"}
        )

        # 2. Configure the Model and Schema using the Config Builder
        # FIX: Changed 'model_id' to 'model' to satisfy Pydantic validation
        builder = DataDesignerConfigBuilder(
            model_configs=[ModelConfig(alias=self.model_alias, model=self.model_id)]
        )

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
        
        job_results = client.generate(config=dataset_config, num_records=total_records)
        
        with open(output_path, "w") as f:
            for record in job_results:
                raw_text = record.get("synthetic_healthcare_case", "{}")
                clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                try:
                    json_data = json.loads(clean_text)
                    f.write(json.dumps(json_data) + "\n")
                except json.JSONDecodeError:
                    # In case the LLM outputs malformed JSON for a single record, skip it and print a warning
                    print("Warning: Skipping a record due to malformed JSON output from LLM.")
                
        print(f"✅ Successfully saved REAL generated data to {output_path}")
        return output_path