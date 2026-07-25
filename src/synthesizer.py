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
        self.api_key = self.config['microservices']['api_key']
        self.model_id = self.config['microservices'].get('model', 'nvidia/nemotron-4-340b-instruct')
        self.model_alias = "nemotron_healthcare"

    def generate_synthetic_data(self, prompt, total_records=10):
        print(f"--- Running STRICT SDK Synthetic Data Generation (Target: {total_records} records) ---")
        output_path = os.path.join(self.config['pipeline']['output_dir'], "raw_synthetic_data.jsonl")

        # 🚨 HARD CHECK 1: The API Key 🚨
        if self.api_key in ["YOUR_ACTUAL_API_KEY", "YOUR_NVIDIA_API_KEY", ""]:
            raise ValueError(
                "\n\n🛑 [CRITICAL ERROR] Missing API Key! 🛑\n"
                "You must paste a real NVIDIA NIM API key into your config/pipeline_config.yaml file.\n"
            )

        # 🚨 HARD CHECK 2: The SDK Installation 🚨
        if not SDK_AVAILABLE:
            raise ImportError(
                f"\n\n🛑 [CRITICAL ERROR] SDK Missing! 🛑\n"
                f"Python cannot find the NeMo Microservices SDK. The specific error was: {SDK_ERROR}\n"
            )

        # 1. Initialize the NeMo Data Designer Client
        # FIX: Removed the base_url override so the SDK uses its correct native endpoints!
      # 1. Initialize the NeMo Data Designer Client
        print("Initializing SDK Client...")
        client = NeMoDataDesignerClient(
            base_url="https://api.nvidia.com", # Root domain only!
            default_headers={"Authorization": f"Bearer {self.api_key}"}
        )

        # 2. Configure the Model and Schema using the Config Builder
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
        job_results = client.create(builder, num_records=total_records)
        
        print("Waiting for job to complete (this may take a minute)...")
        job_results.wait_until_done()
        
        # Load the final generated dataset
        dataset = job_results.load_dataset()
        
        # Convert to a list of dicts (handles both Pandas DataFrames and HuggingFace dataset types)
        records = dataset.to_dict(orient="records") if hasattr(dataset, "to_dict") else dataset
        
        with open(output_path, "w") as f:
            for record in records:
                raw_text = record.get("synthetic_healthcare_case", "{}")
                clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                try:
                    json_data = json.loads(clean_text)
                    f.write(json.dumps(json_data) + "\n")
                except json.JSONDecodeError:
                    print("Warning: Skipping a record due to malformed JSON output from LLM.")
                
        print(f"✅ Successfully saved REAL generated data to {output_path}")
        return output_path