import time
import requests
from typing import Dict

class LoRACustomizer:
    def __init__(self, api_endpoint: str, api_key: str):
        self.api_endpoint = api_endpoint
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def upload_dataset(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.api_endpoint}/v1/files", 
                headers={"Authorization": self.headers["Authorization"]}, 
                files=files
            )
        response.raise_for_status()
        return response.json()['file_id']

    def start_finetune_job(self, file_id: str, base_model: str = "llama-3-70b") -> str:
        payload = {
            "model": base_model,
            "training_file": file_id,
            "hyperparameters": {
                "n_epochs": 3,
                "learning_rate": 2e-4,
                "lora_r": 16,
                "lora_alpha": 32
            }
        }
        
        response = requests.post(f"{self.api_endpoint}/v1/fine_tuning/jobs", headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()['id']

    def monitor_job(self, job_id: str) -> Dict:
        while True:
            response = requests.get(f"{self.api_endpoint}/v1/fine_tuning/jobs/{job_id}", headers=self.headers)
            status = response.json().get("status")
            print(f"Job {job_id} Status: {status}")
            
            if status in ["succeeded", "failed"]:
                return response.json()
            time.sleep(60)

if __name__ == "__main__":
    customizer = LoRACustomizer("https://api.nvidia.com/v1/customization", "YOUR_API_KEY")
    
    # 1. Upload the highly curated synthetic data
    dataset_id = customizer.upload_dataset("data/processed/synthetic_training_data.jsonl")
    
    # 2. Launch LoRA Job
    job_id = customizer.start_finetune_job(dataset_id)
    
    # 3. Evaluate and Optimize (Monitor Loss)
    final_metrics = customizer.monitor_job(job_id)
    print("Optimization Complete. Final Metrics:", final_metrics)