import os
import sys
import yaml
import logging
from pathlib import Path

# Add src to the Python path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from src._1_curator import DocumentCurator
from src._2_data_designer import DataDesignerProcessor
from src._3_synthesizer import DataSynthesizer
from src._4_safe_synthesizer import SafeSynthesizerValidator
from src._5_lora_finetune import LoRACustomizer

# Configure logging for the pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PipelineOrchestrator")

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    logger.info("Initializing NVIDIA Synthetic Data Pipeline...")

    # 1. Setup & Credentials
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.error("NVIDIA_API_KEY environment variable not set. Exiting.")
        sys.exit(1)

    config = load_config("configs/pipeline_config.yaml")
    
    # Define file paths
    manifest_path = "data/raw/manifest.json"
    raw_output_dir = "data/raw/"
    processed_data_path = "data/processed/clean_corpus.json"
    
    try:
        # Step 1: Data Curation
        logger.info("--- Step 1: Curation ---")
        curator = DocumentCurator(manifest_path, raw_output_dir)
        dataset = curator.execute()
        
        # Step 2: Data Designer (Processing & Filtration)
        logger.info("--- Step 2: Data Designer ---")
        processor = DataDesignerProcessor(dataset)
        clean_data = processor.execute_pipeline(processed_data_path)
        logger.info(f"Data cleaned and saved to {processed_data_path}")

        # Step 3: Synthesizer
        logger.info("--- Step 3: Synthetic Data Generation ---")
        synthesizer = DataSynthesizer(
            api_key=api_key, 
            endpoint_url=config['api_settings']['chat_endpoint']
        )
        
        # Example generation using parameters from your provided data
        prompt = synthesizer.generate_prompt(
            case_id="SYNCASE000001", 
            state="AZ", 
            plan_type="Medicare Advantage", 
            topic="benefit or eligibility"
        )
        synthetic_conversation = synthesizer.generate_synthetic_data(prompt)
        logger.info("Synthetic conversation generated successfully.")

        # Step 4: Safe Synthesizer Evaluation
        logger.info("--- Step 4: Safety Validation ---")
        validator = SafeSynthesizerValidator(
            nvcf_url=config['api_settings']['nvcf_safety_url'], 
            api_key=api_key
        )
        is_safe = validator.run_safety_job(synthetic_conversation)
        
        if not is_safe:
            logger.warning("Generated data failed safety checks. Aborting fine-tuning pipeline.")
            sys.exit(1)
        
        logger.info("Data passed safety validation. Proceeding to Customizer.")

        # Step 5 & 6: LoRA Fine-Tuning and Evaluation
        logger.info("--- Step 5 & 6: LoRA Fine-Tuning & Optimization ---")
        customizer = LoRACustomizer(
            api_endpoint=config['api_settings']['customizer_url'], 
            api_key=api_key
        )
        
        # Upload the processed/synthetic dataset
        logger.info("Uploading dataset...")
        dataset_id = customizer.upload_dataset(processed_data_path)
        
        # Launch training
        logger.info(f"Starting fine-tuning job with base model {config['lora_finetuning']['base_model']}...")
        job_id = customizer.start_finetune_job(
            file_id=dataset_id, 
            base_model=config['lora_finetuning']['base_model']
        )
        
        # Monitor until complete
        final_metrics = customizer.monitor_job(job_id)
        logger.info(f"Pipeline Complete! Final Training Metrics: {final_metrics}")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()