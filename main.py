import os
import sys
import yaml
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from curator import DocumentCurator
from data_designer import DataDesignerProcessor
from synthesizer import DataSynthesizer
from safe_synthesizer import SafeSynthesizerValidator
from lora_finetune import LoRACustomizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RayOrchestrator")

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    logger.info("Initializing Ray-Powered NVIDIA Synthetic Data Pipeline...")

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.error("NVIDIA_API_KEY environment variable not set. Exiting.")
        sys.exit(1)

    config = load_config("configs/pipeline_config.yaml")
    
    manifest_path = "data/raw/manifest.json"
    raw_output_dir = "data/raw/"
    processed_output_dir = "data/processed/"
    
    try:
        # Step 1: Curation & Staging
        logger.info("--- Step 1: Curation via Ray ---")
        curator = DocumentCurator(manifest_path, raw_output_dir)
        staged_jsonl_dir = curator.execute()
        
        # Step 2: Data Designer / Ray Workflows
        logger.info("--- Step 2: Ray-based Deduplication Workflows ---")
        processor = DataDesignerProcessor(staged_jsonl_dir, processed_output_dir)
        clean_data_dir = processor.execute_workflows()
        logger.info(f"Clean curated corpus ready at {clean_data_dir}")

        # Step 3: Synthesizer
        logger.info("--- Step 3: Synthetic Data Generation ---")
        synthesizer = DataSynthesizer(
            api_key=api_key, 
            endpoint_url=config['api_settings']['chat_endpoint']
        )
        prompt = synthesizer.generate_prompt(
            case_id="SYNCASE000001", state="AZ", plan_type="Medicare Advantage", topic="benefit or eligibility"
        )
        synthetic_conversation = synthesizer.generate_synthetic_data(prompt)
        logger.info("Synthetic conversation successfully generated.")

        # Step 4: Safe Synthesizer Evaluation
        logger.info("--- Step 4: Safety Validation ---")
        validator = SafeSynthesizerValidator(
            nvcf_url=config['api_settings']['nvcf_safety_url'], 
            api_key=api_key
        )
        is_safe = validator.run_safety_job(synthetic_conversation)
        if not is_safe:
            logger.warning("Generated data failed safety evaluation. Halting execution.")
            sys.exit(1)
        logger.info("Safety validation passed.")

        # Step 5 & 6: LoRA Fine-Tuning
        logger.info("--- Step 5 & 6: LoRA Customization ---")
        customizer = LoRACustomizer(
            api_endpoint=config['api_settings']['customizer_url'], 
            api_key=api_key
        )
        sample_file = list(Path(clean_data_dir).glob("*.jsonl"))[0]
        dataset_id = customizer.upload_dataset(str(sample_file))
        
        job_id = customizer.start_finetune_job(
            file_id=dataset_id, 
            base_model=config['lora_finetuning']['base_model']
        )
        final_metrics = customizer.monitor_job(job_id)
        logger.info(f"Pipeline Execution Complete! Final Metrics: {final_metrics}")

    except Exception as e:
        logger.error(f"Ray pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()