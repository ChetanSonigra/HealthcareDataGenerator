import yaml
from src.curator import NeMoDataCurator
from src.data_designer import DataDesigner
from src.synthesizer import Synthesizer
from src.safe_synthesizer import SafeSynthesizer
from src.lora_finetune import LoRACustomizer

def load_config(config_path="config/pipeline_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Note: Ray initialization is now natively handled by the NeMo Curator 
    # core client inside the process_data() block.
    
    # Step 1 & 2: Curate Data (Quality, PII, and Deduplication via 1.0+ Pipeline API)
    curator = NeMoDataCurator(config)
    curated_data_path = curator.process_data()

    # Step 3.1: Generate prompt
    designer = DataDesigner(config)
    optimal_prompt = designer.generate_prompt()

    # Step 3.2: Synthesize new data
    synthesizer = Synthesizer(config)
    raw_synthetic_data_path = synthesizer.generate_synthetic_data(optimal_prompt)

    # Step 4: Run Safe Synthesizer validation job
    safe_synth = SafeSynthesizer(config)
    is_safe = safe_synth.run_safety_checks(raw_synthetic_data_path)

    # Step 5 & 6: Fine-tune and evaluate via LoRA
    if is_safe:
        customizer = LoRACustomizer(config)
        customizer.finetune_and_evaluate(curated_data_path)
    else:
        print("Pipeline stopped: Generated data failed safety checks. Optimization required.")

    print("--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()