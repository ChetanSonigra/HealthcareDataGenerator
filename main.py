import yaml
import ray
from curator import NeMoDataCurator
from data_designer import DataDesigner
from synthesizer import Synthesizer
from safe_synthesizer import SafeSynthesizer
from lora_finetune import LoRACustomizer

def load_config(config_path="config/pipeline_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    # Initialize Ray modern approach
    ray_address = config['ray'].get('address')
    if ray_address == 'auto':
        try:
            ray.init(address='auto')
        except ConnectionError:
            ray.init() # Local cluster fallback
    else:
        ray.init(num_cpus=config['ray']['num_cpus'])
        
    print(f"Ray initialized successfully. Dashboard: {ray.available_resources()}")

    # Step 1 & 2: Curate Data based on Curator documentation
    curator = NeMoDataCurator(config)
    curated_data_path = curator.process_data()

    # Step 3.1: Generate prompt using Data Designer logic
    designer = DataDesigner(config)
    optimal_prompt = designer.generate_prompt()

    # Step 3.2: Synthesize new data
    synthesizer = Synthesizer(config)
    raw_synthetic_data_path = synthesizer.generate_synthetic_data(optimal_prompt)

    # Step 4: Run Safe Synthesizer job
    safe_synth = SafeSynthesizer(config)
    is_safe = safe_synth.run_safety_checks(raw_synthetic_data_path)

    # Step 5 & 6: Fine-tune and evaluate via LoRA
    if is_safe:
        customizer = LoRACustomizer(config)
        customizer.finetune_and_evaluate(curated_data_path)
    else:
        print("Pipeline stopped: Generated data failed safety checks. Optimization required.")

    ray.shutdown()
    print("--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()