import os
import yaml
from src.curator import NeMoDataCurator
from src.data_designer import DataDesigner
from src.synthesizer import Synthesizer
from src.safe_synthesizer import SafeSynthesizer
from src.lora_finetune import LoRACustomizer

def load_config(config_path="config/pipeline_config.yaml"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_full = os.path.join(base_dir, config_path)
    
    with open(config_full, "r") as f:
        config = yaml.safe_load(f)
        
    pipeline = config['pipeline']
    pipeline['data_dir'] = os.path.abspath(os.path.join(base_dir, pipeline.get('data_dir', 'data')))
    pipeline['manifest_path'] = os.path.abspath(os.path.join(base_dir, pipeline.get('manifest_path', 'data/manifest.json')))
    pipeline['output_dir'] = os.path.abspath(os.path.join(base_dir, pipeline.get('output_dir', 'output')))
    
    # Check that the manifest grounding data exists
    if not os.path.exists(pipeline['manifest_path']):
        raise FileNotFoundError(f"Missing manifest file at {pipeline['manifest_path']}")
        
    return config

def main():
    config = load_config()
    
    print("--- Initiating Zero-Shot Synthetic Data Pipeline ---")

    # Step 1: Generate prompt (Using only grounding context)
    designer = DataDesigner(config)
    optimal_prompt = designer.generate_prompt()

    # Step 2: Synthesize NEW data from scratch
    synthesizer = Synthesizer(config)
    raw_synthetic_data_path = synthesizer.generate_synthetic_data(optimal_prompt)

    # Tell Curator to use the newly generated data instead of the old file
    config['pipeline']['generated_data_path'] = raw_synthetic_data_path

    # Step 3: Curate the NEWLY generated data
    curator = NeMoDataCurator(config)
    curated_data_path = curator.process_data()

    # Step 4: Run Safe Synthesizer validation job on the curated data
    safe_synth = SafeSynthesizer(config)
    is_safe = safe_synth.run_safety_checks(curated_data_path)

    # Step 5 & 6: Fine-tune and evaluate via LoRA
    if is_safe:
        customizer = LoRACustomizer(config)
        customizer.finetune_and_evaluate(curated_data_path)
    else:
        print("Pipeline stopped: Generated data failed safety checks. Optimization required.")

    print("--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()