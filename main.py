import os
import yaml
from src.curator import NeMoDataCurator
from src.data_designer import DataDesigner
from src.synthesizer import Synthesizer
from src.safe_synthesizer import SafeSynthesizer
from src.lora_finetune import LoRACustomizer

def load_config(config_path="configs/pipeline_config.yaml"):
    # Get the absolute directory where main.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_full = os.path.join(base_dir, config_path)
    
    with open(config_full, "r") as f:
        config = yaml.safe_load(f)
        
    # Force all paths to be absolute based on main.py's location to avoid HPC relative path issues
    pipeline = config['pipeline']
    pipeline['data_dir'] = os.path.abspath(os.path.join(base_dir, pipeline.get('data_dir', 'data')))
    pipeline['manifest_path'] = os.path.abspath(os.path.join(base_dir, pipeline.get('manifest_path', 'data/manifest.json')))
    pipeline['generated_data_path'] = os.path.abspath(os.path.join(base_dir, pipeline.get('generated_data_path', 'data/generated_data.json')))
    pipeline['output_dir'] = os.path.abspath(os.path.join(base_dir, pipeline.get('output_dir', 'output')))
    
    # 🚨 HARD CHECK: Verify the file exists before we start Ray or Curator 🚨
    if not os.path.exists(pipeline['generated_data_path']):
        raise FileNotFoundError(
            f"\n\n🛑 [CRITICAL ERROR] Missing File! 🛑\n"
            f"Python is looking for your data file here:\n"
            f"--> {pipeline['generated_data_path']}\n\n"
            f"Please check your file explorer or terminal and ensure the file is saved EXACTLY there.\n"
        )
        
    return config

def main():
    config = load_config()
    
    # Step 1 & 2: Curate Data (Bypassing unsupported HPC dependencies)
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
    print("working")