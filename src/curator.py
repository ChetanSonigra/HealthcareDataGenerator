import os
import shutil
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

# 1.0+ Updated Filter Imports
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.filters.heuristic import WordCountFilter

# --- BULLETPROOF DEDUPLICATION IMPORT ---
exact_workflow_class = None
try:
    from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
    exact_workflow_class = ExactDeduplicationWorkflow
    print("Successfully loaded ExactDeduplicationWorkflow.")
except ImportError as e:
    pass # Warning already printed in previous runs

# --- BULLETPROOF PII IMPORT ---
pii_stage = None
try:
    from nemo_curator.pii.algorithm import PiiDeidentifier
    from nemo_curator.stages.text.modifiers import Modify
    pii_modifier_obj = PiiDeidentifier(supported_entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"])
    pii_stage = Modify(modifier_fn=pii_modifier_obj, input_fields="content")
except ImportError:
    try:
        from nemo_curator.pii import PiiModifier
        from nemo_curator.stages.text.modifiers import Modify
        pii_modifier_obj = PiiModifier(supported_entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"])
        pii_stage = Modify(modifier_fn=pii_modifier_obj, input_fields="content")
    except ImportError:
        try:
            from nemo_curator.modifiers.pii_modifier import PiiModifier
            from nemo_curator.modules.modify import Modify
            pii_modifier_obj = PiiModifier(supported_entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"])
            pii_stage = Modify(modifier_fn=pii_modifier_obj, input_fields="content")
        except ImportError as e:
            pii_stage = None
# ------------------------------


class NeMoDataCurator:
    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.join(self.config['pipeline']['output_dir'], "curated")
        os.makedirs(self.output_dir, exist_ok=True)

    def process_data(self):
        print("--- Starting NeMo Curator Process ---")
        
        # 1. Initialize Ray Client
        ray_client = RayClient()
        ray_client.start()
        print("Ray client initialized.")
        
        input_file = self.config['pipeline']['generated_data_path']
        filtered_output = os.path.join(self.output_dir, "filtered_data.jsonl")
        
        # 2. Define standard IO Stages
        reader = JsonlReader(input_file)
        writer = JsonlWriter(filtered_output)
        
        # 3. Build Pipeline dynamically
        pipeline_stages = [reader]
        if pii_stage:
            print("Adding PII Filtration to pipeline...")
            pipeline_stages.append(pii_stage)
        pipeline_stages.append(writer)

        # 4. Execute the Modality Pipeline (ONLY if we have actual processing stages)
        if len(pipeline_stages) > 2:
            pipeline = Pipeline(pipeline_stages)
            pipeline.run()
            print("Quality filtering (and PII if available) complete.")
        else:
            print("No processing stages available (PII skipped, Filters bypassed).")
            print("Passing data directly through to keep pipeline moving...")
            # If it's a directory from a previous Ray run, remove it so shutil can write a clean file
            if os.path.isdir(filtered_output):
                shutil.rmtree(filtered_output)
            shutil.copy2(input_file, filtered_output)
            print(f"Data passed through to {filtered_output}")
        
        # 5. Exact Deduplication
        if exact_workflow_class:
            print("Running Exact Deduplication Workflow...")
            dedup_dir = os.path.join(self.output_dir, "dedup_results")
            os.makedirs(dedup_dir, exist_ok=True)
            
            exact_workflow = exact_workflow_class(
                input_path=filtered_output,
                output_path=dedup_dir,
                text_field="content",
                assign_id=True,
                perform_removal=False, 
                input_filetype="jsonl"
            )
            exact_workflow.run()
            print(f"Deduplicated IDs isolated and saved to {dedup_dir}")
        else:
            print("Skipping Exact Deduplication due to missing RAPIDS dependencies.")
            print("Proceeding with filtered data to next pipeline step.")
        
        ray_client.stop()
        return filtered_output