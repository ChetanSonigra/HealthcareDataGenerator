import os
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

# 1.0+ Updated Filter Imports
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.filters.heuristic import WordCountFilter

from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

# --- BULLETPROOF PII IMPORT ---
pii_stage = None
try:
    # 1. Try the absolute newest 1.2.0+ API (Renamed to PiiDeidentifier)
    from nemo_curator.pii.algorithm import PiiDeidentifier
    from nemo_curator.stages.text.modifiers import Modify
    print("Successfully loaded PiiDeidentifier from newest API.")
    pii_modifier_obj = PiiDeidentifier(supported_entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"])
    pii_stage = Modify(modifier_fn=pii_modifier_obj, input_fields="content")

except ImportError:
    try:
        # 2. Try the 1.0 API
        from nemo_curator.pii import PiiModifier
        from nemo_curator.stages.text.modifiers import Modify
        print("Successfully loaded PiiModifier from 1.0 API.")
        pii_modifier_obj = PiiModifier(supported_entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"])
        pii_stage = Modify(modifier_fn=pii_modifier_obj, input_fields="content")

    except ImportError:
        try:
            # 3. Try the legacy 0.x API
            from nemo_curator.modifiers.pii_modifier import PiiModifier
            from nemo_curator.modules.modify import Modify
            print("Successfully loaded PiiModifier from 0.x API.")
            pii_modifier_obj = PiiModifier(supported_entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"])
            pii_stage = Modify(modifier_fn=pii_modifier_obj, input_fields="content")
            
        except ImportError as e:
            print(f"WARNING: Skipping PII Filtration. Could not locate PII modules in this NeMo Curator version. Error: {e}")
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
        
        # 3. Define Processing Stages for Pipeline
        print("Applying Quality Assessment Filters...")
        word_count_filter = ScoreFilter(
            filter_obj=WordCountFilter(min_words=5),
            text_field="content" 
        )
        
        # Build Pipeline dynamically based on whether PII loaded successfully
        pipeline_stages = [reader, word_count_filter]
        if pii_stage:
            print("Adding PII Filtration to pipeline...")
            pipeline_stages.append(pii_stage)
        pipeline_stages.append(writer)

        # 4. Execute the Modality Pipeline
        pipeline = Pipeline(pipeline_stages)
        pipeline.run()
        print("Quality filtering (and PII if available) complete.")
        
        # 5. Exact Deduplication
        print("Running Exact Deduplication Workflow...")
        dedup_dir = os.path.join(self.output_dir, "dedup_results")
        os.makedirs(dedup_dir, exist_ok=True)
        
        exact_workflow = ExactDeduplicationWorkflow(
            input_path=filtered_output,
            output_path=dedup_dir,
            text_field="content",
            assign_id=True,
            perform_removal=False, 
            input_filetype="jsonl"
        )
        exact_workflow.run()
        print(f"Deduplicated IDs isolated and saved to {dedup_dir}")
        
        ray_client.stop()
        return filtered_output