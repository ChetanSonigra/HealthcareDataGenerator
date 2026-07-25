import os
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

# 1.0+ Updated Filter Imports
from nemo_curator.stages.text.filters import ScoreFilter
from nemo_curator.stages.text.filters.heuristic import WordCountFilter

# 1.0+ Updated Modifier Imports
from nemo_curator.stages.text.modifiers import Modify
from nemo_curator.modifiers.pii_modifier import PiiModifier

from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow


class NeMoDataCurator:
    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.join(self.config['pipeline']['output_dir'], "curated")
        os.makedirs(self.output_dir, exist_ok=True)

    def process_data(self):
        print("--- Starting NeMo Curator Process (1.0+ API) ---")
        
        # 1. Initialize Ray Client specifically for Curator
        ray_client = RayClient()
        ray_client.start()
        print("Ray client initialized.")
        
        input_file = self.config['pipeline']['generated_data_path']
        filtered_output = os.path.join(self.output_dir, "filtered_data.jsonl")
        
        # 2. Define standard IO Stages
        reader = JsonlReader(input_file)
        writer = JsonlWriter(filtered_output)
        
        # 3. Define Processing Stages for Pipeline
        print("Applying Quality Assessment and PII Filters...")
        
        # Wraps the filter object using ScoreFilter
        word_count_filter = ScoreFilter(
            filter_obj=WordCountFilter(min_words=5),
            text_field="content" # Target field in your JSONL
        )
        
        # Wraps the modifier object using Modify
        pii_modifier = Modify(
            modifier_fn=PiiModifier(supported_entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"]),
            input_fields="content"
        )
        
        # 4. Build and Execute the Modality Pipeline
        pipeline = Pipeline([
            reader,
            word_count_filter,
            pii_modifier,
            writer
        ])
        pipeline.run()
        print("Quality and PII filtering complete.")
        
        # 5. Exact Deduplication (New separate workflow API)
        print("Running Exact Deduplication Workflow...")
        dedup_dir = os.path.join(self.output_dir, "dedup_results")
        os.makedirs(dedup_dir, exist_ok=True)
        
        exact_workflow = ExactDeduplicationWorkflow(
            input_path=filtered_output,
            output_path=dedup_dir,
            text_field="content",
            assign_id=True,
            perform_removal=False, # Standard pattern: identifies and writes IDs out to output_path
            input_filetype="jsonl"
        )
        exact_workflow.run()
        print(f"Deduplicated IDs isolated and saved to {dedup_dir}")
        
        ray_client.stop()
        return filtered_output