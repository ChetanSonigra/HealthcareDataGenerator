import os
from loguru import logger
from nemo_curator.core.client import RayClient
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow

class DataDesignerProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def execute_workflows(self) -> str:
        logger.info("Starting Ray-based Data Designer workflows...")
        
        # Initialize Ray Client cluster context
        ray_client = RayClient()
        ray_client.start()
        
        results_dir = os.path.join(self.output_dir, "dedup_results")
        deduped_output_dir = os.path.join(self.output_dir, "deduplicated")
        
        # 1. Execute Exact Deduplication Workflow via Ray
        logger.info("Running Exact Deduplication Workflow...")
        exact_workflow = ExactDeduplicationWorkflow(
            input_path=os.path.join(self.input_dir, "*.jsonl"),
            output_path=results_dir,
            text_field="text",
            assign_id=True,
            perform_removal=False,
            input_filetype="jsonl"
        )
        exact_result = exact_workflow.run()
        logger.info(f"Exact deduplication metadata: {exact_result.metadata}")
        
        # 2. Remove identified duplicates using TextDuplicatesRemovalWorkflow
        logger.info("Executing Text Duplicates Removal Workflow...")
        removal_workflow = TextDuplicatesRemovalWorkflow(
            input_path=os.path.join(self.input_dir, "*.jsonl"),
            ids_to_remove_path=os.path.join(results_dir, "ExactDuplicateIds"),
            output_path=deduped_output_dir,
            input_filetype="jsonl",
            input_id_field="_curator_dedup_id",
            ids_to_remove_duplicate_id_field="_curator_dedup_id",
            id_generator_path=os.path.join(results_dir, "exact_id_generator.json")
        )
        removal_result = removal_workflow.run()
        logger.info(f"Removal complete. Clean corpus generated at {deduped_output_dir}")
        
        return deduped_output_dir