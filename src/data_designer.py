import os
import argparse
from loguru import logger

# --- Correct NeMo Curator Imports ---
from nemo_curator import Sequential, ExactDuplicates
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers import PiiModifier
from nemo_curator.filters import WordCountFilter

def design_data(raw_data_dir: str, output_dir: str):
    logger.info("Initializing Data Designer (NeMo Curator)...")
    
    # 1. Load dataset (DocumentDataset is a wrapper around Dask DataFrames)
    input_path = os.path.join(raw_data_dir, "*.jsonl")
    dataset = DocumentDataset.read_jsonl(input_path)
    
    # 2. Define the sequential curation pipeline
    # Modifiers alter text; Filters drop bad documents
    curation_pipeline = Sequential([
        # Redact PII leveraging the Presidio framework underneath
        PiiModifier(supported_entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]),
        
        # Drop documents that are too short to be useful for pretraining/fine-tuning
        WordCountFilter(min_words=20)
    ])
    
    logger.info("Applying PII Modification and Quality Filters...")
    curated_dataset = curation_pipeline(dataset)
    
    # 3. Exact Deduplication
    # Hashing documents (MD5 by default) to identify and remove exact matches
    logger.info("Running Exact Deduplication...")
    exact_dup = ExactDuplicates(
        id_field="id",
        text_field="text",
        hash_method="md5"
    )
    
    deduped_dataset = exact_dup(curated_dataset)
    
    # 4. Save the designed data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "curated_output.jsonl")
    
    logger.info(f"Writing curated data to {output_path}")
    deduped_dataset.to_jsonl(output_path)
    logger.info("Data Design Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to raw JSONL data")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save curated JSONL data")
    
    args = parser.parse_args()
    design_data(args.raw_data_dir, args.output_dir)