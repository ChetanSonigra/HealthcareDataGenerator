import os
import nemo_curator
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import WordCountFilter
from nemo_curator.modules import ExactDuplicates, FuzzyDuplicates, SemanticDuplicates
from nemo_curator.modifiers.pii_modifier import PiiModifier

class NeMoDataCurator:
    def __init__(self, config):
        self.config = config
        self.output_dir = os.path.join(self.config['pipeline']['output_dir'], "curated")
        os.makedirs(self.output_dir, exist_ok=True)

    def process_data(self):
        print("--- Starting NeMo Curator Process ---")
        # Load dataset
        dataset = DocumentDataset.read_json(
            self.config['pipeline']['generated_data_path'],
            backend="ray"
        )
        
        # 1. Quality Assessment (Basic word count filter as example)
        print("Applying Quality Assessment Filters...")
        dataset = dataset.filter(WordCountFilter(min_words=5))

        # 2. PII Filtration
        print("Applying PII Filtration...")
        pii_modifier = PiiModifier(
            supported_entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION"]
        )
        dataset = dataset.modify(pii_modifier)

        # 3. Deduplication (Exact, Fuzzy, Semantic)
        print("Running Deduplication...")
        exact_dedup = ExactDuplicates(text_field="content")
        dataset = exact_dedup(dataset)
        
        fuzzy_dedup = FuzzyDuplicates(text_field="content", seed=42)
        dataset = fuzzy_dedup(dataset)

        # Note: Semantic deduplication requires an embedding model initialization
        # semantic_dedup = SemanticDuplicates(text_field="content", embedding_model="...")
        # dataset = semantic_dedup(dataset)
        
        # Save curated data
        output_path = os.path.join(self.output_dir, "curated_data.jsonl")
        dataset.to_json(output_path, write_empty=False)
        print(f"Curated data saved to {output_path}")
        return output_path