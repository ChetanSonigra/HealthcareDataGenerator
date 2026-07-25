import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import nemo_curator as nc
from nemo_curator.download import download_urls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentCurator:
    def __init__(self, manifest_path: str, output_dir: str):
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> List[Dict[str, Any]]:
        """Loads the metadata defining public Humana and US government PDFs[cite: 2]."""
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
        return manifest.get("sources", [])[cite: 2]

    def download_and_extract(self, sources: List[Dict[str, Any]]):
        urls = [source['url'] for source in sources if 'url' in source][cite: 2]
        logger.info(f"Downloading {len(urls)} documents...")
        
        # Download pipeline using NeMo Curator
        downloaded_docs = download_urls(urls, str(self.output_dir))
        return downloaded_docs

    def execute(self):
        sources = self.load_manifest()
        raw_data = self.download_and_extract(sources)
        
        # Initialize a Nemo Curator dataset
        dataset = nc.DocumentDataset.read_json(str(self.output_dir / "*.json"))
        return dataset

if __name__ == "__main__":
    curator = DocumentCurator("data/manifest.json", "data/raw/")
    dataset = curator.execute()