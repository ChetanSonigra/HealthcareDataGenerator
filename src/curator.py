import io
import json
import logging
import requests
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader
from nemo_curator.core.client import RayClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentCurator:
    def __init__(self, manifest_path: str, output_dir: str):
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Ray Client for distributed execution
        self.ray_client = RayClient()
        self.ray_client.start()

    def load_manifest(self) -> List[Dict[str, Any]]:
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
        return manifest.get("sources", [])

    def download_and_extract(self, sources: List[Dict[str, Any]]) -> str:
        download_dir = self.output_dir / "jsonl_inputs"
        download_dir.mkdir(exist_ok=True)
        
        for source in sources:
            url = source.get('url')
            doc_id = source.get('id', 'document')
            if not url:
                continue
                
            try:
                logger.info(f"Downloading and extracting source: {doc_id}")
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                pdf_file = io.BytesIO(response.content)
                reader = PdfReader(pdf_file)
                extracted_text = "".join([page.extract_text() + "\n" for page in reader.pages if page.extract_text()])
                
                file_path = download_dir / f"{doc_id}.jsonl"
                doc_data = {
                    "id": doc_id,
                    "text": extracted_text,
                    "url": url,
                    "category": source.get("category", "unknown")
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(doc_data) + '\n')
            except Exception as e:
                logger.error(f"Failed to process {doc_id}: {str(e)}")
                
        return str(download_dir)

    def execute(self) -> str:
        sources = self.load_manifest()
        output_jsonl_dir = self.download_and_extract(sources)
        logger.info(f"Curation inputs successfully staged at {output_jsonl_dir}")
        return output_jsonl_dir