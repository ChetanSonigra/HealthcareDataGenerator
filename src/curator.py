import json
import logging
import requests
import io
from pathlib import Path
from pypdf import PdfReader
import nemo_curator as nc
from nemo_curator.stages.text.download import DocumentDownloader #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManifestPDFDownloader(DocumentDownloader):
    """
    Custom implementation of NeMo Curator's DocumentDownloader.
    Connects to and downloads data from remote repositories.
    """
    def __init__(self, download_dir: str):
        super().__init__(download_dir=download_dir) #

    def _get_output_filename(self, url: str) -> str:
        # Override this method to provide custom logic for extracting the filename
        return url.split("/")[-1].replace(".pdf", "") + ".jsonl"

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:
        # Override this method to provide custom download logic
        try:
            # 1. Download the file
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # 2. Extract text from the PDF bytes natively in memory
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            extracted_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
            
            # 3. Save as JSONL (Required format for NeMo Curator DocumentDataset)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json.dumps({"url": url, "text": extracted_text}) + '\n')
                
            # Return tuple indicating (success_bool, error_message)
            return True, None 
            
        except Exception as e:
            return False, str(e) #

class DocumentCurator:
    def __init__(self, manifest_path: str, output_dir: str):
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self):
        # Load the manifest metadata[cite: 2]
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Extract URLs from the manifest file[cite: 2]
        urls = [source['url'] for source in manifest.get("sources", []) if 'url' in source] 
        
        logger.info(f"Downloading {len(urls)} documents using NeMo Curator DocumentDownloader...")
        
        # Instantiate the custom DocumentDownloader and execute
        downloader = ManifestPDFDownloader(download_dir=str(self.output_dir))
        
        # The base class provides an underlying download() function that tries to be idempotent
        downloader.download(urls) 
        
        # Initialize a NeMo Curator dataset from the newly written JSONL files
        dataset_path = str(self.output_dir / "*.jsonl")
        dataset = nc.DocumentDataset.read_json(dataset_path)
        
        return dataset

if __name__ == "__main__":
    curator = DocumentCurator("data/raw/manifest.json", "data/raw/downloads/")
    dataset = curator.execute()