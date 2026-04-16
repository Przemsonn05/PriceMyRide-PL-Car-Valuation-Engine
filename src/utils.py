# src/utils.py - POPRAWIONE

import requests
import os
import joblib
import tempfile
from typing import Dict, Optional, Any
from pathlib import Path
from huggingface_hub import HfApi, login
from typing import Union


def get_current_eur_pln_rate(timeout: int = 5) -> float:
    """
    Fetch current EUR/PLN exchange rate from NBP API.
    
    Parameters
    ----------
    timeout : int, default=5
        Request timeout in seconds
        
    Returns
    -------
    float
        Exchange rate or default 4.30 if API fails
    """
    url = "http://api.nbp.pl/api/exchangerates/rates/a/eur/?format=json"
    
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        rate = data['rates'][0]['mid']
        print(f"[OK] Exchange rate loaded from NBP: {rate:.4f}")
        return rate

    except Exception as e:
        print(f"[WARN] API failed ({e}). Using default rate: 4.30")
        return 4.30


def upload_models_to_hf(
    models_dict: Dict[str, Any],
    repo_id: str,
    token: Optional[str] = None
) -> None:
    """
    Save models and upload to Hugging Face Hub.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary of {filename: model_object}
    repo_id : str
        HF repository ID (e.g., "username/repo-name")
    token : str, optional
        HF token (if None, reads from HF_TOKEN environment variable)
        
    Raises
    ------
    ValueError
        If HF token not found
        
    Examples
    --------
    >>> models = {
    ...     'ridge_model.pkl': ridge_pipeline,
    ...     'xgb_model.pkl': xgb_pipeline
    ... }
    >>> upload_models_to_hf(models, "username/car-price-models")
    """
    hf_token = token or os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError(
            "HF_TOKEN not found. Set it as environment variable or pass as argument."
        )
    
    print("Logging in to Hugging Face...")
    login(hf_token)
    api = HfApi()
    
    print(f"Creating/verifying repository: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmp:
        for filename, model in models_dict.items():
            model_path = os.path.join(tmp, filename)
            
            print(f"Saving {filename}...")
            joblib.dump(model, model_path)
            
            print(f"Uploading {filename} to {repo_id}...")
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"[OK] Successfully uploaded: {filename}")
    
    print(f"\n[OK] All models uploaded to: https://huggingface.co/{repo_id}")


def load_local_model(file_path: Union[str, Path]) -> Any:
    """
    Load model from local file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to model file
        
    Returns
    -------
    Any
        Loaded model object
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    print(f"Loading model from: {file_path}")
    model = joblib.load(file_path)
    print("[OK] Model loaded successfully")
    
    return model