import asyncio
import json
import logging
import random
import urllib.parse
import urllib.request
import requests
from typing import Optional, Dict, Any, Union, List

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
from open_webui.env import SRC_LOG_LEVELS
from pydantic import BaseModel

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SWARMUI"] if "SWARMUI" in SRC_LOG_LEVELS else logging.INFO)

default_headers = {"Content-Type": "application/json"}


def queue_prompt(prompt, client_id, base_url, auth_header):
    log.info("queue_prompt")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    log.debug(f"queue_prompt data: {data}")
    try:
        headers = {**default_headers}
        if auth_header:
            headers["Authorization"] = auth_header
            
        req = urllib.request.Request(
            f"{base_url}/prompt",
            data=data,
            headers=headers,
        )
        response = urllib.request.urlopen(req).read()
        return json.loads(response)
    except Exception as e:
        log.exception(f"Error while queuing prompt: {e}")
        raise e


def get_image(filename, subfolder, folder_type, base_url, auth_header):
    log.info("get_image")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    
    headers = {**default_headers}
    if auth_header:
        headers["Authorization"] = auth_header
        
    req = urllib.request.Request(
        f"{base_url}/view?{url_values}",
        headers=headers,
    )
    with urllib.request.urlopen(req) as response:
        return response.read()


def get_image_url(filename, subfolder, folder_type, base_url):
    log.info("get_image")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    return f"{base_url}/view?{url_values}"


def get_history(prompt_id, base_url, auth_header):
    log.info("get_history")
    
    headers = {**default_headers}
    if auth_header:
        headers["Authorization"] = auth_header
        
    req = urllib.request.Request(
        f"{base_url}/history/{prompt_id}",
        headers=headers,
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def get_images(ws, prompt, client_id, base_url, auth_header):
    log.info("get_images")
    # This is a placeholder for the SwarmUI websocket/image retrieval logic
    return []


class SwarmUIGenerateImageForm(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int
    height: int
    n: int = 1
    steps: Optional[int] = None
    seed: Optional[int] = None
    model: Optional[str] = None


def get_swarmui_session_id(base_url: str, auth_header: str) -> str:
    """Obtain a SwarmUI session ID by calling the correct session API."""
    headers = {**default_headers}
    if auth_header:
        headers["Authorization"] = auth_header
    
    try:
        log.info(f"Getting SwarmUI session ID from {base_url}/API/GetNewSession")
        log.debug(f"Headers: {headers}")
        
        resp = requests.post(
            f"{base_url}/API/GetNewSession",
            headers=headers,
            json={},
            timeout=20
        )
        resp.raise_for_status()
        data = resp.json()
        session_id = data.get("session_id")
        log.info(f"Obtained SwarmUI session ID: {session_id}")
        return session_id
    except Exception as e:
        log.error(f"Failed to get SwarmUI session id: {e}")
        return None


def swarmui_generate_image(
    model: str,
    payload: Union[SwarmUIGenerateImageForm, Dict[str, Any]],
    client_id: str,
    base_url: str,
    auth_header: str
) -> Dict[str, Any]:
    """
    Calls SwarmUI's HTTP API to generate images from text prompt.
    First gets a session ID, then calls the generation API.
    """
    log.info(f"swarmui_generate_image: model={model}, client_id={client_id}")
    log.debug(f"Payload: {payload}")
    
    headers = {**default_headers}
    if auth_header:
        headers["Authorization"] = auth_header
    
    # Get session id
    session_id = get_swarmui_session_id(base_url, auth_header)
    if not session_id:
        return {"error": "Could not obtain SwarmUI session id"}
    
    # Prepare parameters for generation
    try:
        # Handle payload as either a dict or a SwarmUIGenerateImageForm
        if isinstance(payload, dict):
            # It's already a dictionary
            prompt = payload.get("prompt", "")
            negative_prompt = payload.get("negative_prompt", "")
            width = payload.get("width", 512)
            height = payload.get("height", 512)
            n_images = payload.get("n", 1)
            steps = payload.get("steps", 20)
            seed = payload.get("seed", None)
        else:
            # It's a SwarmUIGenerateImageForm
            prompt = payload.prompt
            negative_prompt = payload.negative_prompt if hasattr(payload, "negative_prompt") else None
            width = payload.width
            height = payload.height
            n_images = payload.n if hasattr(payload, "n") else 1
            steps = payload.steps if hasattr(payload, "steps") else 20
            seed = payload.seed if hasattr(payload, "seed") else None
        
        # Create the raw input for the API call
        raw_input = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "images": n_images,
            "session_id": session_id,
        }
        
        if negative_prompt:
            raw_input["negative_prompt"] = negative_prompt
        if seed is not None:
            raw_input["seed"] = seed
        if model:
            raw_input["model"] = model
        
        # Prepare final data payload
        data = raw_input
        
        log.info(f"Sending image generation request to SwarmUI: {base_url}/API/GenerateText2Image")
        log.debug(f"Request data: {data}")
        log.debug(f"Headers: {headers}")
        
        resp = requests.post(
            f"{base_url}/API/GenerateText2Image",
            headers=headers,
            json=data,
            timeout=120
        )
        resp.raise_for_status()
        result = resp.json()
        log.info(f"SwarmUI image generation response: {result}")

        # Handle invalid session ID case
        if "error_id" in result and result["error_id"] == "invalid_session_id":
            # Try again with a new session
            log.warning("Invalid session ID, getting a new one")
            session_id = get_swarmui_session_id(base_url, auth_header)
            if not session_id:
                return {"error": "Could not obtain SwarmUI session id (retry)"}
            data["session_id"] = session_id
            resp = requests.post(
                f"{base_url}/API/GenerateText2Image",
                headers=headers,
                json=data,
                timeout=120
            )
            resp.raise_for_status()
            result = resp.json()

        return result
    except Exception as e:
        log.exception(f"Error generating image with SwarmUI: {e}")
        return {"error": str(e)}


def list_swarmui_models(base_url: str, auth_header: str, path: str = "", depth: int = 1, subtype: str = "Stable-Diffusion", sortBy: str = "Name", allowRemote: bool = True, sortReverse: bool = False):
    """
    Calls SwarmUI's /API/ListModels endpoint to retrieve available models.
    First obtains a session ID, then calls the models API.
    """
    headers = {**default_headers}
    if auth_header:
        headers["Authorization"] = auth_header
    
    # First get a session ID
    session_id = get_swarmui_session_id(base_url, auth_header)
    if not session_id:
        log.error("Failed to obtain SwarmUI session ID")
        return []
        
    # Now call the ListModels API with the session ID
    url = f"{base_url}/API/ListModels"
    payload = {
        "path": path,
        "depth": depth,
        "subtype": subtype,
        "sortBy": sortBy,
        "allowRemote": allowRemote,
        "sortReverse": sortReverse,
        "session_id": session_id
    }
    
    try:
        log.info(f"Requesting SwarmUI models: {url}")
        log.debug(f"Payload: {payload}")
        log.debug(f"Headers: {headers}")
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        log.debug(f"SwarmUI models response: {data}")
        
        # If we got an empty list or no 'files' key, try a different approach
        if not data.get("files", []):
            log.warning("No models found in the initial response, checking for fallback method")
            
            # Some SwarmUI instances might have models directly in the response
            models_fallback = []
            
            # Try to parse models from the response in different possible formats
            if "models" in data:
                models_fallback = data.get("models", [])
                log.info(f"Found {len(models_fallback)} models in 'models' key")
            elif isinstance(data, list):
                models_fallback = data
                log.info(f"Response was a direct list with {len(models_fallback)} models")
            
            # If we found models in a fallback method, return them
            if models_fallback:
                return models_fallback
        
        # Return a flat list of model files with their metadata
        return data.get("files", [])
    except Exception as e:
        log.exception(f"Error fetching SwarmUI models: {e}")
        return []


def select_swarmui_model(base_url: str, auth_header: str, model_path: str, backendId: str = None):
    """
    Calls SwarmUI's /API/SelectModel endpoint to load the selected model.
    First obtains a session ID, then calls the select model API.
    """
    headers = {**default_headers}
    if auth_header:
        headers["Authorization"] = auth_header
    
    # First get a session ID
    session_id = get_swarmui_session_id(base_url, auth_header)
    if not session_id:
        log.error("Failed to obtain SwarmUI session ID")
        return False
        
    url = f"{base_url}/API/SelectModel"
    payload = {
        "model": model_path,
        "backendId": backendId,
        "session_id": session_id
    }
    
    try:
        log.info(f"Selecting SwarmUI model: {url}")
        log.debug(f"Payload: {payload}")
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("success", False)
    except Exception as e:
        log.exception(f"Error selecting SwarmUI model: {e}")
        return False
