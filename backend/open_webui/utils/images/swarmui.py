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
    """Form for generating images with SwarmUI"""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    n: int = 1
    steps: int = 20
    seed: Optional[int] = None
    cfg_scale: Optional[float] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    preset: Optional[str] = None
    preset_data: Optional[Dict[str, Any]] = None


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
    
    # Clean up model name by removing file extension if present
    if model and "." in model:
        # Remove file extension like .safetensors, .ckpt, etc.
        model = model.split(".")[0]
        log.info(f"Using cleaned model name: {model}")
    
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
            cfg_scale = payload.get("cfg_scale", None)
            sampler = payload.get("sampler", None)
            scheduler = payload.get("scheduler", None)
            preset = payload.get("preset", None)
            preset_data = payload.get("preset_data", None)
        else:
            # It's a SwarmUIGenerateImageForm
            prompt = payload.prompt
            negative_prompt = payload.negative_prompt if hasattr(payload, "negative_prompt") else None
            width = payload.width
            height = payload.height
            n_images = payload.n if hasattr(payload, "n") else 1
            steps = payload.steps if hasattr(payload, "steps") else 20
            seed = payload.seed if hasattr(payload, "seed") else None
            cfg_scale = payload.cfg_scale if hasattr(payload, "cfg_scale") else None
            sampler = payload.sampler if hasattr(payload, "sampler") else None
            scheduler = payload.scheduler if hasattr(payload, "scheduler") else None
            preset = payload.preset if hasattr(payload, "preset") else None
            preset_data = payload.preset_data if hasattr(payload, "preset_data") else None
        
        # Create the raw input for the API call
        raw_input = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "images": n_images,
            "session_id": session_id,
            "do_not_save": True  # Request base64 instead of file paths
        }
        
        if negative_prompt:
            raw_input["negative_prompt"] = negative_prompt
        if seed is not None:
            raw_input["seed"] = seed
        if cfg_scale is not None:
            raw_input["cfg_scale"] = cfg_scale
        if sampler is not None:
            raw_input["sampler"] = sampler
        if scheduler is not None:
            raw_input["scheduler"] = scheduler
        if preset is not None:
            raw_input["preset"] = preset
        if preset_data is not None:
            raw_input["preset_data"] = preset_data
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


def list_swarmui_models(base_url: str, auth_header: str, path: str = "", depth: int = 3, subtype: str = "Stable-Diffusion", sortBy: str = "Name", allowRemote: bool = True, sortReverse: bool = False):
    """
    Calls SwarmUI's /API/ListModels endpoint to get a list of available models.
    Recursively traverses folders to depth specified.
    
    Args:
        base_url: SwarmUI base URL
        auth_header: Authentication header value
        path: Folder path to search, "" for root
        depth: Maximum folder depth to search
        subtype: Model subtype (Stable-Diffusion, LoRA, etc.)
        sortBy: How to sort models (Name, DateCreated, DateModified)
        allowRemote: Include remote models not on local server
        sortReverse: Reverse sort order if True
        
    Returns:
        List of model objects or names
    """
    log.info(f"Fetching SwarmUI models from {base_url}")
    
    headers = {**default_headers}
    if auth_header:
        headers["Authorization"] = auth_header
    
    # First get a session ID
    session_id = get_swarmui_session_id(base_url, auth_header)
    if not session_id:
        log.error("Failed to obtain SwarmUI session ID for listing models")
        return []
    
    log.debug(f"Making request to {base_url}/API/ListModels with session_id: {session_id}")
    
    try:
        data = {
            "session_id": session_id,
            "subtype": subtype,
            "path": path,
            "sortBy": sortBy,
            "allowRemote": allowRemote,
            "sortReverse": sortReverse
        }
        
        log.debug(f"ListModels request data: {data}")
        
        resp = requests.post(
            f"{base_url}/API/ListModels",
            headers=headers,
            json=data,
            timeout=20
        )
        
        # Log the raw response for debugging
        log.debug(f"Raw response status: {resp.status_code}")
        log.debug(f"Raw response headers: {resp.headers}")
        log.debug(f"Raw response text snippet: {resp.text[:500]}...")  # Log first 500 chars
        
        resp.raise_for_status()
        
        # Try to parse as JSON
        try:
            result = resp.json()
            log.debug(f"Parsed JSON response: {result}")
        except Exception as e:
            log.error(f"Failed to parse ListModels response as JSON: {e}")
            # If it's not JSON, try to determine if it's HTML or other format
            if resp.text and resp.text.strip().startswith("<!DOCTYPE html>"):
                log.error("Received HTML response instead of JSON. This might indicate an authentication issue or incorrect URL")
            return []
            
        # Extract models from response
        models = []
        if "models" in result:
            models = result["models"]
        elif "data" in result and "models" in result["data"]:
            models = result["data"]["models"]
            
        # Process models into the expected format for the UI
        formatted_models = []
        for model in models:
            model_id = model.get("name", "")
            model_name = model.get("display_name", model_id)
            
            # Remove file extension if present
            if "." in model_id:
                base_model_id = model_id.split(".")[0]
                formatted_models.append({"id": base_model_id, "name": model_name})
            else:
                formatted_models.append({"id": model_id, "name": model_name})
            
        log.debug(f"Extracted {len(formatted_models)} models")
        return formatted_models
        
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


def get_swarmui_presets(base_url: str, auth_header: str) -> List[Dict[str, Any]]:
    """
    Fetches user presets from SwarmUI using the /API/GetMyUserData endpoint.
    """
    headers = {**default_headers}
    if auth_header:
        headers["Authorization"] = auth_header
    
    log.debug(f"Making request to {base_url}/API/GetMyUserData for presets")
    
    try:
        # First get a session ID
        session_id = get_swarmui_session_id(base_url, auth_header)
        if not session_id:
            log.error("Failed to obtain SwarmUI session ID for presets")
            return []
            
        # Some implementations require session_id in the request
        data = {"session_id": session_id}
        
        resp = requests.post(
            f"{base_url}/API/GetMyUserData",
            headers=headers,
            json=data,
            timeout=20
        )
        
        # Log responses for debugging
        log.debug(f"Raw presets response status: {resp.status_code}")
        log.debug(f"Raw presets response headers: {resp.headers}")
        log.debug(f"Raw presets response text snippet: {resp.text[:500]}...")
        
        resp.raise_for_status()
        
        try:
            data = resp.json()
            log.debug(f"Parsed presets data: {data}")
            
            # Extract presets from response (they may be nested)
            presets = []
            if data and isinstance(data, dict):
                if "presets" in data:
                    presets = data["presets"]
                elif "data" in data and isinstance(data["data"], dict) and "presets" in data["data"]:
                    presets = data["data"]["presets"]
            
            # Format presets
            formatted_presets = []
            for preset in presets:
                preset_id = preset.get("id", "")
                preset_name = preset.get("name", preset_id)
                formatted_presets.append({
                    "id": preset_id,
                    "name": preset_name,
                    "data": preset
                })
            
            log.debug(f"Extracted {len(formatted_presets)} presets")
            return formatted_presets
        except Exception as e:
            log.error(f"Failed to parse presets response as JSON: {e}")
            return []
            
    except Exception as e:
        log.exception(f"Error fetching SwarmUI presets: {e}")
        return []


def get_swarmui_parameters(base_url: str, auth_header: str) -> Dict[str, Any]:
    """
    Fetches available parameters for SwarmUI using the /API/ListT2IParams endpoint.
    This includes available samplers, schedulers, and other parameters.
    """
    headers = {**default_headers}
    if auth_header:
        headers["Authorization"] = auth_header
    
    log.debug(f"Making request to {base_url}/API/ListT2IParams")
    
    try:
        # First get a session ID
        session_id = get_swarmui_session_id(base_url, auth_header)
        if not session_id:
            log.error("Failed to obtain SwarmUI session ID for parameters")
            return {}
            
        # Some implementations require session_id in the request
        data = {"session_id": session_id}
        
        resp = requests.post(
            f"{base_url}/API/ListT2IParams",
            headers=headers,
            json=data,
            timeout=20
        )
        
        # Log responses for debugging
        log.debug(f"Raw parameters response status: {resp.status_code}")
        log.debug(f"Raw parameters response headers: {resp.headers}")
        log.debug(f"Raw parameters response text snippet: {resp.text[:500]}...")
        
        resp.raise_for_status()
        
        try:
            data = resp.json()
            log.debug(f"Parsed parameters data: {data}")
            
            # Extract parameters
            params = {}
            if "samplers" in data:
                params["samplers"] = data["samplers"]
            elif "data" in data and "samplers" in data["data"]:
                params["samplers"] = data["data"]["samplers"]
            
            if "schedulers" in data:
                params["schedulers"] = data["schedulers"]
            elif "data" in data and "schedulers" in data["data"]:
                params["schedulers"] = data["data"]["schedulers"]
                
            log.debug(f"Extracted parameters: {params}")
            return params
        except Exception as e:
            log.error(f"Failed to parse parameters response as JSON: {e}")
            return {}
            
    except Exception as e:
        log.exception(f"Error fetching SwarmUI parameters: {e}")
        return {}
