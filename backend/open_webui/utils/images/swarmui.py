import asyncio
import json
import logging
import random
import urllib.parse
import urllib.request
import requests
from typing import Optional, Dict, Any

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
from open_webui.env import SRC_LOG_LEVELS
from pydantic import BaseModel

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["SWARMUI"] if "SWARMUI" in SRC_LOG_LEVELS else logging.INFO)

default_headers = {"User-Agent": "Mozilla/5.0"}


def queue_prompt(prompt, client_id, base_url, api_key):
    log.info("queue_prompt")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    log.debug(f"queue_prompt data: {data}")
    try:
        req = urllib.request.Request(
            f"{base_url}/prompt",
            data=data,
            headers={**default_headers, "Authorization": f"Bearer {api_key}"},
        )
        response = urllib.request.urlopen(req).read()
        return json.loads(response)
    except Exception as e:
        log.exception(f"Error while queuing prompt: {e}")
        raise e


def get_image(filename, subfolder, folder_type, base_url, api_key):
    log.info("get_image")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    req = urllib.request.Request(
        f"{base_url}/view?{url_values}",
        headers={**default_headers, "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req) as response:
        return response.read()


def get_image_url(filename, subfolder, folder_type, base_url):
    log.info("get_image")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    return f"{base_url}/view?{url_values}"


def get_history(prompt_id, base_url, api_key):
    log.info("get_history")

    req = urllib.request.Request(
        f"{base_url}/history/{prompt_id}",
        headers={**default_headers, "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def get_images(ws, prompt, client_id, base_url, api_key):
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


def get_swarmui_session_id(base_url: str, api_key: str) -> str:
    """Obtain a SwarmUI session ID by calling the correct session API."""
    headers = {"User-Agent": "Mozilla/5.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.post(
            f"{base_url}/API/GetNewSession",
            headers=headers,
            json={},
            timeout=20
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("session_id")
    except Exception as e:
        log.error(f"Failed to get SwarmUI session id: {e}")
        return None


def swarmui_generate_image(
    model: str,
    payload: SwarmUIGenerateImageForm,
    client_id: str,
    base_url: str,
    api_key: str
) -> Dict[str, Any]:
    """
    Calls SwarmUI's HTTP API to generate images from text prompt.
    Now includes session id in payload.
    """
    log.info(f"swarmui_generate_image: model={model}, client_id={client_id}")
    headers = {"User-Agent": "Mozilla/5.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Get session id
    session_id = get_swarmui_session_id(base_url, api_key)
    if not session_id:
        return {"error": "Could not obtain SwarmUI session id"}
    # Prepare rawInput with all required parameters
    raw_input = {
        "prompt": payload.prompt,
        "width": payload.width,
        "height": payload.height,
        "steps": payload.steps,
        "images": payload.n,
        "session_id": session_id,
    }
    if payload.negative_prompt:
        raw_input["negative_prompt"] = payload.negative_prompt
    if payload.seed is not None:
        raw_input["seed"] = payload.seed
    if model:
        raw_input["model"] = model
    # POST to /API/GenerateText2Image
    data = {
        "images": payload.n,
        "session_id": session_id,
        **{k: v for k, v in raw_input.items() if k != "session_id"}
    }
    try:
        resp = requests.post(
            f"{base_url}/API/GenerateText2Image",
            headers=headers,
            json=data,
            timeout=120
        )
        resp.raise_for_status()
        result = resp.json()
        # Swarm returns {"images": [ ... ]} or {"error": ...}
        if "error_id" in result and result["error_id"] == "invalid_session_id":
            # Try again with a new session
            session_id = get_swarmui_session_id(base_url, api_key)
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
        images = []
        for img_path in result.get("images", []):
            if img_path.startswith("data:"):
                images.append({"url": img_path})
            else:
                img_url = f"{base_url}/{img_path}" if not img_path.startswith("http") else img_path
                images.append({"url": img_url})
        return {"data": images}
    except Exception as e:
        log.exception(f"Error in SwarmUI image generation: {e}")
        return {"error": str(e)}


def list_swarmui_models(base_url: str, api_key: str, path: str = "", depth: int = 1, subtype: str = "Stable-Diffusion", sortBy: str = "Name", allowRemote: bool = True, sortReverse: bool = False):
    """
    Calls SwarmUI's /API/ListModels endpoint to retrieve available models.
    """
    import requests
    url = f"{base_url}/API/ListModels"
    payload = {
        "path": path,
        "depth": depth,
        "subtype": subtype,
        "sortBy": sortBy,
        "allowRemote": allowRemote,
        "sortReverse": sortReverse
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        # Return a flat list of model files with their metadata
        return data.get("files", [])
    except Exception as e:
        log.exception(f"Error fetching SwarmUI models: {e}")
        return []


def select_swarmui_model(base_url: str, api_key: str, model_path: str, backendId: str = None):
    """
    Calls SwarmUI's /API/SelectModel endpoint to load the selected model.
    """
    import requests
    url = f"{base_url}/API/SelectModel"
    payload = {
        "model": model_path,
        "backendId": backendId
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("success", False)
    except Exception as e:
        log.exception(f"Error selecting SwarmUI model: {e}")
        return False
