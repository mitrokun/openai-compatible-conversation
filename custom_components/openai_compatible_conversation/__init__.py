"""The OpenAI Compatible Conversation integration."""

from __future__ import annotations

import base64
import openai
import os
import voluptuous as vol
import httpx
import json

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN, LOGGER, CONF_BASE_URL, CONF_CHAT_MODEL, CONF_MAX_TOKENS, RECOMMENDED_CHAT_MODEL, RECOMMENDED_MAX_TOKENS

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_MISTRAL_VISION = "mistral_vision"
SERVICE_WEB_SEARCH = "web_search"

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type OpenAICompatibleConfigEntry = ConfigEntry[openai.AsyncClient]


async def web_search(hass: HomeAssistant, call: ServiceCall) -> ServiceResponse:
    """
    Задает вопрос агенту Mistral с возможностью поиска в интернете.
    (остальное описание без изменений)
    """
    entry_id = call.data["config_entry"]
    entry = hass.config_entries.async_get_entry(entry_id)

    if entry is None or entry.domain != DOMAIN:
        raise ServiceValidationError(
            translation_domain=DOMAIN,
            translation_key="invalid_config_entry",
            translation_placeholders={"config_entry": entry_id},
        )

    api_key = entry.data[CONF_API_KEY]
    agent_id = entry.data.get("agent_id")

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    http_client = get_async_client(hass)

    # --- 1. Создаем агента, если его еще нет ---
    if not agent_id:
        LOGGER.info("No agent_id found, creating a new web search agent via HTTP...")
        agent_creation_url = "https://api.mistral.ai/v1/agents"
        agent_payload = {
            "model": "mistral-medium-latest",
            "name": f"Home Assistant Agent ({entry.title})",
            "tools": [{"type": "web_search"}]
        }
        try:
            response = await http_client.post(
                agent_creation_url, headers=headers, json=agent_payload, timeout=20.0
            )
            response.raise_for_status()
            
            agent_id = response.json().get("id")
            if not agent_id:
                raise HomeAssistantError("Failed to get agent_id from creation response.")

            LOGGER.info(f"New agent created with ID: {agent_id}")
            
            new_data = entry.data.copy()
            new_data["agent_id"] = agent_id
            hass.config_entries.async_update_entry(entry, data=new_data)

        except httpx.HTTPStatusError as err:
            raise HomeAssistantError(f"API error creating Mistral agent: {err.response.status_code} - {err.response.text}") from err
        except Exception as err:
            raise HomeAssistantError(f"Failed to create Mistral agent: {err}") from err

    # --- 2. Начинаем разговор с агентом ---
    LOGGER.info(f"Starting conversation with agent_id: {agent_id}")
    conversation_url = "https://api.mistral.ai/v1/conversations"
    conversation_payload = {
        "agent_id": agent_id,
        "inputs": [
            {"role": "user", "content": call.data["prompt"]}
        ]
    }
    try:
        response = await http_client.post(
            conversation_url, headers=headers, json=conversation_payload, timeout=60.0
        )
        LOGGER.debug(
            "Mistral agent raw response | Status: %s | Body: %s",
            response.status_code,
            response.text
        )
        response.raise_for_status()
        
        try:
            data = response.json()
        except json.JSONDecodeError:
            LOGGER.warning("Mistral API returned a non-JSON response. Using raw text as result.")
            return {"text": response.text.strip()}

        # --- 3. УНИВЕРСАЛЬНЫЙ ПАРСЕР ---
        final_text = ""
        if not isinstance(data, dict):
            raise HomeAssistantError(f"Expected a dictionary from API, but got {type(data)}")

        for output in data.get("outputs", []):
            if output.get("type") == "message.output":
                content = output.get("content")

                # Случай 1: content - это одна строка
                if isinstance(content, str):
                    final_text = content
                    break

                # Случай 2: content - это список из частей
                elif isinstance(content, list):
                    text_parts = []
                    for chunk in content:
                        if isinstance(chunk, dict) and chunk.get("type") == "text":
                            text_parts.append(chunk.get("text", ""))
                    
                    if text_parts:
                        final_text = "".join(text_parts)
                        break
        
        # --- 4. Возвращаем ответ, содержащий только текст ---
        return {"text": final_text.strip()}

    except httpx.HTTPStatusError as err:
        error_body = err.response.text
        LOGGER.error(
            "API error during conversation: %s - %s",
            err.response.status_code,
            error_body
        )
        raise HomeAssistantError(f"API error during conversation: {err.response.status_code} - {error_body}") from err
    except Exception as err:
        LOGGER.error("Unexpected error during conversation with agent", exc_info=True)
        raise HomeAssistantError(f"Error during conversation with agent: {err}") from err


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Compatible Conversation."""

    async def render_image(call: ServiceCall) -> ServiceResponse:
        """Render an image with dall-e."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        client: openai.AsyncClient = entry.runtime_data

        try:
            response = await client.images.generate(
                model="dall-e-3",
                prompt=call.data["prompt"],
                size=call.data["size"],
                quality=call.data["quality"],
                style=call.data["style"],
                response_format="url",
                n=1,
            )
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        return response.data[0].model_dump(exclude={"b64_json"})

    async def mistral_vision(call: ServiceCall) -> ServiceResponse:
        """Describe an image using Mistral AI."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        client: openai.AsyncClient = entry.runtime_data

        image_path = call.data["image_path"]
        if not hass.config.is_allowed_path(image_path):
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="cannot_read_path",
                translation_placeholders={"image_path": image_path}
            )
        if not os.path.exists(image_path):
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="file_not_found",
                translation_placeholders={"image_path": image_path}
            )

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="image_encoding_error",
                translation_placeholders={"error": str(e)}
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": call.data["prompt"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]

        try:
            response = await client.chat.completions.create(
                model=entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                messages=messages,
                max_tokens=call.data.get("max_tokens", entry.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)),
            )
            return {"text": response.choices[0].message.content}
        except openai.OpenAIError as err:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="image_processing_error",
                translation_placeholders={"error": str(err)}
            )

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        render_image,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {"integration": DOMAIN}
                ),
                vol.Required("prompt"): cv.string,
                vol.Optional("size", default="1024x1024"): vol.In(
                    ("1024x1024", "1024x1792", "1792x1024")
                ),
                vol.Optional("quality", default="standard"): vol.In(("standard", "hd")),
                vol.Optional("style", default="vivid"): vol.In(("vivid", "natural")),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        SERVICE_MISTRAL_VISION, # <-- Заменил константу на новую
        mistral_vision,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {"integration": DOMAIN}
                ),
                vol.Required("prompt"): cv.string,
                vol.Required("image_path"): cv.string,
                vol.Optional("max_tokens", default=RECOMMENDED_MAX_TOKENS): vol.All(
                    vol.Coerce(int), vol.Range(min=50, max=1000)
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    async def handle_web_search(call: ServiceCall) -> ServiceResponse:
        """Асинхронная обертка для правильного вызова сервиса web_search."""
        return await web_search(hass, call)

    hass.services.async_register(
        DOMAIN,
        SERVICE_WEB_SEARCH,
        handle_web_search,
        schema=vol.Schema({
            vol.Required("config_entry"): selector.ConfigEntrySelector({"integration": DOMAIN}),
            vol.Required("prompt"): cv.string,
        }),
        supports_response=SupportsResponse.ONLY,
    )
    return True

async def async_setup_entry(hass: HomeAssistant, entry: OpenAICompatibleConfigEntry) -> bool:
    """Set up OpenAI Compatible Conversation from a config entry."""
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
        base_url=entry.data[CONF_BASE_URL],
    )

    _ = await hass.async_add_executor_job(client.platform_headers)

    try:
        await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)
    except openai.AuthenticationError as err:
        LOGGER.error("Invalid API key: %s", err)
        return False
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)