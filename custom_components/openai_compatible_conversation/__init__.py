"""The OpenAI Compatible Conversation integration."""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import httpx
import openai
import voluptuous as vol

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
from homeassistant.helpers import (
    chat_session,
    config_validation as cv,
    entity_registry as er,
    selector,
)
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from homeassistant.components import conversation as ha_conversation, tts
from homeassistant.components.assist_pipeline import async_get_pipeline
from homeassistant.components.assist_satellite.const import (
    DOMAIN as ASSIST_SATELLITE_DOMAIN,
)

from .const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
)
from .conversation import OpenAICompatibleConversationEntity

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_MISTRAL_VISION = "mistral_vision"
SERVICE_WEB_SEARCH = "web_search"
SERVICE_STREAM_RESPONSE = "stream_response"
SERVICE_GENERATE_STRUCTURED = "generate_structured_data"


PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type OpenAICompatibleConfigEntry = ConfigEntry[openai.AsyncClient]


# --- web_search ---
async def web_search(hass: HomeAssistant, call: ServiceCall) -> ServiceResponse:

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

    if not agent_id:
        LOGGER.info("No agent_id found, creating a new web search agent via HTTP...")
        agent_creation_url = "https://api.mistral.ai/v1/agents"
        agent_payload = {
            "model": "mistral-medium-latest",
            "name": f"Home Assistant Agent ({entry.title})",
            "instructions": (
                "Follow these instructions:\n"
                "- Use the `web_search` tool for up-to-date information on the internet.\n"
                "- Always write numerical values as words, not digits, in your final answer."
            ),
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

            LOGGER.info("New agent created with ID: %s", agent_id)
            
            new_data = dict(entry.data)
            new_data["agent_id"] = agent_id
            hass.config_entries.async_update_entry(entry, data=new_data)

        except httpx.HTTPStatusError as err:
            raise HomeAssistantError(f"API error creating Mistral agent: {err.response.status_code} - {err.response.text}") from err
        except Exception as err:
            raise HomeAssistantError(f"Failed to create Mistral agent: {err}") from err

    LOGGER.info("Starting conversation with agent_id: %s", agent_id)
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

        final_text = ""
        if not isinstance(data, dict):
            raise HomeAssistantError(f"Expected a dictionary from API, but got {type(data)}")

        for output in data.get("outputs", []):
            if output.get("type") == "message.output":
                content = output.get("content")
                if isinstance(content, str):
                    final_text = content
                    break
                elif isinstance(content, list):
                    text_parts = []
                    for chunk in content:
                        if isinstance(chunk, dict) and chunk.get("type") == "text":
                            text_parts.append(chunk.get("text", ""))
                    
                    if text_parts:
                        final_text = "".join(text_parts)
                        break
        
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
        LOGGER.exception("Unexpected error during conversation with agent")
        raise HomeAssistantError(f"Error during conversation with agent: {err}") from err


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Compatible Conversation."""

    # ---  mistral_vision ---
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
        
        if not await hass.async_add_executor_job(os.path.exists, image_path):
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="file_not_found",
                translation_placeholders={"image_path": image_path}
            )

        try:
            image_bytes = await hass.async_add_executor_job(
                lambda: open(image_path, "rb").read()
            )
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
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
            model_to_use = call.data.get("model", "mistral-small-latest")

            response = await client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=call.data.get("max_tokens", RECOMMENDED_MAX_TOKENS),
            )
            content = response.choices[0].message.content
            return {"text": content if content else ""}
        except openai.OpenAIError as err:
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="image_processing_error",
                translation_placeholders={"error": str(err)}
            )

    hass.services.async_register(
        DOMAIN,
        SERVICE_MISTRAL_VISION,
        mistral_vision,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {"integration": DOMAIN}
                ),
                vol.Required("prompt"): cv.string,
                vol.Required("image_path"): cv.string,
                vol.Optional("model"): cv.string,
                vol.Optional("max_tokens", default=RECOMMENDED_MAX_TOKENS): vol.All(
                    vol.Coerce(int), vol.Range(min=50, max=1000)
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    # --- web_search ---
    async def handle_web_search(call: ServiceCall) -> ServiceResponse:
        """Async wrapper for the web_search service call."""
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

    # --- stream_response ---
    async def handle_stream_response(call: ServiceCall) -> ServiceResponse:
        """Handle the stream_response service call."""
        
        INITIAL_BUFFER_CHAR_COUNT = 30
        
        full_prompt = call.data["prompt"]
        conversation_id_in = call.data.get("conversation_id")
        max_tokens = call.data.get("max_tokens")


        def get_history_friendly_prompt(text: str) -> str:
            if len(text) < 200 and "\n" not in text:
                return text
            
            first_line = text.split('\n')[0].strip()
            
            if len(first_line) > 200:
                return first_line[:200] + "... [original data intentionally excluded]"
            
            return first_line + " ... [original data intentionally excluded]"
        
        history_prompt = get_history_friendly_prompt(full_prompt)


        with chat_session.async_get_chat_session(hass, conversation_id_in) as session, \
             ha_conversation.async_get_chat_log(hass, session) as chat_log:
            
            conversation_id = session.conversation_id
            chat_log.async_add_user_content(ha_conversation.UserContent(content=history_prompt))

            if not (device_ids := call.data.get("device_id")):
                raise HomeAssistantError("device_id must be specified in 'target'")
            
            device_id = (device_ids[0] if isinstance(device_ids, list) else device_ids)
            
            ent_reg = er.async_get(hass)
            device_entries = er.async_entries_for_device(ent_reg, device_id)
            
            assist_satellite_entity_id = next(
                (e.entity_id for e in device_entries if e.domain == "assist_satellite"), None
            )
            if not assist_satellite_entity_id:
                raise HomeAssistantError(f"Could not find an assist_satellite entity for device {device_id}")

            satellite_entity = hass.data.get("assist_satellite").get_entity(assist_satellite_entity_id)
            pipeline = async_get_pipeline(hass, satellite_entity._resolve_pipeline())
            
            conversation_agent_id = call.data.get("agent_entity_id") or pipeline.conversation_engine
            if not conversation_agent_id:
                raise HomeAssistantError(f"Agent not specified in service call and not configured in pipeline '{pipeline.name}'")

            agent_entity = hass.data.get("conversation").get_entity(conversation_agent_id)
            if not isinstance(agent_entity, OpenAICompatibleConversationEntity):
                raise HomeAssistantError(f"Agent '{conversation_agent_id}' is not an OpenAI Compatible agent.")

            user_input = ha_conversation.ConversationInput(
                text=full_prompt,
                context=call.context, 
                conversation_id=conversation_id,
                device_id=device_id, 
                satellite_id=assist_satellite_entity_id,
                language=pipeline.language, 
                agent_id=conversation_agent_id,
            )

            text_generator = agent_entity.async_stream_response(
                user_input, 
                max_tokens_override=max_tokens
            )
            
            text_iterator = text_generator
            initial_buffer = []
            buffered_chars = 0
            
            try:
                async for chunk in text_iterator:
                    initial_buffer.append(chunk)
                    buffered_chars += len(chunk)
                    if buffered_chars >= INITIAL_BUFFER_CHAR_COUNT:
                        break
            except StopAsyncIteration:
                pass

            if not initial_buffer:
                LOGGER.debug("LLM returned an empty response.")
                if call.return_response:
                    return {"response": ""}
                return None

            initial_buffer_str = "".join(initial_buffer)
            full_response_parts = [initial_buffer_str]

            async def combined_generator():
                yield initial_buffer_str
                async for chunk in text_iterator:
                    full_response_parts.append(chunk)
                    yield chunk
            
            tts_engine, tts_language, tts_voice = pipeline.tts_engine, pipeline.tts_language, pipeline.tts_voice
            if not tts_engine:
                raise HomeAssistantError(f"TTS engine not configured for satellite {assist_satellite_entity_id}")
            
            tts_options = {}
            if tts_voice: tts_options[tts.ATTR_VOICE] = tts_voice
            tts_stream = tts.async_create_stream(hass, engine=tts_engine, language=tts_language, options=tts_options)

            tts_stream.async_set_message_stream(combined_generator())

            await hass.services.async_call(
                ASSIST_SATELLITE_DOMAIN, "announce",
                {"entity_id": assist_satellite_entity_id, "media_id": tts_stream.media_source_id, "preannounce": False},
                blocking=True,
                context=call.context,
            )

            try:
                async for _ in tts_stream.async_stream_result():
                    pass
            except Exception as e:
                LOGGER.error("Error with TTS stream: %s", e, exc_info=True)
                raise HomeAssistantError(f"Error while streaming audio: {e}") from e

            full_response_text = "".join(full_response_parts)
            
            chat_log.async_add_assistant_content_without_tools(
                ha_conversation.AssistantContent(content=full_response_text, agent_id=conversation_agent_id)
            )
            
            if call.return_response:
                return {"response": full_response_text}
            
            return None

    hass.services.async_register(
        DOMAIN,
        SERVICE_STREAM_RESPONSE,
        handle_stream_response,
        schema=vol.Schema(
            vol.All(
                {
                    vol.Optional("agent_entity_id"): cv.entity_id,
                    vol.Required("prompt"): cv.string,
                    vol.Optional("conversation_id"): cv.string,
                    vol.Optional("device_id"): cv.ensure_list,
                    vol.Optional("area_id"): cv.ensure_list,
                    vol.Optional("entity_id"): cv.entity_ids,
                    vol.Optional("max_tokens"): cv.positive_int,
                }
            )
        ),
        supports_response=SupportsResponse.OPTIONAL,
    )

    # --- generate_structured_data ---
    async def generate_structured_data(call: ServiceCall) -> ServiceResponse:
        """Generate a structured response using a JSON schema, optionally from an image."""

        def clean_data_recursively(data: Any) -> Any:
            """Recursively traverse a dict/list and replace non-breaking spaces in strings."""
            if isinstance(data, str):
                return data.replace('\xa0', ' ')
            if isinstance(data, dict):
                return {k: clean_data_recursively(v) for k, v in data.items()}
            if isinstance(data, list):
                return [clean_data_recursively(item) for item in data]
            return data

        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        client: openai.AsyncClient = entry.runtime_data
        prompt = call.data["prompt"]
        
        json_schema = clean_data_recursively(call.data["json_schema"])

        model = call.data.get("model", "mistral-large-latest")
        image_path = call.data.get("image_path")

        # Умное исправление схемы
        if json_schema.get("type") == "object" and "additionalProperties" not in json_schema:
            LOGGER.debug("Adding 'additionalProperties: false' to the schema for API compatibility.")
            json_schema["additionalProperties"] = False

        content_parts = [{"type": "text", "text": prompt}]

        if image_path:
            if not hass.config.is_allowed_path(image_path):
                raise HomeAssistantError(f"Cannot read image from path {image_path}, path not allowed.")
            
            if not await hass.async_add_executor_job(os.path.exists, image_path):
                raise HomeAssistantError(f"Image file not found at {image_path}.")

            try:
                image_bytes = await hass.async_add_executor_job(
                    lambda: open(image_path, "rb").read()
                )
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            except Exception as e:
                raise HomeAssistantError(f"Error processing image file: {e}") from e

        messages = [
            {"role": "user", "content": content_parts}
        ]

        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "structured_data", "strict": True, "schema": json_schema}
        }
        
        content = ""
        try:
            LOGGER.debug("Calling chat completions API with cleaned structured response format.")
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format, # type: ignore
                max_tokens=call.data.get("max_tokens", 2000),
                temperature=call.data.get("temperature", 0.1),
            )
            
            if not response.choices:
                LOGGER.error("API returned a successful response but with no choices. Full response: %s", response)
                raise HomeAssistantError("API returned an empty response. This might be due to a content filter or an API-side issue.")

            content = response.choices[0].message.content
            if not content:
                raise HomeAssistantError("Received an empty content from the API.")

            structured_response = json.loads(content)
            return {"data": structured_response}

        except json.JSONDecodeError as err:
            LOGGER.error("Failed to decode JSON from API response. Content: %s", content)
            raise HomeAssistantError(f"Failed to decode JSON from API response: {err}") from err
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"API error: {err}") from err
        except Exception as err:
            LOGGER.error("Unexpected error during structured data generation: %s", err, exc_info=True)
            raise HomeAssistantError(f"An unexpected error occurred: {err}") from err

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_STRUCTURED,
        generate_structured_data,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {"integration": DOMAIN}
                ),
                vol.Required("prompt"): cv.string,
                vol.Optional("image_path"): cv.string,
                vol.Required("json_schema"): dict,
                vol.Optional("model"): cv.string,
                vol.Optional("max_tokens"): cv.positive_int,
                vol.Optional("temperature"): vol.Coerce(float),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    return True

async def async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_setup_entry(hass: HomeAssistant, entry: OpenAICompatibleConfigEntry) -> bool:
    """Set up OpenAI Compatible Conversation from a config entry."""
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
        base_url=entry.data[CONF_BASE_URL],
    )

    # This line seems to be for internal library usage, keeping it.
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

    entry.async_on_unload(entry.add_update_listener(async_update_listener))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
