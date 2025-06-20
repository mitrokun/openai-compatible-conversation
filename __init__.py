"""The OpenAI Compatible Conversation integration."""

from __future__ import annotations

import base64
import openai
import os
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
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN, LOGGER, CONF_BASE_URL, CONF_CHAT_MODEL, CONF_MAX_TOKENS, RECOMMENDED_CHAT_MODEL, RECOMMENDED_MAX_TOKENS

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_DESCRIBE_IMAGE = "mistral_vision"
PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type OpenAICompatibleConfigEntry = ConfigEntry[openai.AsyncClient]

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

        # Проверка доступа к файлу
        image_path = call.data["image_path"]
        if not hass.config.is_allowed_path(image_path):
            raise HomeAssistantError(
                message=f"Cannot read `{image_path}`, no access to path",
                translation_domain=DOMAIN,
                translation_key="cannot_read_path",
                translation_placeholders={"image_path": image_path}
            )
        if not os.path.exists(image_path):
            raise HomeAssistantError(
                message=f"`{image_path}` does not exist",
                translation_domain=DOMAIN,
                translation_key="file_not_found",
                translation_placeholders={"image_path": image_path}
            )

        # Кодирование изображения в base64
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise HomeAssistantError(
                message=f"Error encoding image: {e}",
                translation_domain=DOMAIN,
                translation_key="image_encoding_error",
                translation_placeholders={"error": str(e)}
            )

        # Формирование сообщения для Mistral API
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

        # Отправка запроса
        try:
            response = await client.chat.completions.create(
                model=entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                messages=messages,
                max_tokens=call.data.get("max_tokens", entry.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)),
            )
            return {"text": response.choices[0].message.content}
        except openai.OpenAIError as err:
            raise HomeAssistantError(
                message=f"Error processing image: {err}",
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
        SERVICE_DESCRIBE_IMAGE,
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
    return True

async def async_setup_entry(hass: HomeAssistant, entry: OpenAICompatibleConfigEntry) -> bool:
    """Set up OpenAI Compatible Conversation from a config entry."""
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
        base_url=entry.data[CONF_BASE_URL],
    )

    # Cache current platform data which gets added to each request (caching done by library)
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