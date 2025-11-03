"""Config flow for OpenAI Compatible Conversation integration."""

from __future__ import annotations

import logging
from typing import Any

import openai
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
)

from .const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_ENABLE_KV_CACHE_FIX,
    CONF_MAX_TOKENS,
    CONF_NAME,
    CONF_NO_THINK,
    CONF_PROMPT,
    CONF_STRIP_THINK_TAGS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_AGENT_NAME,
    DOMAIN,
    RECOMMENDED_BASE_URL,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required("name", default="OpenAI Compatible"): str,
        vol.Required(CONF_API_KEY): str,
        vol.Required(CONF_BASE_URL, default=RECOMMENDED_BASE_URL): str,
    }
)

async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect."""
    client = openai.AsyncOpenAI(api_key=data[CONF_API_KEY], base_url=data[CONF_BASE_URL])
    await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)


def agent_schema(
    hass: HomeAssistant, is_new: bool, data: dict[str, Any] | None = None
) -> vol.Schema:
    """Return a schema for the agent configuration."""
    if data is None:
        data = {}

    schema_dict = {}
    
    if is_new:
        schema_dict[vol.Required(CONF_NAME, default=DEFAULT_AGENT_NAME)] = str

    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(label="No control", value="none")
    ]
    hass_apis.extend(
        SelectOptionDict(label=api.name, value=api.id)
        for api in llm.async_get_apis(hass)
    )

    schema_dict.update(
        {
            vol.Optional(
                CONF_PROMPT,
                description={"suggested_value": data.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT)},
            ): TemplateSelector(),
            vol.Optional(
                CONF_LLM_HASS_API,
                description={"suggested_value": data.get(CONF_LLM_HASS_API)},
                default="none",
            ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
            vol.Required(
                CONF_CHAT_MODEL,
                description={"suggested_value": data.get(CONF_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): str,
            vol.Required(
                CONF_MAX_TOKENS,
                description={"suggested_value": data.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Required(
                CONF_TOP_P,
                description={"suggested_value": data.get(CONF_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Required(
                CONF_TEMPERATURE,
                description={"suggested_value": data.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_ENABLE_KV_CACHE_FIX,
                default=data.get(CONF_ENABLE_KV_CACHE_FIX, False),
            ): BooleanSelector(),
            vol.Required(
                CONF_NO_THINK,
                default=data.get(CONF_NO_THINK, False)
            ): BooleanSelector(),
            vol.Required(
                CONF_STRIP_THINK_TAGS,
                default=data.get(CONF_STRIP_THINK_TAGS, False)
            ): BooleanSelector(),
        }
    )
    return vol.Schema(schema_dict)


class OpenAICompatibleConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Compatible Conversation."""

    VERSION = 2

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this handler."""
        return {
            "conversation": ConversationFlowHandler,
        }

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step for creating the main entry."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=STEP_USER_DATA_SCHEMA,
            )

        errors: dict[str, str] = {}
        try:
            await validate_input(self.hass, user_input)
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            title = user_input.pop("name")
            return self.async_create_entry(title=title, data=user_input)

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_DATA_SCHEMA, user_input
            ),
            errors=errors,
        )


class ConversationFlowHandler(ConfigSubentryFlow):
    """Handle a conversation subentry flow for OpenAI Compatible."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the initial step for creating or editing an agent."""

        is_new = self.source != "reconfigure"
        current_data = {}
        if not is_new:
            current_data = self._get_reconfigure_subentry().data

        if user_input is not None:
            updated_data = {**current_data, **user_input}
            if updated_data.get(CONF_LLM_HASS_API) == "none":
                updated_data.pop(CONF_LLM_HASS_API, None)

            if is_new:
                title = updated_data.pop(CONF_NAME)
                return self.async_create_entry(title=title, data=updated_data)
            
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=updated_data,
            )

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                agent_schema(self.hass, is_new=is_new), current_data
            ),
        )

    async_step_user = async_step_init
    async_step_reconfigure = async_step_init
