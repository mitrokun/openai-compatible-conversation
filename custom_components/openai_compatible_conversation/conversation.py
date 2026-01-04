# conversation.py

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
import json
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import openai
from openai import AsyncStream
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
    Function,
)
from openai.types.shared_params import FunctionDefinition
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback

if TYPE_CHECKING:
    from . import OpenAICompatibleConfigEntry

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_NO_THINK,
    CONF_PROMPT,
    CONF_STRIP_THINK_TAGS,
    CONF_ENABLE_KV_CACHE_FIX,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

LOGGER = logging.getLogger(__name__)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAICompatibleConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities from subentries."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue
        agent = OpenAICompatibleConversationEntity(config_entry, subentry)
        async_add_entities([agent], config_subentry_id=subentry.subentry_id)


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)


def _convert_content_to_param(
    content: conversation.Content,
    tool_id_mapping: dict[str, str] | None = None,
) -> ChatCompletionMessageParam:
    """Convert any native chat message for this agent to the native format."""

    def get_mistral_id(original_id: str) -> str:
        # If mapping is not provided (not Mistral), return original
        if tool_id_mapping is None:
            return original_id
        
        # If ID is already mapped, return the cached short version
        if original_id in tool_id_mapping:
            return tool_id_mapping[original_id]
        
        # Take the last 9 characters
        clean_id = original_id[-9:] if len(original_id) >= 9 else original_id.ljust(9, "0")
        
        # Fallback
        if clean_id in tool_id_mapping.values():
            clean_id = original_id[:9]
            
        tool_id_mapping[original_id] = clean_id
        return clean_id

    if content.role == "tool_result":
        assert isinstance(content, conversation.ToolResultContent)
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=get_mistral_id(content.tool_call_id),
            content=json.dumps(content.tool_result),
        )

    if content.role in ("user", "system"):
        return cast(
            ChatCompletionMessageParam,
            {"role": content.role, "content": content.content},
        )

    if content.role == "assistant":
        assert isinstance(content, conversation.AssistantContent)
        if content.tool_calls:
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=content.content if content.content else None,
                tool_calls=[
                    ChatCompletionMessageFunctionToolCallParam(
                        id=get_mistral_id(tool_call.id),
                        function=Function(
                            arguments=json.dumps(tool_call.tool_args),
                            name=tool_call.tool_name,
                        ),
                        type="function",
                    )
                    for tool_call in content.tool_calls
                ],
            )
        else:
            final_content = content.content
            if not final_content:
                final_content = " "
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=final_content,
            )

    LOGGER.warning("Unhandled content type during conversion: %s", type(content))
    return cast(
        ChatCompletionMessageParam,
        {"role": content.role, "content": content.content},
    )


async def _openai_to_ha_stream(
    stream: AsyncStream[ChatCompletionChunk],
    strip_think_tags: bool,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
    """Transform the OpenAI stream into the Home Assistant delta format."""
    first_chunk = True
    active_tool_calls_by_index: dict[int, dict[str, Any]] = {}

    buffer = ""
    is_in_think_block = False
    start_tag = "<think>"
    end_tag = "</think>"

    async for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        LOGGER.debug("Δ: %s", delta)

        if first_chunk and delta.role:
            yield {"role": delta.role}
            first_chunk = False

        if delta.content:
            content_text = ""
            if isinstance(delta.content, list):
                for part in delta.content:
                    if hasattr(part, "type") and part.type == 'text' and hasattr(part, "text"):
                        content_text += part.text
            elif isinstance(delta.content, str):
                content_text = delta.content
            
            if not content_text:
                continue

            if not strip_think_tags:
                yield {"content": content_text}
                continue

            buffer += content_text
            while True:
                if is_in_think_block:
                    end_tag_pos = buffer.find(end_tag)
                    if end_tag_pos != -1:
                        is_in_think_block = False
                        buffer = buffer[end_tag_pos + len(end_tag) :]
                        continue
                    break
                else:
                    start_tag_pos = buffer.find(start_tag)
                    if start_tag_pos != -1:
                        content_to_yield = buffer[:start_tag_pos]
                        if content_to_yield:
                            yield {"content": content_to_yield}

                        is_in_think_block = True
                        buffer = buffer[start_tag_pos + len(start_tag) :]
                        continue

                    if len(buffer) > len(start_tag):
                        safe_yield_len = len(buffer) - len(start_tag)
                        yield {"content": buffer[:safe_yield_len]}
                        buffer = buffer[safe_yield_len:]

                    break

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                if tc_delta.index is None:
                    continue
                current_tc_data = active_tool_calls_by_index.setdefault(
                    tc_delta.index, {"id": None, "name": None, "args_str": ""}
                )
                if tc_delta.id:
                    current_tc_data["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        current_tc_data["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        current_tc_data["args_str"] += tc_delta.function.arguments

        choice_finish_reason = chunk.choices[0].finish_reason
        if choice_finish_reason and active_tool_calls_by_index:
            final_tool_inputs = []
            for index, tc_data in sorted(active_tool_calls_by_index.items()):
                tool_id = tc_data.get("id", f"call_{index}")
                tool_name = tc_data.get("name")
                args_str = tc_data.get("args_str", "{}")
                if not tool_name:
                    continue
                try:
                    parsed_args = json.loads(args_str if args_str else "{}")
                    final_tool_inputs.append(
                        llm.ToolInput(
                            id=tool_id, tool_name=tool_name, tool_args=parsed_args
                        )
                    )
                except json.JSONDecodeError:
                    LOGGER.error(
                        "Failed to decode JSON for tool %s: %s", tool_name, args_str
                    )
            if final_tool_inputs:
                yield {"tool_calls": final_tool_inputs}
            active_tool_calls_by_index.clear()

    if strip_think_tags and not is_in_think_block and buffer:
        yield {"content": buffer}


class OpenAICompatibleConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """OpenAI Compatible conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(
        self,
        entry: OpenAICompatibleConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the agent."""
        self.entry = entry
        self.subentry = subentry
        self.client = entry.runtime_data
        self.options = subentry.data
        self.frozen_times = {}

        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="OpenAI Compatible",
            model="OpenAI Compatible",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle a message."""
        
        assert user_input.agent_id
        options = self.options
        client = self.client
        conversation_id = chat_log.conversation_id

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        enable_kv_cache_fix = options.get(CONF_ENABLE_KV_CACHE_FIX, False)
        
        tools: list[ChatCompletionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        # Determine if we are using Mistral (affects ID generation and 'user' field)
        base_url = str(getattr(client, "base_url", ""))
        is_mistral = "mistral.ai" in base_url.lower()

        for _iteration in range(MAX_TOOL_ITERATIONS):
            # === HISTORY SANITIZER ===
            # ВАЖНО: Выполняем внутри цикла, чтобы видеть свежие изменения в chat_log
            sanitized_content_list = []
            raw_content = chat_log.content
            
            i = 0
            while i < len(raw_content):
                item = raw_content[i]

                # [FIX START] Удаление "сирот" (Orphaned Tool Results)
                # Если текущий элемент - результат инструмента, но предыдущий элемент в списке
                # НЕ является Ассистентом (с вызовами) или другим Результатом, значит
                # этот результат "оторвался" (например, из-за вклинившегося User message).
                # Мы его пропускаем, чтобы не ломать структуру API.
                if isinstance(item, conversation.ToolResultContent):
                    prev = sanitized_content_list[-1] if sanitized_content_list else None
                    
                    # Валидный родитель: Assistant с tool_calls ИЛИ другой ToolResult (в случае параллельных вызовов)
                    is_valid_parent = False
                    if prev:
                        if isinstance(prev, conversation.AssistantContent) and prev.tool_calls:
                            is_valid_parent = True
                        elif isinstance(prev, conversation.ToolResultContent):
                            is_valid_parent = True
                    
                    if not is_valid_parent:
                        LOGGER.debug(
                            "Sanitizer: Dropping orphaned ToolResult (ID: %s). Previous role was: %s", 
                            item.tool_call_id, 
                            prev.role if prev else "None"
                        )
                        i += 1
                        continue
                # [FIX END]

                sanitized_content_list.append(item)
                
                # Если это сообщение с вызовом инструментов
                if isinstance(item, conversation.AssistantContent) and item.tool_calls:
                    next_item = raw_content[i + 1] if i + 1 < len(raw_content) else None
                    
                    # Если следом идет НЕ результат
                    if not isinstance(next_item, conversation.ToolResultContent):
                        LOGGER.debug("Sanitizer: Injecting fake ToolResult for broken chain.")
                        
                        # 1. Вставляем фейковый результат с инструкцией для LLM
                        for tool_call in item.tool_calls:
                            fake_result = conversation.ToolResultContent(
                                agent_id=item.agent_id,
                                tool_call_id=tool_call.id,
                                tool_name=tool_call.tool_name,
                                tool_result={
                                    "error": "Internal handler failed. Do not retry."
                                }
                            )
                            sanitized_content_list.append(fake_result)
                        
                        # 2. Вставляем фейковый ответ ассистента (для Mistral)
                        fake_assistant_response = conversation.AssistantContent(
                            agent_id=item.agent_id,
                            content="The command returned no result."
                        )
                        sanitized_content_list.append(fake_assistant_response)
                
                i += 1
            # ===============================================

            # Create mapping dict for Mistral, otherwise None
            tool_id_mapping = {} if is_mistral else None

            try:
                messages = [
                    _convert_content_to_param(content, tool_id_mapping) 
                    for content in sanitized_content_list
                ]
            except Exception as err:
                LOGGER.error("Error during history regeneration: %s", err)
                raise HomeAssistantError(f"Failed to process history: {err}") from err

            # --- Freeze timestamp ---
            if (
                enable_kv_cache_fix
                and _iteration == 0
                and messages
                and messages[0].get("role") == "system"
            ):
                import re
                time_pattern = re.compile(r"Current time is \d{2}:\d{2}:\d{2}")
                date_pattern = re.compile(r"Today's date is \d{4}-\d{2}-\d{2}")
                system_content = messages[0].get("content", "")

                if conversation_id in self.frozen_times:
                    frozen_context = self.frozen_times[conversation_id]
                    new_content = time_pattern.sub(frozen_context["time"], system_content, 1)
                    new_content = date_pattern.sub(frozen_context["date"], new_content, 1)
                    if system_content != new_content:
                        messages[0]["content"] = new_content
                else:
                    time_match = time_pattern.search(system_content)
                    date_match = date_pattern.search(system_content)
                    if time_match and date_match:
                        MAX_CACHE_SIZE = 10
                        if len(self.frozen_times) >= MAX_CACHE_SIZE:
                            del self.frozen_times[next(iter(self.frozen_times))]
                        self.frozen_times[conversation_id] = {
                            "time": time_match.group(0),
                            "date": date_match.group(0),
                        }
            # --- end ---
            
            if _iteration == 0:
                no_think_enabled = options.get(CONF_NO_THINK, False)
                if no_think_enabled and messages and messages[-1]["role"] == "user":
                    user_message = messages[-1]
                    current_content = str(user_message.get("content", ""))
                    if not current_content.endswith("/no_think"):
                        user_message["content"] = current_content + "/no_think"

            model_args: dict[str, Any] = {
                "model": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                "messages": messages,
                "tools": tools or NOT_GIVEN,
                "tool_choice": "auto" if tools else NOT_GIVEN,
                "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": options.get(
                    CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                ),
                "stream": True,
            }

            # Do not send 'user' field for Mistral
            if not is_mistral:
                model_args["user"] = conversation_id

            LOGGER.debug(
                "FULL LLM REQUEST:\n%s", 
                json.dumps(model_args, indent=2, ensure_ascii=False, default=str)
            )

            max_retries = 3
            retry_delay_s = 1

            for attempt in range(max_retries):
                try:
                    response_stream = await client.chat.completions.create(**model_args)
                    strip_tags = options.get(CONF_STRIP_THINK_TAGS, False)
                    async for _ in chat_log.async_add_delta_content_stream(
                        user_input.agent_id,
                        _openai_to_ha_stream(response_stream, strip_think_tags=strip_tags),
                    ):
                        pass
                    
                    break

                except openai.RateLimitError as err:
                    if attempt + 1 == max_retries:
                        LOGGER.error("Rate limit error after %d attempts: %s", max_retries, err)
                        raise HomeAssistantError("Rate limited or insufficient funds") from err
                    
                    LOGGER.debug(
                        "Rate limited by API. Attempt %d of %d. Retrying in %d s...",
                        attempt + 1,
                        max_retries,
                        retry_delay_s,
                    )
                    await asyncio.sleep(retry_delay_s)

                except openai.OpenAIError as err:
                    LOGGER.error("Error talking to API: %s", err)
                    error_body = getattr(err, "body", None)
                    if error_body:
                        LOGGER.error("API Error Body: %s", error_body)
                    raise HomeAssistantError(f"Error talking to API: {err}") from err
                
                except Exception as err:
                    LOGGER.error("Unexpected streaming error: %s", err)
                    raise HomeAssistantError(f"An unexpected error occurred: {err}") from err

            if not chat_log.unresponded_tool_results:
                break
        
        # --- Фикс на выполнение build-in комманд после ответа LLM с вопросом  ---
        
        result = conversation.async_get_result_from_chat_log(user_input, chat_log)

        if chat_log.continue_conversation:
            KEY_PIPELINE_DATA = "pipeline_conversation_data"
            conv_id = chat_log.conversation_id
            
            # LOGGER.debug(f"Sabotage: Scheduled lock removal for {conv_id}")

            @callback
            def sabotage_agent_lock():
                try:
                    pipeline_data = self.hass.data.get(KEY_PIPELINE_DATA)
                    if not pipeline_data: return

                    data_entry = pipeline_data.get(conv_id)
                    if data_entry and data_entry.continue_conversation_agent:
                        # LOGGER.debug(f"Sabotage: REMOVING LOCK for {conv_id}")
                        data_entry.continue_conversation_agent = None
                        
                except Exception:
                    pass

            self.hass.loop.call_later(0.5, sabotage_agent_lock)

        return result
        # --- end  ---
        # return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def async_stream_response(
        self,
        user_input: conversation.ConversationInput,
        max_tokens_override: int | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream the response from the LLM as text chunks."""  
        chat_log = conversation.ChatLog(self.hass, user_input.conversation_id)
        chat_log.async_add_user_content(
            conversation.UserContent(content=user_input.text)
        )

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                self.options.get(CONF_LLM_HASS_API),
                self.options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError:
            return

        base_url = str(getattr(self.client, "base_url", ""))
        is_mistral = "mistral.ai" in base_url.lower()
        tool_id_mapping = {} if is_mistral else None

        messages = [
            _convert_content_to_param(content, tool_id_mapping) 
            for content in chat_log.content
        ]

        max_tokens = max_tokens_override if max_tokens_override else self.options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)

        model_args: dict[str, Any] = {
            "model": self.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            "messages": messages,
            "max_tokens": max_tokens,
            "top_p": self.options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            "temperature": self.options.get(
                CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
            ),
            "stream": True,
        }
        
        try:
            response_stream = await self.client.chat.completions.create(**model_args)
            async for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if content_text := delta.content:
                    yield content_text

        except openai.OpenAIError as err:
            LOGGER.error("API error during streaming: %s", err)
            yield "Sorry, an error occurred while talking to the AI service."
        except Exception as err:
            LOGGER.error("Unexpected error during streaming: %s", err)
            yield "An unexpected error occurred."
