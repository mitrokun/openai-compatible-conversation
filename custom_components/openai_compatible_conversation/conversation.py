"""Conversation support for OpenAI Compatible APIs."""

import json
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal, cast

import openai
from openai import AsyncStream
from openai._types import NOT_GIVEN

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessage,
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
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import chat_session, device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import OpenAICompatibleConfigEntry
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_NO_THINK,
    CONF_PROMPT,
    CONF_STRIP_THINK_TAGS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenAICompatibleConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = OpenAICompatibleConversationEntity(config_entry)
    async_add_entities([agent])


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
) -> ChatCompletionMessageParam:
    """Convert any native chat message for this agent to the native format."""
    if content.role == "tool_result":
        assert type(content) is conversation.ToolResultContent
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=content.tool_call_id,
            content=json.dumps(content.tool_result),
        )

    if content.role in ("user", "system"):
        return cast(
            ChatCompletionMessageParam,
            {"role": content.role, "content": content.content},
        )

    if content.role == "assistant":
        assert type(content) is conversation.AssistantContent
        if content.tool_calls:
            return ChatCompletionAssistantMessageParam(
                role="assistant",
                content=content.content or "",
                tool_calls=[
                    ChatCompletionMessageFunctionToolCallParam(
                        id=tool_call.id,
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
        {"role": content.role, "content": content.content},  # type: ignore[union-attr]
    )


async def _openai_to_ha_stream(
    stream: AsyncStream[ChatCompletionChunk],
    strip_think_tags: bool,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict, None]:
    """Transform the OpenAI stream into the Home Assistant delta format."""
    first_chunk = True
    active_tool_calls_by_index: dict[int, dict[str, Any]] = {}
    
    # State for stripping think tags
    buffer = ""
    is_in_think_block = False
    start_tag = "<think>"
    end_tag = "</think>"

    async for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        LOGGER.debug("Received stream delta: %s", delta)

        # Handle role on the first chunk
        if first_chunk and delta.role:
            yield {"role": delta.role}
            first_chunk = False
        
        # Handle content with filtering
        if delta.content:
            if not strip_think_tags:
                yield {"content": delta.content}
                continue

            buffer += delta.content
            while True:
                if is_in_think_block:
                    end_tag_pos = buffer.find(end_tag)
                    if end_tag_pos != -1:
                        is_in_think_block = False
                        buffer = buffer[end_tag_pos + len(end_tag):]
                        # Continue loop to process buffer after the tag
                        continue
                    break # Need more data
                else:
                    start_tag_pos = buffer.find(start_tag)
                    if start_tag_pos != -1:
                        content_to_yield = buffer[:start_tag_pos]
                        if content_to_yield:
                            yield {"content": content_to_yield}
                        
                        is_in_think_block = True
                        buffer = buffer[start_tag_pos + len(start_tag):]
                        # Continue loop to process buffer after the tag
                        continue
                    
                    # No start tag found. We can safely yield a portion of the buffer
                    # to avoid holding too much data, leaving a small part to prevent
                    # splitting a potential start tag.
                    if len(buffer) > len(start_tag):
                        safe_yield_len = len(buffer) - len(start_tag)
                        yield {"content": buffer[:safe_yield_len]}
                        buffer = buffer[safe_yield_len:]
                    
                    break # Need more data
        
        # --- Tool call processing remains the same ---
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
                if not tool_name: continue
                try:
                    parsed_args = json.loads(args_str if args_str else "{}")
                    final_tool_inputs.append(
                        llm.ToolInput(id=tool_id, tool_name=tool_name, tool_args=parsed_args)
                    )
                except json.JSONDecodeError:
                    LOGGER.error("Failed to decode JSON for tool %s: %s", tool_name, args_str)
            if final_tool_inputs:
                yield {"tool_calls": final_tool_inputs}
            active_tool_calls_by_index.clear()

    # After the loop, yield any remaining valid content from the buffer
    if strip_think_tags and not is_in_think_block and buffer:
        yield {"content": buffer}


class OpenAICompatibleConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """OpenAI Compatible conversation agent."""
    _attr_has_entity_name = True
    _attr_name = None
    _attr_supports_streaming = True

    def __init__(self, entry: OpenAICompatibleConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="OpenAI Compatible",
            model="OpenAI Compatible",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
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
            self.entry.async_on_unload(
                self.entry.add_update_listener(self._async_entry_update_listener)
            )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        with (
            chat_session.async_get_chat_session(
                self.hass, user_input.conversation_id
            ) as session,
            conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self._async_handle_message(user_input, chat_log)

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the API using streaming."""
        assert user_input.agent_id
        options = self.entry.options
        client = self.entry.runtime_data

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        tools: list[ChatCompletionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                messages = [
                    _convert_content_to_param(content) for content in chat_log.content
                ]
            except Exception as err:
                LOGGER.error("Error during history regeneration: %s", err)
                raise HomeAssistantError(f"Failed to process history: {err}") from err

            # Original "No Think" logic remains here
            if _iteration == 0:
                no_think_enabled = options.get(CONF_NO_THINK, False)
                if no_think_enabled and messages and messages[-1]["role"] == "user":
                    user_message = messages[-1]
                    current_content = str(user_message.get("content", ""))
                    if not current_content.endswith("/no_think"):
                        user_message["content"] = current_content + "/no_think"
                        LOGGER.debug("Appended /no_think to user message.")

            base_url = str(getattr(self.entry.runtime_data, "base_url", ""))
            is_mistral = "mistral.ai" in base_url.lower()

            model_args: dict[str, Any] = {
                "model": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                "messages": messages,
                "tools": tools or NOT_GIVEN,
                "tool_choice": "auto" if tools else NOT_GIVEN,
                "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "stream": True,
            }
            if not is_mistral:
                model_args["user"] = chat_log.conversation_id

            try:
                response_stream = await client.chat.completions.create(**model_args)

                # Use the new option to control the filtering
                strip_tags = options.get(CONF_STRIP_THINK_TAGS, False)
                async for _ in chat_log.async_add_delta_content_stream(
                    user_input.agent_id,
                    _openai_to_ha_stream(response_stream, strip_think_tags=strip_tags),
                ):
                    pass

            except openai.RateLimitError as err:
                LOGGER.error("Rate limited by API: %s", err)
                raise HomeAssistantError("Rate limited or insufficient funds") from err
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

        if not chat_log.content or not isinstance(
            chat_log.content[-1], conversation.AssistantContent
        ):
            raise HomeAssistantError("Last message in chat log is not from the assistant.")

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(chat_log.content[-1].content or " ")
        return conversation.ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        await hass.config_entries.async_reload(entry.entry_id)
