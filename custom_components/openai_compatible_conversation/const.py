"""Constants for the OpenAI Compatible Conversation integration."""

import logging

DOMAIN = "openai_compatible_conversation"
LOGGER = logging.getLogger(__package__)

CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "mistral-small-latest"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 300
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0
CONF_BASE_URL = "base_url"
RECOMMENDED_BASE_URL = "https://api.mistral.ai/v1/"
CONF_NO_THINK = "no_think"
CONF_STRIP_THINK_TAGS = "strip_think_tags"