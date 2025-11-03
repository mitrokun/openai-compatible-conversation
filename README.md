>[!CAUTION]
>The new version does not support old entries. Before updating the component, save the agent settings data if required.

The transition to subentry is complete. It is now convenient to create multiple agents for a single provider.

## In this fork:
* Streaming support for LLM
* Fix for Mistral AI
* Mistral Vision action
* Mistral Web serch agent
* Timestamp fixation for each conversation session

# OpenAI Compatible Conversation
This project started off as a copy of Home Assistant's built-in OpenAI Conversation Agent, with support for changing the base URL. Only the minimal changes to make this a standalone custom component capable of supporting a different base URL to make it compatible with other services offering an OpenAI-compatible API were  made.

As development on Home Assistant's built-in OpenAI Conversation Agent has progressed, more features have been added that are OpenAI specific and less compatible with other providers that offer an OpenAI compatible API. Due to this, this project does have the following limitations:

* OpenAI's reasoning parameters are not supported.
* The project currently continues to use the `max_tokens` parameter as opposed to the newer `max_completion_tokens` parameter for backwards compatibility.

### Note about changing the model

The standard model set for use with MistralAI is 'mistral-small-latest'; it is likely that you will need to change this value.
