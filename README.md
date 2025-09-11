## In this fork:
* Streaming support for LLM
* Fix for Mistral AI
* Mistral Vision action
* Web serch agent

# OpenAI Compatible Conversation
This project started off as a copy of Home Assistant's built-in OpenAI Conversation Agent, with support for changing the base URL. Only the minimal changes to make this a standalone custom component capable of supporting a different base URL to make it compatible with other services offering an OpenAI-compatible API were  made.

As development on Home Assistant's built-in OpenAI Conversation Agent has progressed, more features have been added that are OpenAI specific and less compatible with other providers that offer an OpenAI compatible API. Due to this, this project does have the following limitations:

* OpenAI's reasoning parameters are not supported.
* The project currently continues to use the `max_tokens` parameter as opposed to the newer `max_completion_tokens` parameter for backwards compatibility.

## Note about changing the model

The standard model set for use with MistralAI is 'mistral-small-latest'; it is likely that you will need to change this value. To make that happen, click on "configure" on this integration and de-select the "recommended" toggle and click on "send". A placeholder to change the default model will appear and you'll be able to change it to any model you want.
