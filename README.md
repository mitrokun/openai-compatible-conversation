
> [!NOTE]
> A note from the maintainer: This project was initially created as a fork of Home Assistant's built-in OpenAI Conversation Agent that simply had the added ability to change the base URL. It was intended to be a point-in-time snapshot becuase I created it as the first step of another project, and I was surprised it didn't already exist. As Home Assistant implemented new features for their own OpenAI Conversation agent, it was trivial to integrate them, because the codebase was practically identical. Unfortunately, with the recent move to the Responses API, which many "OpenAI-compatible" APIs do not support, this project has diverged from Home Assistant's built-in OpenAI Conversation Agent, and that requires additional development and testing time. I personally cannot support this, as I don't actually use this integration, which makes ensuring its quality challenging. I have developed and personally use [Custom Conversation](https://github.com/michelle-avery/custom-conversation), which you may want to check out - it may be overkill for your needs, as it's intended for those who want to control and experiment with various aspects of LLM conversation agents, but it does support multiple providers (including OpenAI, Gemini, Mistral, and OpenRouter), as well as streaming, and the ability to fallback to a secondary provider once the primary provider has exhausted its quota. I'm also happy to add additional providers there, as the underlying code is designed to be extensible. If anyone is intersted in picking up this project and maintaining it while adding support for the quirks of different providers, please reach out to me, and I'll be happy to transfer ownership.

# OpenAI Compatible Conversation
This project started off as a copy of Home Assistant's built-in OpenAI Conversation Agent, with support for changing the base URL. Only the minimal changes to make this a standalone custom component capable of supporting a different base URL to make it compatible with other services offering an OpenAI-compatible API were  made.

As development on Home Assistant's built-in OpenAI Conversation Agent has progressed, more features have been added that are OpenAI specific and less compatible with other providers that offer an OpenAI compatible API. Due to this, this project does have the following limitations:

* OpenAI's reasoning parameters are not supported.
* Responses are not streamed to the chat log.
* The project currently continues to use the `max_tokens` parameter as opposed to the newer `max_completion_tokens` parameter for backwards compatibility.

## Note about changing the model

If you're not using OpenAI models, then you need to change the default values that this integration will send to the API. To make that happen, click on "configure" on this integration and de-select the "recommended" toggle and click on "send". A placeholder to change the default model will appear and you'll be able to change it to any model you want.
