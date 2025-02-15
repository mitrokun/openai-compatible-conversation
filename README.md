# OpenAI Compatible Conversation
A copy of Home Assistant's built-in OpenAI Conversation Agent, with support for changing the base URL. Only the minimal changes to make this a standalone custom component capable of supporting a different base URL to make it compatible with other services offering an OpenAI-compatible API were  made.

## Note about changing the model

If you're not using OpenAI models, then you need to change the default values that this integration will send to the API. To make that happen, click on "configure" on this integration and de-select the "recommended" toggle and click on "send". A placeholder to change the default model will appear and you'll be able to change it to any model you want.
