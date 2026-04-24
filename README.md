# langchain_deepseek_v4

LangChain chat model integration for the current DeepSeek Chat Completions API.

Install the distribution package and import the integration module:

```bash
pip install langchain_deepseek_v4
```

This implementation targets the DeepSeek API documented at
<https://api-docs.deepseek.com>, including:

- Tool calling
- JSON structured output through DeepSeek JSON Output
- Thinking mode and `reasoning_content`
- Streaming chunks that preserve reasoning deltas
- Multi-turn conversations that pass tool-call reasoning back to DeepSeek

```python
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-v4-pro",
    api_key="...",
    thinking={"type": "enabled"},
    reasoning_effort="high",
)

msg = llm.invoke("Say hello in JSON: {\"message\": \"...\"}")
```
