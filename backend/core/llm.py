import os
import json
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


def parse_ai_json(response_text: str):
    """Parse AI response that may contain markdown code fences."""
    clean = response_text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
    return json.loads(clean)


class LlmChat:
    def __init__(self, system_message: str):
        self.system_message = system_message
        self.history = []

    async def send_message(self, text: str, image_base64: str = None, image_type: str = "image/jpeg") -> str:
        if image_base64:
            content = [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_base64}"}}
            ]
        else:
            content = text

        self.history.append({"role": "user", "content": content})
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": self.system_message}] + self.history,
            max_tokens=4000
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply


async def create_llm_chat(session_prefix: str, system_message: str) -> LlmChat:
    """Create and configure an LLM chat instance."""
    return LlmChat(system_message=system_message)
