import argparse
import base64
import mimetypes
import os
import sys
from typing import Optional

import google.generativeai as genai
import ollama
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()

class LLMClient:
    def __init__(self, provider="openai", model=None):
        self.provider = provider
        self.client = self._create_client(provider)
        self.model = model or self._default_model(provider)

    @staticmethod
    def encode_image_file(image_path: str) -> tuple[str, str]:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = 'image/png'
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string, mime_type

    @staticmethod
    def _default_model(provider):
        models = {
            "openai": "gpt-4o",
            "azure": os.getenv('AZURE_OPENAI_MODEL_DEPLOYMENT', 'gpt-4o-ms'),
            "deepseek": "deepseek-chat",
            "siliconflow": "deepseek-ai/DeepSeek-R1",
            "anthropic": "claude-3-5-sonnet-20241022",
            "gemini": "gemini-2.0-flash-exp",
            "local": "Qwen/Qwen2.5-32B-Instruct-AWQ",
            "ollama": "qwen3:0.6b"
        }
        return models.get(provider, "gpt-4o")

    def _create_client(self, provider):
        if provider == "openai":
            return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif provider == "azure":
            return AzureOpenAI(
                api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                api_version="2024-08-01-preview",
                azure_endpoint="https://msopenai.openai.azure.com"
            )
        elif provider == "deepseek":
            return OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com/v1"
            )
        elif provider == "siliconflow":
            return OpenAI(
                api_key=os.getenv('SILICONFLOW_API_KEY'),
                base_url="https://api.siliconflow.cn/v1"
            )
        elif provider == "anthropic":
            return Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        elif provider == "gemini":
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            return genai
        elif provider == "local":
            return OpenAI(
                base_url="http://192.168.180.137:8006/v1",
                api_key="not-needed"
            )
        elif provider == "ollama":
            return None
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def query(self, prompt: str, image_path: Optional[str] = None) -> Optional[str]:
        try:
            if self.provider == "ollama":
                messages = [{"role": "user", "content": prompt}]
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": 0.7}
                )
                return response['message']['content']

            elif self.provider in ["openai", "local", "deepseek", "azure", "siliconflow"]:
                messages = [{"role": "user", "content": prompt}]
                if image_path and self.provider == "openai":
                    encoded_image, mime_type = self.encode_image_file(image_path)
                    messages[0]["content"] = [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
                    ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7
                )
                return response.choices[0].message.content

            elif self.provider == "anthropic":
                content = [{"type": "text", "text": prompt}]
                if image_path:
                    encoded_image, mime_type = self.encode_image_file(image_path)
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": encoded_image
                        }
                    })
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": content}]
                )
                return response.content[0].text

            elif self.provider == "gemini":
                model_instance = self.client.GenerativeModel(self.model)
                if image_path:
                    file = genai.upload_file(image_path, mime_type="image/png")
                    response = model_instance.generate_content([prompt, file])
                else:
                    response = model_instance.generate_content(prompt)
                return response.text

        except Exception as e:
            print(f"Error querying LLM: {e}", file=sys.stderr)
            return None

    def call(self, messages=None):
        if messages and len(messages) > 0:
            prompt = messages[-1].get("content", "")
            return self.query(prompt)
        return None

def main():
    parser = argparse.ArgumentParser(description='Query an LLM with a prompt')
    parser.add_argument('--prompt', type=str, help='The prompt to send to the LLM', required=True)
    parser.add_argument('--provider', choices=['openai','anthropic','gemini','local','deepseek','azure','siliconflow','ollama'], default='ollama', help='The API provider to use')
    parser.add_argument('--model', type=str, help='The model to use (default depends on provider)')
    parser.add_argument('--image', type=str, help='Path to an image file to attach to the prompt')
    args = parser.parse_args()

    llm = LLMClient(provider=args.provider, model=args.model)
    response = llm.query(args.prompt, image_path=args.image)
    if response:
        print(response)
    else:
        print("Failed to get response from LLM")

if __name__ == "__main__":
    messages = [{"role": "user", "content": "In a few words, what's the meaning of life?"}]
    llm = LLMClient(provider="ollama")
    response = llm.call(messages=messages)
    print(f"Prompt: {messages[0]['content']}")
    print(f"Response: {response}")

    if len(sys.argv) > 1:
        main()
