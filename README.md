<div align="center">
  <img src="https://github.com/The-Pocket/.github/raw/main/assets/title.png" alt="Pocket Flow – 100-line minimalist LLM framework" width="600"/>
</div>

> **Note**: This is a fork of the original [PocketFlow](https://github.com/The-Pocket/PocketFlow) by [Zachary Huang](https://github.com/ZacharyL2) and [The Pocket](https://github.com/The-Pocket). All credit for the core framework goes to them.

> This fork adds extra utilities, including a [Universal LLM API Client](#-additional-feature-universal-llm-api-client) for easy access to 8+ LLM providers.

<!-- For translation, replace English with [English](https://github.com/The-Pocket/PocketFlow/blob/main/README.md), and remove the link for the target language. -->

English | [中文](https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-batch/translations/README_CHINESE.md) | [Español](https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-batch/translations/README_SPANISH.md) | [日本語](https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-batch/translations/README_JAPANESE.md) | [Deutsch](https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-batch/translations/README_GERMAN.md) | [Русский](https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-batch/translations/README_RUSSIAN.md) | [Português](https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-batch/translations/README_PORTUGUESE.md) | [Français](https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-batch/translations/README_FRENCH.md) | [한국어](https://github.com/The-Pocket/PocketFlow/blob/main/cookbook/pocketflow-batch/translations/README_KOREAN.md)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://the-pocket.github.io/PocketFlow/)
<a href="https://discord.gg/hUHHE9Sa6T">
<img src="https://img.shields.io/discord/1346833819172601907?logo=discord&style=flat">
</a>

Pocket Flow is a [100-line](https://github.com/The-Pocket/PocketFlow/blob/main/pocketflow/__init__.py) minimalist LLM framework

- **Lightweight**: Just 100 lines. Zero bloat, zero dependencies, zero vendor lock-in.

- **Expressive**: Everything you love—([Multi-](https://the-pocket.github.io/PocketFlow/design_pattern/multi_agent.html))[Agents](https://the-pocket.github.io/PocketFlow/design_pattern/agent.html), [Workflow](https://the-pocket.github.io/PocketFlow/design_pattern/workflow.html), [RAG](https://the-pocket.github.io/PocketFlow/design_pattern/rag.html), and more.

- **[Agentic Coding](https://zacharyhuang.substack.com/p/agentic-coding-the-most-fun-way-to)**: Let AI Agents (e.g., Cursor AI) build Agents—10x productivity boost!

Get started with Pocket Flow:

- To install, `pip install pocketflow`or just copy the [source code](https://github.com/The-Pocket/PocketFlow/blob/main/pocketflow/__init__.py) (only 100 lines).
- To learn more, check out the [video tutorial](https://youtu.be/0Zr3NwcvpA0) and [documentation](https://the-pocket.github.io/PocketFlow/)
- 🎉 Join our [Discord](https://discord.gg/hUHHE9Sa6T) to connect with other developers building with Pocket Flow!
- 🎉 Pocket Flow now has [Typescript](https://github.com/The-Pocket/PocketFlow-Typescript), [Java](https://github.com/The-Pocket/PocketFlow-Java), [C++](https://github.com/The-Pocket/PocketFlow-CPP), [Go](https://github.com/The-Pocket/PocketFlow-Go), [Rust](https://github.com/The-Pocket/PocketFlow-Rust) and [PHP](https://github.com/The-Pocket/PocketFlow-PHP) versions!

## Why Pocket Flow?

Current LLM frameworks are bloated... You only need 100 lines for LLM Framework!

<div align="center">
  <img src="https://github.com/The-Pocket/.github/raw/main/assets/meme.jpg" width="400"/>

|                | **Abstraction** |                     **App-Specific Wrappers**                      |                       **Vendor-Specific Wrappers**                       |                **Lines**                 |                  **Size**                   |
| -------------- | :-------------: | :----------------------------------------------------------------: | :----------------------------------------------------------------------: | :--------------------------------------: | :-----------------------------------------: |
| LangChain      |  Agent, Chain   |      Many <br><sup><sub>(e.g., QA, Summarization)</sub></sup>      |      Many <br><sup><sub>(e.g., OpenAI, Pinecone, etc.)</sub></sup>       |                   405K                   |                   +166MB                    |
| CrewAI         |  Agent, Chain   | Many <br><sup><sub>(e.g., FileReadTool, SerperDevTool)</sub></sup> | Many <br><sup><sub>(e.g., OpenAI, Anthropic, Pinecone, etc.)</sub></sup> |                   18K                    |                   +173MB                    |
| SmolAgent      |      Agent      |   Some <br><sup><sub>(e.g., CodeAgent, VisitWebTool)</sub></sup>   |  Some <br><sup><sub>(e.g., DuckDuckGo, Hugging Face, etc.)</sub></sup>   |                    8K                    |                   +198MB                    |
| LangGraph      |  Agent, Graph   |       Some <br><sup><sub>(e.g., Semantic Search)</sub></sup>       | Some <br><sup><sub>(e.g., PostgresStore, SqliteSaver, etc.) </sub></sup> |                   37K                    |                    +51MB                    |
| AutoGen        |      Agent      |   Some <br><sup><sub>(e.g., Tool Agent, Chat Agent)</sub></sup>    | Many <sup><sub>[Optional]<br> (e.g., OpenAI, Pinecone, etc.)</sub></sup> | 7K <br><sup><sub>(core-only)</sub></sup> | +26MB <br><sup><sub>(core-only)</sub></sup> |
| **PocketFlow** |    **Graph**    |                              **None**                              |                                 **None**                                 |                 **100**                  |                  **+56KB**                  |

</div>

## How does Pocket Flow work?

The [100 lines](https://github.com/The-Pocket/PocketFlow/blob/main/pocketflow/__init__.py) capture the core abstraction of LLM frameworks: Graph!
<br>

<div align="center">
  <img src="https://github.com/The-Pocket/.github/raw/main/assets/abstraction.png" width="900"/>
</div>
<br>

From there, it's easy to implement popular design patterns like ([Multi-](https://the-pocket.github.io/PocketFlow/design_pattern/multi_agent.html))[Agents](https://the-pocket.github.io/PocketFlow/design_pattern/agent.html), [Workflow](https://the-pocket.github.io/PocketFlow/design_pattern/workflow.html), [RAG](https://the-pocket.github.io/PocketFlow/design_pattern/rag.html), etc.
<br>

<div align="center">
  <img src="https://github.com/The-Pocket/.github/raw/main/assets/design.png" width="900"/>
</div>
<br>
✨ Below are basic tutorials:

<div align="center">

|                                                      Name                                                       |        Difficulty         | Description                                                               |
| :-------------------------------------------------------------------------------------------------------------: | :-----------------------: | :------------------------------------------------------------------------ |
|               [Chat](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-chat)               |  ☆☆☆ <sup>_Dummy_</sup>   | A basic chat bot with conversation history                                |
|  [Structured Output](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-structured-output)  |  ☆☆☆ <sup>_Dummy_</sup>   | Extracting structured data from resumes by prompting                      |
|           [Workflow](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-workflow)           |  ☆☆☆ <sup>_Dummy_</sup>   | A writing workflow that outlines, writes content, and applies styling     |
|              [Agent](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-agent)              |  ☆☆☆ <sup>_Dummy_</sup>   | A research agent that can search the web and answer questions             |
|                [RAG](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-rag)                |  ☆☆☆ <sup>_Dummy_</sup>   | A simple Retrieval-augmented Generation process                           |
|              [Batch](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-batch)              |  ☆☆☆ <sup>_Dummy_</sup>   | A batch processor that translates markdown into multiple languages        |
|        [Streaming](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-llm-streaming)        |  ☆☆☆ <sup>_Dummy_</sup>   | A real-time LLM streaming demo with user interrupt capability             |
|     [Chat Guardrail](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-chat-guardrail)     |  ☆☆☆ <sup>_Dummy_</sup>   | A travel advisor chatbot that only processes travel-related queries       |
|      [Majority Vote](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-majority-vote)      |  ☆☆☆ <sup>_Dummy_</sup>   | Improve reasoning accuracy by aggregating multiple solution attempts      |
|         [Map-Reduce](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-map-reduce)         |  ☆☆☆ <sup>_Dummy_</sup>   | Batch resume qualification using map-reduce pattern                       |
|           [CLI HITL](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-cli-hitl)           |  ☆☆☆ <sup>_Dummy_</sup>   | A command-line joke generator with human-in-the-loop feedback             |
|        [Multi-Agent](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-multi-agent)        | ★☆☆ <sup>_Beginner_</sup> | A Taboo word game for async communication between 2 agents                |
|         [Supervisor](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-supervisor)         | ★☆☆ <sup>_Beginner_</sup> | Research agent is getting unreliable... Let's build a supervision process |
|        [Parallel](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-parallel-batch)        | ★☆☆ <sup>_Beginner_</sup> | A parallel execution demo that shows 3x speedup                           |
|   [Parallel Flow](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-parallel-batch-flow)   | ★☆☆ <sup>_Beginner_</sup> | A parallel image processing showing 8x speedup                            |
|           [Thinking](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-thinking)           | ★☆☆ <sup>_Beginner_</sup> | Solve complex reasoning problems through Chain-of-Thought                 |
|          [Memory](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-chat-memory)           | ★☆☆ <sup>_Beginner_</sup> | A chat bot with short-term and long-term memory                           |
|           [Text2SQL](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-text2sql)           | ★☆☆ <sup>_Beginner_</sup> | Convert natural language to SQL queries with an auto-debug loop           |
|     [Code Generator](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-code-generator)     | ★☆☆ <sup>_Beginner_</sup> | Generate test cases, implement solutions, and iteratively improve code    |
|                [MCP](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-mcp)                | ★☆☆ <sup>_Beginner_</sup> | Agent using Model Context Protocol for numerical operations               |
|                [A2A](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-a2a)                | ★☆☆ <sup>_Beginner_</sup> | Agent wrapped with A2A protocol for inter-agent communication             |
|      [Streamlit FSM](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-streamlit-fsm)      | ★☆☆ <sup>_Beginner_</sup> | Streamlit app with finite state machine for HITL image generation         |
|  [FastAPI WebSocket](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-fastapi-websocket)  | ★☆☆ <sup>_Beginner_</sup> | Real-time chat interface with streaming LLM responses via WebSocket       |
| [FastAPI Background](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-fastapi-background) | ★☆☆ <sup>_Beginner_</sup> | FastAPI app with background jobs and real-time progress via SSE           |
|         [Voice Chat](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-voice-chat)         | ★☆☆ <sup>_Beginner_</sup> | An interactive voice chat application with VAD, STT, LLM, and TTS.        |

</div>

👀 Want to see other tutorials for dummies? [Create an issue!](https://github.com/The-Pocket/PocketFlow/issues/new)

## How to Use Pocket Flow?

🚀 Through **Agentic Coding**—the fastest LLM App development paradigm-where _humans design_ and _agents code_!

<br>
<div align="center">
  <a href="https://zacharyhuang.substack.com/p/agentic-coding-the-most-fun-way-to" target="_blank">
    <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F423a39af-49e8-483b-bc5a-88cc764350c6_1050x588.png" width="700" alt="IMAGE ALT TEXT" style="cursor: pointer;">
  </a>
</div>
<br>

✨ Below are examples of more complex LLM Apps:

<div align="center">

|                                                                                           App Name                                                                                           |     Difficulty      |                                                                                                                          Topics                                                                                                                          |                                                  Human Design                                                  |                                               Agent Code                                               |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
|            [Website Chatbot](https://github.com/The-Pocket/PocketFlow-Tutorial-Website-Chatbot) <br> <sup><sub>Turn your website into a 24/7 customer support genius</sup></sub>             |  ★★☆ <br> _Medium_  |                                                  [Agent](https://the-pocket.github.io/PocketFlow/design_pattern/agent.html) <br> [RAG](https://the-pocket.github.io/PocketFlow/design_pattern/rag.html)                                                  |    [Design Doc](https://github.com/The-Pocket/PocketFlow-Tutorial-Website-Chatbot/blob/main/docs/design.md)    |    [Flow Code](https://github.com/The-Pocket/PocketFlow-Tutorial-Website-Chatbot/blob/main/flow.py)    |
| [Danganronpa Simulator](https://github.com/The-Pocket/PocketFlow-Tutorial-Danganronpa-Simulator) <br> <sup><sub>Forget the Turing test. Danganronpa, the ultimate AI experiment!</sup></sub> | ★★★ <br> _Advanced_ |                                             [Workflow](https://the-pocket.github.io/PocketFlow/design_pattern/workflow.html) <br> [Agent](https://the-pocket.github.io/PocketFlow/design_pattern/agent.html)                                             | [Design Doc](https://github.com/The-Pocket/PocketFlow-Tutorial-Danganronpa-Simulator/blob/main/docs/design.md) | [Flow Code](https://github.com/The-Pocket/PocketFlow-Tutorial-Danganronpa-Simulator/blob/main/flow.py) |
|          [Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge) <br> <sup><sub>Life's too short to stare at others' code in confusion</sup></sub>           |  ★★☆ <br> _Medium_  |                                                                                     [Workflow](https://the-pocket.github.io/PocketFlow/design_pattern/workflow.html)                                                                                     |        [Design Doc](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge/blob/main/docs/design.md)        |        [Flow Code](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge/blob/main/flow.py)        |
|                          [Build Cursor with Cursor](https://github.com/The-Pocket/Tutorial-Cursor) <br> <sup><sub>We'll reach the singularity soon ...</sup></sub>                           | ★★★ <br> _Advanced_ |                                                                                        [Agent](https://the-pocket.github.io/PocketFlow/design_pattern/agent.html)                                                                                        |              [Design Doc](https://github.com/The-Pocket/Tutorial-Cursor/blob/main/docs/design.md)              |              [Flow Code](https://github.com/The-Pocket/Tutorial-Cursor/blob/main/flow.py)              |
|                       [Ask AI Paul Graham](https://github.com/The-Pocket/Tutorial-YC-Partner) <br> <sup><sub>Ask AI Paul Graham, in case you don't get in</sup></sub>                        |  ★★☆ <br> _Medium_  | [RAG](https://the-pocket.github.io/PocketFlow/design_pattern/rag.html) <br> [Map Reduce](https://the-pocket.github.io/PocketFlow/design_pattern/mapreduce.html) <br> [TTS](https://the-pocket.github.io/PocketFlow/utility_function/text_to_speech.html) |          [Design Doc](https://github.com/The-Pocket/Tutorial-AI-Paul-Graham/blob/main/docs/design.md)          |          [Flow Code](https://github.com/The-Pocket/Tutorial-AI-Paul-Graham/blob/main/flow.py)          |
|                  [Youtube Summarizer](https://github.com/The-Pocket/Tutorial-Youtube-Made-Simple) <br> <sup><sub> Explain YouTube Videos to you like you're 5 </sup></sub>                   | ★☆☆ <br> _Beginner_ |                                                                                   [Map Reduce](https://the-pocket.github.io/PocketFlow/design_pattern/mapreduce.html)                                                                                    |       [Design Doc](https://github.com/The-Pocket/Tutorial-Youtube-Made-Simple/blob/main/docs/design.md)        |       [Flow Code](https://github.com/The-Pocket/Tutorial-Youtube-Made-Simple/blob/main/flow.py)        |
|             [Cold Opener Generator](https://github.com/The-Pocket/Tutorial-Cold-Email-Personalization) <br> <sup><sub> Instant icebreakers that turn cold leads hot </sup></sub>             | ★☆☆ <br> _Beginner_ |                                      [Map Reduce](https://the-pocket.github.io/PocketFlow/design_pattern/mapreduce.html) <br> [Web Search](https://the-pocket.github.io/PocketFlow/utility_function/websearch.html)                                      |   [Design Doc](https://github.com/The-Pocket/Tutorial-Cold-Email-Personalization/blob/master/docs/design.md)   |   [Flow Code](https://github.com/The-Pocket/Tutorial-Cold-Email-Personalization/blob/master/flow.py)   |

</div>

- Want to learn **Agentic Coding**?

  - Check out [my YouTube](https://www.youtube.com/@ZacharyLLM?sub_confirmation=1) for video tutorial on how some apps above are made!

  - Want to build your own LLM App? Read this [post](https://zacharyhuang.substack.com/p/agentic-coding-the-most-fun-way-to)! Start with [this template](https://github.com/The-Pocket/PocketFlow-Template-Python)!

---

## 🆕 Additional Feature: Universal LLM API Client

This fork includes an enhanced `utils/llm_api.py` module that provides a unified interface for multiple LLM providers:

### Features

- **Multiple Providers**: OpenAI, Azure OpenAI, Anthropic, Google Gemini, DeepSeek, SiliconFlow, Local models, and Ollama
- **Image Support**: Attach images to prompts across compatible providers
- **Easy Configuration**: Simple environment variable setup with `.env` support
- **Consistent Interface**: Same API regardless of the underlying provider

### Quick Start

```python
from utils.llm_api import LLMClient

# Simple query with any provider
llm = LLMClient(provider="openai")
response = llm.query("What's the meaning of life?")

# Query with image (supported providers)
llm = LLMClient(provider="openai")
response = llm.query("Describe this image", image_path="photo.jpg")

# Backward compatibility with existing code
llm = LLMClient(provider="ollama")
messages = [{"role": "user", "content": "Hello!"}]
response = llm.call(messages=messages)
```

### Supported Providers

| Provider      | Model             | Notes               |
| ------------- | ----------------- | ------------------- |
| `openai`      | GPT-4o            | Supports images     |
| `anthropic`   | Claude 3.5 Sonnet | Supports images     |
| `gemini`      | Gemini 2.0 Flash  | Supports images     |
| `azure`       | Azure OpenAI      | Enterprise solution |
| `deepseek`    | DeepSeek Chat     | Cost-effective      |
| `siliconflow` | DeepSeek R1       | High performance    |
| `local`       | Self-hosted       | Custom endpoint     |
| `ollama`      | Local models      | Privacy-first       |

### Environment Setup

Create a `.env` file with your API keys:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google Gemini
GOOGLE_API_KEY=your_google_api_key

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_MODEL_DEPLOYMENT=gpt-4o-ms

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key

# SiliconFlow
SILICONFLOW_API_KEY=your_siliconflow_api_key
```

This makes it incredibly easy to switch between different LLM providers in your PocketFlow applications!
