# Open-Source Implementation of Deep Research
## STAT 5243 - Applied Data Science Bonus Project
## Team: Shayan Chowdhury, Anqi Wu, Thomas Bordino, Mei Yue, Fatih Uysal

**"Deep Research"** refers to AI-powered systems that autonomously conduct multi-step research by searching, analyzing, and synthesizing information from a wide range of sources to generate comprehensive, well-cited reports. Leading companies implementing similar deep research capabilities include OpenAI ([ChatGPT Deep Research](https://openai.com/index/introducing-deep-research/)), Google ([Gemini Deep Research](https://gemini.google/overview/deep-research/)), and Perplexity AI ([Perplexity Deep Research](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)), each offering advanced agentic workflows that leverage large language models for in-depth, expert-level analysis. 

For our bonus project, we chose to implement a deep research workflow using open-source language models deployed locally using Ollama. Our code is adapted from LangGraph's [implementation](https://github.com/langchain-ai/langchain/tree/main/examples/open_deep_research) of Deep Research but with significant refactoring and optimizations for the purposes of report generation using open-source language models deployed locally using Ollama. 

**Main features**:
- Using reasoning LLMs for report planning and reflection/grading to ensure each of the sections are well-researched and of high quality
- Allowing for human feedback and iteration on the report plan for greater flexibility (human-in-the-loop design)
- Web search integration
- Parallel section writing for improved throughput and efficiency
- Memory-based checkpointing for partial runs

**Architecture**
```text
STAT5243-DeepResearch/
├── .streamlit/
│   └── .secrets.toml 
├── streamlit_app_gemini.py
├── utils.py
├── prompts.py
├── deep_research_test.ipynb
├── requirements.txt
└── README.md
```

**Supported Search APIs**
- **[Tavily](https://www.tavily.com/)** 
- **[DuckDuckGo](https://duckduckgo.com/)** 
- **[Google Programmable Search](https://programmablesearchengine.google.com/about/)**
- **[GitHub API](https://docs.github.com/en/rest/search)**

**Supported Large Language Models (LLMs)**
**Local (via Ollama):**
- `llama3:8b`
- `qwen2:7b`
- `yi:6b`
- `nous-hermes2`
- For a full list of supported models, visit the [Ollama Model Library](https://ollama.com/library).

**Cloud (via Google Gemini):**
- `gemini-2.0-flash`

**Getting Started**
1. **Clone the repo**
```bash
git clone https://github.com/shayantist/STAT5243-DeepResearch.git
cd deep-research-assistant
```

2. **Set up virtual environment**
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure secrets**
- Create a file named .streamlit/secrets.toml with your API keys:
```bash
GOOGLE_SEARCH_API_KEY = "your_google_api_key"
GOOGLE_CSE_ID = "your_cse_id"
GITHUB_API_TOKEN = "your_github_token"
TAVILY_API_KEY = "your_tavily_key"
GEMINI_API_KEY = "your_google_gemini_key"
```

5. Usage
```bash
streamlit run streamlit_app_gemini.py
```

