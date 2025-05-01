# Open-Source Implementation of Deep Research w/ LangGraph
## STAT 5243 - Applied Data Science Bonus Project
## Team: Shayan Chowdhury, Anqi Wu, Thomas Bordino, Mei Yue, Fatih Uysal

**"Deep Research"** refers to AI-powered systems that autonomously conduct multi-step research by searching, analyzing, and synthesizing information from a wide range of sources to generate comprehensive, well-cited reports. Leading companies implementing similar deep research capabilities include OpenAI ([ChatGPT Deep Research](https://openai.com/index/introducing-deep-research/)), Google ([Gemini Deep Research](https://gemini.google/overview/deep-research/)), and Perplexity AI ([Perplexity Deep Research](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)), each offering advanced agentic workflows that leverage large language models for in-depth, expert-level analysis. 

For our bonus project, we chose to implement a deep research workflow using open-source language models deployed locally using Ollama. Our code is adapted from LangGraph's [implementation](https://github.com/langchain-ai/langchain/tree/main/examples/open_deep_research) of Deep Research but with significant refactoring and optimizations for the purposes of report generation using open-source language models deployed locally using Ollama. 

Main features:
- Using reasoning LLMs for report planning and reflection/grading to ensure each of the sections are well-researched and of high quality
- Allowing for human feedback and iteration on the report plan for greater flexibility (human-in-the-loop design)
- Web search integration with [Tavily](https://tavily.com/)
- Using [LangGraph](https://www.langchain.com/langgraph) for easier implementation of agentic workflows
- Parallel section writing for improved throughput and efficiency
- Memory-based checkpointing for partial runs
