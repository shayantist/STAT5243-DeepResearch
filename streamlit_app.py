import streamlit as st
import asyncio
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import tavily_search_async, deduplicate_and_format_sources
from prompts import (
    report_planner_instructions,
    report_planner_query_writer_instructions,
    section_writer_instructions,
    section_writer_inputs,
)
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------
# Helper Function
# ----------------------
def parse_report_plan_advanced(plan_text):
    """
    Extracts section titles from a structured report plan by finding lines 
    starting with 'Name:'. This version is robust to minor formatting variations.
    """
    pattern = r"[Nn]ame:\s*(.+)"
    matches = re.findall(pattern, plan_text)
    return [m.strip() for m in matches]




# ----------------------
# Streamlit Config
# ----------------------
st.set_page_config(page_title="Deep Research with LangGraph", layout="wide")

AVAILABLE_MODELS = ["llama3:8b", "qwen2:7b", "nous-hermes2", "yi:6b"]

defaults = {
    "step": 1,
    "topic": "",
    "structure": "",
    "section_topic": "",
    "section_name": "",
    "report_plan": None,
    "parsed_sections": [],
    "editable_sections": [],
    "section_queries": [],
    "search_results": [],
    "written_sections": [],
    "feedback_text": "",
    "feedback_submitted": False,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ----------------------
# Sidebar
# ----------------------
st.sidebar.title("Settings")
selected_model = st.sidebar.selectbox("Choose Model", AVAILABLE_MODELS)
ollama_model = ChatOllama(model=selected_model)

st.sidebar.markdown("---")
if st.sidebar.button("Next Step"):
    st.session_state.step += 1
if st.sidebar.button("Reset All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.update(defaults)
    st.rerun()

# ----------------------
# App Title
# ----------------------
st.title("Deep Research Report Generator")

st.markdown("""
Follow **5 steps** to build your custom research report:

1. **Define Topic & Structure**
2. **Parse & Edit Section Titles**
3. **Search & Fetch Materials**
4. **Write Sections**
5. **Preview & Export**

---
""")

# ----------------------
# Step 1: Define Topic and Report Plan
# ----------------------
if st.session_state.step >= 1:
    st.header("Step 1: Define Topic and Structure")

    st.text_input("Research Topic:", key="topic")
    st.text_area("Report Structure:", key="structure", height=150)

    if st.button("Generate Report Plan"):
        planner_prompt = PromptTemplate.from_template(report_planner_instructions)
        chain = planner_prompt | ollama_model | StrOutputParser()

        with st.spinner("Generating report plan..."):
            st.session_state.report_plan = chain.invoke({
                "topic": st.session_state.topic,
                "report_organization": st.session_state.structure,
                "context": "",
                "feedback": "",
            })

    if st.session_state.report_plan:
        st.subheader("Generated Report Plan")
        st.code(st.session_state.report_plan)

        st.markdown("## Feedback")
        st.text_area("Suggest improvements (optional):", key="feedback_text", height=100)

        if st.button("Submit Feedback"):
            planner_prompt = PromptTemplate.from_template(report_planner_instructions)
            chain = planner_prompt | ollama_model | StrOutputParser()

            with st.spinner("Updating report plan..."):
                st.session_state.report_plan = chain.invoke({
                    "topic": st.session_state.topic,
                    "report_organization": st.session_state.structure,
                    "context": "",
                    "feedback": st.session_state.feedback_text,
                })
            st.success("Updated! You can give more feedback or proceed.")
            st.rerun()

# ----------------------
# Step 2: Parse and Edit Sections
# ----------------------
if st.session_state.step >= 2:
    st.header("Step 2: Parse and Edit Section Titles")

    if st.session_state.report_plan and st.button("Parse Sections from Report Plan"):
        sections = parse_report_plan_advanced(st.session_state.report_plan)
        if sections:
            st.session_state.parsed_sections = sections
            st.session_state.editable_sections = sections.copy()
            st.success(f"✅ Parsed {len(sections)} section(s). You can now edit them.")
        else:
            st.warning("⚠️ No sections found. Please check the format of the Report Plan.")

    if st.session_state.editable_sections:
        st.subheader("Editable Section Titles")

        updated_sections = []
        for idx, title in enumerate(st.session_state.editable_sections):
            new_title = st.text_input(f"Section {idx+1} Title", value=title, key=f"section_edit_{idx}")
            updated_sections.append(new_title.strip())

        if st.button("Save Updated Section Titles"):
            st.session_state.section_queries = updated_sections
            st.success("✅ Section titles saved! You can now proceed to Search.")

# ----------------------
# Step 3: Search per Section
# ----------------------
if st.session_state.step >= 3:
    st.header("Step 3: Search Web Resources for Each Section")

    if st.button("Run Parallel Tavily Search"):
        async def search_all_queries():
            search_tasks = []
            searchable_indices = []
            filtered_queries = []

            # Identify which queries to search (exclude Intro/Conclusion)
            for idx, q in enumerate(st.session_state.section_queries):
                q_lower = q.strip().lower()
                if q_lower in {"introduction", "conclusion", "summary", "closing remarks", "references"}:
                    st.session_state.search_results.append(f"(Search skipped for section: {q})")
                else:
                    search_tasks.append(tavily_search_async([q]))
                    searchable_indices.append(idx)
                    filtered_queries.append(q)

            # Run searches in parallel
            results = await asyncio.gather(*search_tasks)
            combined = [deduplicate_and_format_sources(r, max_tokens_per_source=2000) for r in results]

            # Inject results back into the original order
            for idx, content in zip(searchable_indices, combined):
                st.session_state.search_results[idx] = content

        # Clear previous results and reinitialize the list
        st.session_state.search_results = [""] * len(st.session_state.section_queries)
        asyncio.run(search_all_queries())

    if st.session_state.search_results:
        st.subheader("Fetched Search Results")
        for idx, result in enumerate(st.session_state.search_results, 1):
            st.text_area(f"Search Result {idx}", result, height=300)

# ----------------------
# Step 4: Write Sections
# ----------------------
if st.session_state.step >= 4:
    st.header("Step 4: Write Report Sections")

    if st.button("Generate All Sections"):
        st.session_state.written_sections = []

        # Main writer prompt (uses context from search)
        full_writer_chain = (
            PromptTemplate.from_template(
                "System Instruction:\nYou must start fresh.\n\n" + section_writer_instructions
            ) | ollama_model | StrOutputParser()
        )

        # Fallback writer for intro/skipped sections
        fallback_prompt = PromptTemplate.from_template(
            """Write a concise, self-contained section titled "{section_topic}" for a research report on "{topic}".
Avoid relying on external references. Focus on providing context, clarity, and informative content suitable for an introductory or standalone section."""
        )
        fallback_writer_chain = fallback_prompt | ollama_model | StrOutputParser()

        # Dedicated conclusion generator (summarize prior content)
        conclusion_prompt = PromptTemplate.from_template(
            """Write a conclusion for a research report on "{topic}".
Summarize the key insights from the following sections:

{written_content}

End with a thoughtful remark on the significance of the topic and possible directions for further exploration."""
        )
        conclusion_chain = conclusion_prompt | ollama_model | StrOutputParser()

        progress = st.progress(0)

        for idx, (query, search_content) in enumerate(zip(st.session_state.section_queries, st.session_state.search_results)):
            section_title = query.strip()
            section_lower = section_title.lower()
            is_skipped = search_content.startswith("(Search skipped for section:")
            is_conclusion = section_lower in {"conclusion", "summary", "closing remarks"}

            with st.spinner(f"Writing Section {idx+1}..."):
                if is_conclusion:
                    # Use summary of previous written sections
                    previous_texts = "\n\n".join(st.session_state.written_sections)
                    section_text = conclusion_chain.invoke({
                        "topic": st.session_state.topic,
                        "written_content": previous_texts,
                    })
                    section_text = f"## Section {idx+1}: {section_title}\n\n{section_text}"
                    st.session_state.written_sections.append(section_text)
                elif is_skipped:
                    # Use fallback writer for intro-like sections
                    section_text = fallback_writer_chain.invoke({
                        "topic": st.session_state.topic,
                        "section_topic": section_title,
                    })
                    section_text = f"## Section {idx+1}: {section_title}\n\n{section_text}"
                    st.session_state.written_sections.append(section_text)
                elif search_content.strip():
                    # Normal case with context
                    section_text = full_writer_chain.invoke({
                        "topic": st.session_state.topic,
                        "section_name": f"Section {idx+1}: {section_title}",
                        "section_topic": section_title,
                        "section_content": "",
                        "context": search_content,
                        "SECTION_WORD_LIMIT": 500,
                    })
                    st.session_state.written_sections.append(section_text)
                else:
                    st.session_state.written_sections.append(f"## Section {idx+1}: {section_title}\n\n(No sufficient information found.)")
            
            progress.progress((idx + 1) / len(st.session_state.section_queries))

    if st.session_state.written_sections:
        st.subheader("Generated Sections")
        for idx, section in enumerate(st.session_state.written_sections, 1):
            st.markdown(f"### Section {idx}")
            st.markdown(section)


# ----------------------
# Step 5: Export Full Report
# ----------------------
if st.session_state.step >= 5:
    st.header("Step 5: Export Full Report")

    if st.session_state.written_sections:
        numbered_sections = []
        for idx, section in enumerate(st.session_state.written_sections, 1):
            numbered = section.replace("## ", f"## {idx}. ", 1)
            numbered_sections.append(numbered)

        final_md = f"# {st.session_state.topic}\n\n" + "\n\n".join(numbered_sections)

        st.subheader("Final Report Preview")
        st.text_area("Markdown Output", final_md, height=400)

        st.download_button("📄 Download Report", final_md, file_name="deep_research.md")
