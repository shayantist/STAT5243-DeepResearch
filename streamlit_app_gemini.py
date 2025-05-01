import streamlit as st
import asyncio
import os
import re
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import tavily_search_async, deduplicate_and_format_sources
from prompts import (
    report_planner_instructions,
    report_planner_query_writer_instructions,
    section_writer_instructions,
    section_writer_inputs,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

# from dotenv import load_dotenv
# load_dotenv()

# ----------------------
# Utility Function
# ----------------------
def parse_report_plan_advanced(plan_input):
    """
    Extracts section titles from either a structured dict/list (tool-generated) 
    or from a plain text report plan.

    Args:
        plan_input (Union[str, List[dict]]): Either a text-based plan or a list of section dicts.

    Returns:
        List[str]: A list of parsed section titles.
    """
    if isinstance(plan_input, list) and all(isinstance(item, dict) and "Name" in item for item in plan_input):
        # Handle structured dict input (tool-style)
        return [item["Name"].strip() for item in plan_input]
    
    elif isinstance(plan_input, str):
        # Handle plain text input (ChatGPT-style plan)
        pattern = r"[Nn]ame:\s*(.+)"
        matches = re.findall(pattern, plan_input)
        return [m.strip() for m in matches]

    else:
        raise ValueError("Invalid report plan format: must be either a list of dicts or a formatted string.")


def load_structured_plan():
    """
    Load a structured report plan from either pasted JSON or uploaded .json file.
    Parses it and stores section titles in session state.
    """

    st.markdown("Load Structured Report Plan")

    # Developer paste input
    with st.expander("Paste JSON Plan"):
        raw_json = st.text_area("Paste your JSON-formatted section list here:", height=250)
        if st.button("Parse from JSON Text"):
            try:
                parsed_obj = json.loads(raw_json)
                section_titles = parse_report_plan_advanced(parsed_obj)
                st.session_state.parsed_sections = section_titles
                st.session_state.editable_sections = section_titles.copy()
                st.success(f"Parsed {len(section_titles)} section(s) from pasted JSON.")
                return
            except Exception as e:
                st.error(f"Failed to parse JSON: {e}")

    # Upload file
    uploaded = st.file_uploader("Or upload a .json file", type="json")
    if uploaded:
        try:
            uploaded_plan = json.load(uploaded)
            section_titles = parse_report_plan_advanced(uploaded_plan)
            st.session_state.parsed_sections = section_titles
            st.session_state.editable_sections = section_titles.copy()
            st.success(f"Parsed {len(section_titles)} section(s) from uploaded file.")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

# ----------------------
# Streamlit Config
# ----------------------
st.set_page_config(page_title="Deep Research with LangGraph", layout="wide")

AVAILABLE_MODELS = ["llama3:8b (ollama)", "qwen2:7b (ollama)", "nous-hermes2 (ollama)", "yi:6b (ollama)", "gemini-2.0-flash (google)"]

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

if 'ollama' in selected_model:
    selected_model = selected_model.split(' (')[0]
    ollama_model = ChatOllama(model=selected_model)
elif 'google' in selected_model:
    selected_model = selected_model.split(' (')[0]
    ollama_model = ChatGoogleGenerativeAI(model=selected_model, google_api_key=st.secrets["GEMINI_API_KEY"])

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

1. **Define Topic & Structure**: Enter your research topic and outline the desired structure for your report.
2. **Parse & Edit Section Titles**: Review and customize the automatically generated section titles.
3. **Search & Fetch Materials**: The system searches the web for relevant information for each section.
4. **Write Sections**: AI generates detailed content for each section using the gathered materials.
5. **Preview & Export**: Review the complete report and export it in Markdown format.

---
""")

# ----------------------
# Step 1: Define Topic and Report Plan
# ----------------------
if st.session_state.step >= 1:
    st.header("Step 1: Define Topic and Structure")
    
    st.markdown("""
    **Instructions:**
    - Enter a specific research topic or question in the left panel
    - Provide a detailed structure for your report with main sections and subsections
    - Click "Generate Report Plan" when you're ready to proceed
    - You can provide feedback and regenerate the plan if needed
    """)
    
    # Create two columns for side-by-side layout
    left_col, right_col = st.columns(2)
    
    # Left column - User input
    with left_col:
        st.markdown("### Your Input")
        st.markdown("**Research Topic:**")
        st.markdown("*Be specific and focused on a particular area of research*")
        st.text_input("Enter your research topic:", key="topic", label_visibility="collapsed")
        
        st.markdown("**Report Structure:**")
        st.markdown("*Include main sections and subsections with clear numbering*")
        st.text_area("Enter your report structure:", key="structure", height=300, label_visibility="collapsed")
        
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
    
    # Right column - Examples
    with right_col:
        st.markdown("### Examples")
        st.markdown("**Research Topic Example:**")
        st.markdown("*Impact of Artificial Intelligence on Healthcare Diagnostics*")
        
        st.markdown("**Report Structure Example:**")
        st.markdown("""
        ```
        1. Introduction
           1.1. Background - Overview of AI in healthcare diagnostics and its significance
           1.2. Research Objectives - Clear statement of what the research on AI in healthcare diagnostics aims to achieve
           1.3. Scope and Limitations - Boundaries of the research on AI's impact in healthcare diagnostics
        2. Literature Review
           2.1. Historical Context - Evolution of AI applications in healthcare diagnostics
           2.2. Current State of Research - Recent studies on AI in diagnostic medicine
           2.3. Theoretical Framework - Conceptual models for AI implementation in healthcare diagnostics
        3. Methodology
           3.1. Research Design - Approach to studying AI's impact on healthcare diagnostics
           3.2. Data Collection - Sources and procedures for gathering information on AI diagnostic systems
           3.3. Analysis Techniques - Methods for evaluating AI's effectiveness in diagnostics
        4. Findings
           4.1. Primary Results - Key discoveries about AI's performance in healthcare diagnostics
           4.2. Statistical Analysis - Quantitative evaluation of AI diagnostic accuracy
           4.3. Comparative Analysis - Comparison of AI vs. traditional diagnostic methods
        5. Discussion
           5.1. Interpretation of Results - Meaning of AI's impact on diagnostic processes
           5.2. Implications - Practical applications of AI in healthcare diagnostics
           5.3. Limitations - Constraints in current AI diagnostic systems
        6. Conclusion
           6.1. Summary of Findings - Recap of AI's impact on healthcare diagnostics
           6.2. Recommendations - Suggestions for improving AI diagnostic systems
           6.3. Future Research Directions - Areas for further investigation in AI diagnostics
        ```
        """)

    if st.session_state.report_plan:
        st.subheader("Generated Report Plan")
        st.text_area("", value=st.session_state.report_plan, height=500, disabled=True)

        st.markdown("## Feedback")
        st.text_area("Suggest improvements (optional):", key="feedback_text", height=100)

        if st.button("Submit Feedback"):
            if not st.session_state.feedback_text.strip():
                st.warning("Feedback is empty, no changes will be made.")
            else: 
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
    st.markdown("You can parse from a plain report plan, paste structured JSON, or upload a JSON file.")
    st.markdown("""
    **What this step does:**
    - Extracts section titles from your report plan
    - Allows you to review and customize each section title
    - Supports both plain text and JSON-style structured plans
    """)
    
    if st.session_state.report_plan and st.button("Parse from Text-Based Report Plan"):
        try:
            section_titles = parse_report_plan_advanced(st.session_state.report_plan)
            st.session_state.parsed_sections = section_titles
            st.session_state.editable_sections = section_titles.copy()
            st.success(f"✅ Parsed {len(section_titles)} section(s) from text.")
        except Exception as e:
            st.error(f"Parsing failed: {e}")

    # This is the new part:
    load_structured_plan()

    # Editable titles
    if st.session_state.editable_sections:
        st.subheader("✏️ Editable Section Titles")
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

    st.markdown("""
    **What this step does:**
    - Searches the web for relevant information for each section
    - Gathers high-quality sources using different APIs (Tavily, Google, GitHub)
    - Processes and formats the search results for content generation
    """)

    search_option = st.selectbox("Select Search Engine", ["Tavily", "Google", "GitHub"])

    def skip_section(q):
        return q.strip().lower() in {"introduction", "conclusion", "summary", "closing remarks", "references"}

    async def run_search(search_api):
        search_results = []

        if search_api == "Google":
            from utils import google_search
            api_key = st.secrets["GOOGLE_SEARCH_API_KEY"]
            cse_id = st.secrets["GOOGLE_CSE_ID"]

            for q in st.session_state.section_queries:
                if skip_section(q):
                    search_results.append(f"(Search skipped for section: {q})")
                else:
                    result = google_search(q, api_key, cse_id)
                    formatted = deduplicate_and_format_sources(result, max_tokens_per_source=2000)
                    search_results.append(formatted)

        elif search_api == "GitHub":
            from utils import github_search
            tasks = []
            for q in st.session_state.section_queries:
                if skip_section(q):
                    search_results.append(f"(Search skipped for section: {q})")
                else:
                    tasks.append(github_search([q]))
            raw_results = await asyncio.gather(*tasks)
            formatted = [deduplicate_and_format_sources(r, max_tokens_per_source=2000) for r in raw_results]
            idx = 0
            for q in st.session_state.section_queries:
                if skip_section(q):
                    continue
                search_results.append(formatted[idx])
                idx += 1

        elif search_api == "Tavily":
            from utils import tavily_search_async
            tasks, indices = [], []
            for idx, q in enumerate(st.session_state.section_queries):
                if skip_section(q):
                    search_results.append(f"(Search skipped for section: {q})")
                else:
                    tasks.append(tavily_search_async([q]))
                    indices.append(idx)

            raw_results = await asyncio.gather(*tasks)
            formatted = [deduplicate_and_format_sources(r, max_tokens_per_source=2000) for r in raw_results]
            result_map = dict(zip(indices, formatted))
            for idx in range(len(st.session_state.section_queries)):
                if idx in result_map:
                    search_results.append(result_map[idx])

        return search_results

    if st.button(f"Run {search_option} Search"):
        st.session_state.search_results = asyncio.run(run_search(search_option))

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