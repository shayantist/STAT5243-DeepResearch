import streamlit as st
import asyncio
import os

tavily_api_key = st.secrets["TAVILY_API_KEY"]

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

#######
#######
# modified models
# https://ollama.com/search
#######
#######
ollama_model = ChatOllama(model="llama3:8b")

st.set_page_config(page_title="Deep Research with LangGraph", layout="wide")
st.title("Deep Research System")

# initialize session state variables
defaults = {
    "step": 1,
    "topic": "",
    "structure": "",
    "section_topic": "",
    "section_name": "",
    "report_plan": None,
    "section_queries": [],
    "search_results": "",
    "written_section": "",
    "written_sections": []  # <-- added to support multiple sections
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# sidebar navigation #to save info to look back
st.sidebar.title("Navigation")
if st.sidebar.button("Next Step"): # going to next step down the line
    st.session_state.step += 1
if st.sidebar.button("Reset"): ## reset every thing in the field
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 1
    st.rerun()

# choosing topic and structure
if st.session_state.step >= 1:
    st.header("Step 1: Define Topic and Structure")

    st.session_state.topic = st.text_input("Enter your research topic:", value=st.session_state.topic)
    st.session_state.structure = st.text_area("Specify the report structure:", height=200, value=st.session_state.structure)

    if st.button("Generate Report Plan"):
        planner_prompt = PromptTemplate.from_template(report_planner_instructions)
        chain = planner_prompt | ollama_model | StrOutputParser()
        with st.spinner("Generating plan..."):
            st.session_state.report_plan = chain.invoke({
                "topic": st.session_state.topic,
                "report_organization": st.session_state.structure,
                "context": "",
                "feedback": ""
            })
    if st.session_state.report_plan:
        st.subheader("Report Plan")
        st.code(st.session_state.report_plan)

# query generation
if st.session_state.step >= 2:
    st.header("Step 2: Generate Queries for a Section")

    st.session_state.section_topic = st.text_input("Choose a section topic:", value=st.session_state.section_topic)

    num_queries = st.slider("Number of queries", 1, 10, 3)
    if st.button("Generate Queries"):
        query_prompt = PromptTemplate.from_template(report_planner_query_writer_instructions)
        chain = query_prompt | ollama_model | StrOutputParser()
        with st.spinner("Generating queries..."):
            output = chain.invoke({
                "topic": st.session_state.topic,
                "report_organization": st.session_state.structure,
                "num_queries": num_queries
            })
            queries = [line.strip(" -•1234567890.").strip() for line in output.split("\n") if line.strip()]
            st.session_state.section_queries = queries
    if st.session_state.section_queries:
        st.subheader("Generated Queries")
        for q in st.session_state.section_queries:
            st.markdown(f"- {q}")

# Tavily Web Search
if st.session_state.step >= 3:
    st.header("Step 3: Perform Web Search")
    if st.button("Run Tavily Search"):
        async def search_and_show():
            # clear previous search results first
            st.session_state.search_results = ""

            results = await tavily_search_async(st.session_state.section_queries)
            formatted = deduplicate_and_format_sources(results, max_tokens_per_source=2000)
            st.session_state.search_results = formatted
        asyncio.run(search_and_show())
    if st.session_state.search_results:
        st.text_area("Search Results", st.session_state.search_results, height=500)

# write the section
if st.session_state.step >= 4:
    st.header("Step 4: Write the Report Section")

    st.session_state.section_name = st.text_input("Section Name (e.g. Introduction, Comparison):", value=st.session_state.section_name)

    if st.button("Generate Section"):
        # create a writer prompt
        writer_prompt = PromptTemplate.from_template(section_writer_instructions)
        fill_inputs = PromptTemplate.from_template(section_writer_inputs)

        filled = fill_inputs.format(
            topic=st.session_state.topic,
            section_name=st.session_state.section_name,
            section_topic=st.session_state.section_topic,
            section_content="",
            context=st.session_state.search_results
        )

        # inject a system message to tell model to reset
        system_message = (
            "You must start a brand-new generation. "
            "Forget all previous conversations. "
            "ONLY use the provided topic, section, and search results below."
        )

        chain = (
            PromptTemplate.from_template(f"System Instruction:\n{system_message}\n\n" + section_writer_instructions)
            | ollama_model
            | StrOutputParser()
        )

        # Debug Info
        # st.markdown("**DEBUG INFO**")
        # st.write("Current Topic:", st.session_state.topic)
        # st.write("Current Section Topic:", st.session_state.section_topic)
        # st.write("Current Section Name:", st.session_state.section_name)
        # st.write("Search Results Preview:", st.session_state.search_results[:500])

        with st.spinner("Writing section..."):
            section_text = chain.invoke({
                "topic": st.session_state.topic,
                "section_name": st.session_state.section_name,
                "section_topic": st.session_state.section_topic,
                "section_content": "",
                "context": st.session_state.search_results,
                "SECTION_WORD_LIMIT": 500
            })

            st.session_state.written_section = section_text
            st.session_state.written_sections.append(section_text)

    if st.session_state.written_sections:
        st.header("Generated Sections So Far")
        for idx, section in enumerate(st.session_state.written_sections, 1):
            st.subheader(f"Section {idx}")
            st.markdown(section)

# export report
if st.session_state.step >= 5:
    st.header("Step 5: Export Full Report")
    if st.session_state.written_sections:
        # add automatic numbering for sections
        numbered_sections = []
        for idx, section in enumerate(st.session_state.written_sections, 1):
            numbered_section = section.replace("## ", f"## {idx}. ", 1)
            numbered_sections.append(numbered_section)

        final_md = f"# {st.session_state.topic}\n\n" + "\n\n".join(numbered_sections)
        st.download_button("📄 Download Markdown", final_md, file_name="deep_research.md")
