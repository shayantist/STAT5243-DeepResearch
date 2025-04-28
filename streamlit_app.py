import streamlit as st
import asyncio
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

st.set_page_config(page_title="Deep Research with LangGraph", layout="wide")

#######
#######
# modified models
# https://ollama.com/search
#######
#######


# Define available models
AVAILABLE_MODELS = {
    "llama3": "llama3:8b",
    "mistral": "mistral",
    "qwen2": "qwen2:12b",
    "gemma": "gemma"
}


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
    "written_sections": [],  # added to support multiple sections
    "feedback_text": "", # default blank feedback
    "feedback_rating": 3,
    "feedback_submitted": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# sidebar navigation #to save info to look back
# Sidebar model selection
st.sidebar.title("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose your LLM Model",
    options=["llama3:8b", "qwen2:7b", "nous-hermes2", "yi:6b"],
    index=0  # choose the first one as default（like llama3:8b）
)

# initialize model based on selection
ollama_model = ChatOllama(model=selected_model)


# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# set sidebar  
st.sidebar.markdown("---")

# sidebar navigation #to save info to look back
st.sidebar.title("Navigation")
if st.sidebar.button("Next Step"): # going to next step down the line
    st.session_state.step += 1
if st.sidebar.button("Reset All"): ## reset every thing in the field
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 1
    st.rerun()

# show existing section names
if st.session_state.written_sections:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Sections Created:")
    for idx, section in enumerate(st.session_state.written_sections, 1):
        st.sidebar.markdown(f"- Section {idx}")

    if st.sidebar.button("Clear Last Section"):
        if st.session_state.written_sections:
            st.session_state.written_sections.pop()

# title
st.title("Deep Research Report Generator")

# Define model explanations
MODEL_DESCRIPTIONS = {
    "llama3:8b": "🦙 Llama3-8B: Strong in logical writing and structured reasoning.",
    "qwen2:7b": "🧠 Qwen2-7B: Excellent at fluent report generation and analysis.",
    "nous-hermes2": "📝 Nous Hermes2: Optimized for clear summarization and coherent writing.",
    "yi:6b": "🚀 Yi-6B: Lightweight and fast, great for short and fluent reports."
}

# Get the description for the selected model
model_description = MODEL_DESCRIPTIONS.get(selected_model, "🚀 Using selected model for report generation.")

# After st.title(...) but before the main content, show this!
st.markdown(f"**Model Selected:** {model_description}")


st.markdown("""

Welcome! This app will guide you through **five easy steps** to create a full research report:

1. **Define Topic & Structure** – Set what your report is about and outline sections.
2. **Generate Queries** – Create search queries for a section.
3. **Web Search** – Fetch research material from real websites.
4. **Write Section** – AI writes a polished section based on your research.
5. **Export Report** – Download your completed markdown file!

**Tip**: Save one section at a time, then continue adding more sections before final export.

---
""")

# choosing topic and structure
if st.session_state.step >= 1:
    st.header("Step 1: Define Topic and Structure")

    st.text_input("Enter your research topic:", key="topic")
    st.text_area("Specify the report structure:", key="structure", height=200)

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
        # clear feedback space
        st.session_state.feedback_text = ""
        st.session_state.feedback_rating = 3
        st.session_state.feedback_submitted = False
        st.rerun()

if st.session_state.report_plan:
    st.subheader("Report Plan")
    st.code(st.session_state.report_plan)

# Step 1.5: Provide Feedback on Report Plan
# Check at the beginning of the page. If there are marks, clear feedback_text
if st.session_state.get("clear_feedback_flag"):
    st.session_state.feedback_text = ""
    st.session_state.feedback_rating = 3
    #st.session_state.feedback_submitted = False
    del st.session_state.clear_feedback_flag  # clear this flag

if st.session_state.get("proceed_anyway_clicked", False):
    st.success("A new Report Plan has been generated based on your feedback! Proceed to Step 2 when ready.")
    st.session_state.feedback_submitted = True
    del st.session_state["proceed_anyway_clicked"]
    st.stop()


if st.session_state.get("show_success"):
    st.success("A new Report Plan has been generated based on your feedback! Proceed to Step 2 when ready.")
    #st.stop()
    #st.session_state.feedback_submitted = True
    del st.session_state["show_success"]



if st.session_state.step >= 1 and st.session_state.report_plan:

    st.markdown("## Step 1.5: Provide Feedback on Report Plan")

    if st.session_state.get("show_success"):
        st.success("A new Report Plan has been generated based on your feedback! Proceed to Step 2 when ready.")
        #st.session_state.success_shown_once = True
        st.session_state.feedback_submitted = True 
        #del st.session_state["show_success"]
        #st.stop() 

    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False

    if not st.session_state.feedback_submitted:
        st.caption("""
        (Optional) If you are satisfied with the plan, you can skip feedback and proceed to Step 2. 
        If you think improvements are needed, please describe them below.
        A new report plan will be generated based on your feedback.
        """)

        st.text_area(
            "Your feedback (optional):",
            key="feedback_text",
            height=150
        )

        st.slider(
            "Rate the Report Plan (1 = Very Poor, 5 = Excellent):",
            1, 5,
            key="feedback_rating"
        )

        if st.button("Submit Feedback"):
            feedback_text = st.session_state.feedback_text
            feedback_rating = st.session_state.feedback_rating


            sentiment_score = analyzer.polarity_scores(feedback_text)
            compound_score = sentiment_score['compound']

            if feedback_text.strip() == "" and feedback_rating >= 3:
                # If the feedback is empty but the score is good, it will be directly judged as successful
                st.session_state.feedback_submitted = True
                st.session_state.show_success = True
                st.rerun()

            # First, check for conflicts where the text is positive but given a low score
            elif compound_score >= 0.2 and feedback_rating <= 2:
                with st.expander("Your feedback seems positive but the rating is low. Please confirm:"):
                    st.warning("""
                    It looks like you wrote positive feedback, but gave a low rating.

                    - If you actually liked the plan, please consider adjusting your rating to 4 or 5.
                    - If you are dissatisfied, please provide more specific feedback.
                    - You can also proceed with the current rating if you prefer.
                    """)
                    revise_feedback = st.button("Revise My Feedback")
                    proceed_anyway = st.button("Proceed Anyway")
                    #proceed_anyway = st.button("Proceed Anyway", key="proceed_anyway_in_expander")


                    if revise_feedback:
                        st.session_state.feedback_text = ""
                        st.session_state.feedback_rating = 3
                        st.session_state.feedback_submitted = False  
                        st.rerun() 
                        st.stop()
                
                    if proceed_anyway:
                        st.session_state.feedback_submitted = True
                        st.session_state.show_success = True
                        st.rerun()

                    if not proceed_anyway:
                        st.stop()
           
            elif compound_score >= 0.2 and feedback_rating >= 3:
                # Here is Situation 2: Good feedback + good score
                st.session_state.feedback_submitted = True
                st.session_state.show_success = True  
                st.rerun()
            
            else:
                # If the text is negative or has a low rating, it will all be regenerated
                if compound_score <= -0.4 or feedback_rating <= 2:
                    planner_prompt = PromptTemplate.from_template(report_planner_instructions)

                    chain = planner_prompt | ollama_model | StrOutputParser()

                    with st.spinner("Regenerating Report Plan based on your feedback..."):
                        st.session_state.report_plan = chain.invoke({
                            "topic": st.session_state.topic,
                            "report_organization": st.session_state.structure,
                            "context": "",
                            "feedback": feedback_text
                        })
                    # after regeneration，clear feedback
                    st.session_state.clear_feedback_flag = True  
                    st.success("A new Report Plan has been generated based on your feedback! Proceed to Step 2 when ready.")
                    st.rerun()
    else:
        # The feedback has been submitted and a success prompt is displayed
        st.success("✅ Feedback submitted! You can now click 'Next Step' on the sidebar ➡️")
            



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

        st.subheader("Full Report Preview")
        st.text_area("Full Report Markdown", final_md, height=600)

        st.download_button("Download Markdown", final_md, file_name="deep_research.md")
