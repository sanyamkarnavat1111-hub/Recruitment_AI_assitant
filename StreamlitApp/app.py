import streamlit as st
from utils import convert_to_documents
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import tempfile
from dotenv import load_dotenv

# === LOAD ENV VARIABLES ===
load_dotenv()

GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.1-8b-instant"

# === VALIDATE ===
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in `.env` file. Please add it and restart.")
    st.stop()

# === INITIALIZE LLM ===
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

# === SESSION STATE ===
if 'start_clicked' not in st.session_state:
    st.session_state.start_clicked = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = ""
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'jd_filename' not in st.session_state:
    st.session_state.jd_filename = ""
if 'resume_filename' not in st.session_state:
    st.session_state.resume_filename = ""

# === SETUP ===
UPLOADS_DIR = "Uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# === BEFORE START: SHOW UPLOADS + START BUTTON ===
if not st.session_state.start_clicked:
    st.title("Recruitment AI")
    st.markdown("Upload a **job description** and **candidate resume** to get instant AI-powered analysis.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Job Description")
        uploaded_jd = st.file_uploader(
            "Choose JD (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="jd_uploader",
            label_visibility="collapsed"
        )

        if uploaded_jd and uploaded_jd.name != st.session_state.jd_filename:
            with st.spinner("Reading job description..."):
                path = os.path.join(UPLOADS_DIR, uploaded_jd.name)
                with open(path, "wb") as f:
                    f.write(uploaded_jd.getvalue())
                docs = convert_to_documents(path)
                st.session_state.jd_text = "\n".join([d.page_content for d in docs])
                st.session_state.jd_filename = uploaded_jd.name
                st.session_state.analysis_done = False
                st.success(f"JD loaded: `{uploaded_jd.name}`")

    with col2:
        st.subheader("2. Candidate Resume")
        uploaded_resume = st.file_uploader(
            "Choose Resume (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="resume_uploader",
            label_visibility="collapsed"
        )

        if uploaded_resume and uploaded_resume.name != st.session_state.resume_filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_resume.name)[1]) as tmp:
                tmp.write(uploaded_resume.getvalue())
                tmp_path = tmp.name

            with st.spinner("Parsing resume..."):
                try:
                    docs = convert_to_documents(tmp_path)
                    st.session_state.resume_text = "\n".join([d.page_content for d in docs])
                    st.session_state.resume_filename = uploaded_resume.name
                    st.session_state.analysis_done = False
                    st.success(f"Resume loaded: `{uploaded_resume.name}`")
                finally:
                    os.unlink(tmp_path)

    # === START BUTTON BELOW UPLOADS ===
    st.markdown("---")
    col_start1, col_start2, col_start3 = st.columns([1, 1, 1])
    with col_start2:
        start_clicked = st.button(
            "Start Analysis",
            type="primary",
            use_container_width=True,
            disabled=not (st.session_state.jd_text and st.session_state.resume_text)
        )

    if start_clicked:
        if not st.session_state.jd_text or not st.session_state.resume_text:
            st.error("Please upload both files before starting.")
        else:
            st.session_state.start_clicked = True
            st.rerun()

    # === INFO MESSAGES ===
    if st.session_state.jd_text and st.session_state.resume_text:
        st.info("Both files uploaded. Click **Start Analysis** to begin.")
    elif st.session_state.jd_text:
        st.info("Upload a resume to continue.")
    else:
        st.info("Start by uploading a job description.")

# === AFTER START: ONLY CHAT ===
else:
    st.title("AI Recruitment Assistant")

    # === INITIAL ANALYSIS (RUN ONCE) ===
    if st.session_state.jd_text and st.session_state.resume_text and not st.session_state.analysis_done:
        with st.spinner("Analyzing resume against job description..."):
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert HR assistant. Be concise, professional, and insightful."),
                ("human", """
                Analyze the following resume against the provided job description.
                Focus on: experience match, skills alignment, and technical fit.

                Provide a concise, easy-to-read analysis (100–200 words) that includes:
                - A brief summary of how well the candidate’s background aligns with the job.
                - Observations on strengths and possible gaps.
                - A Fit Score (0–10) based on overall suitability.
                - A one-sentence verdict summarizing the fit.

                Strict criteria:
                - Experience must meet or exceed JD requirements.
                - 70–80% key skill overlap.
                - Past roles largely relevant.

                Finally, suggest one follow-up question to deepen the analysis.

                Job Description:
                {job_description_text}

                Resume:
                {resume_text}
                """)
            ])
            chain = prompt | llm
            response = chain.invoke({
                "job_description_text": st.session_state.jd_text,
                "resume_text": st.session_state.resume_text
            })
            analysis = response.content

        st.session_state.messages.append({"role": "assistant", "content": analysis})
        st.session_state.analysis_done = True
        st.rerun()

    # === CHAT HISTORY ===
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # === CHAT INPUT ===
    if user_prompt := st.chat_input("Ask follow-up questions (e.g., 'Why the low score?', 'Suggest improvements')"):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context_msg = {
                    "role": "user",
                    "content": f"Context (do not repeat):\nJob Description:\n{st.session_state.jd_text}\n\nResume:\n{st.session_state.resume_text}"
                }
                full_history = [
                    {"role": "system", "content": "You are a helpful HR assistant. Use context. Keep answers concise and actionable."},
                    context_msg
                ] + st.session_state.messages

                response = llm.invoke(full_history)
                reply = response.content
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})