import streamlit as st
from parse_resume import convert_to_documents
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import tempfile

# === SIDEBAR: API KEY & MODEL ===
with st.sidebar:
    groq_api_key = st.text_input("GROQ API Key", type="password", help="Get it from [console.groq.com](https://console.groq.com)")
    groq_model = st.text_input("GROQ Model", value="llama-3.1-8b-instant")
    start_click = st.button("Start")

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

# === HANDLE START BUTTON CLICK ===
if start_click:
    st.session_state.start_clicked = True

if not groq_api_key:
    st.warning("Please enter your GROQ API key in the sidebar to continue.")
    st.stop()

# Initialize LLM
llm = ChatGroq(model=groq_model, api_key=groq_api_key)

# === SETUP ===
UPLOADS_DIR = "Uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

st.title("Recruitment AI")
st.markdown("Upload a **job description** and **candidate resume** to get instant AI-powered analysis.")

# === UPLOAD: JOB DESCRIPTION ===
st.subheader("1. Upload Job Description")
uploaded_jd = st.file_uploader(
    "Choose JD (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    key="jd_uploader"
)

if uploaded_jd and (uploaded_jd.name != st.session_state.jd_filename):
    with st.spinner("Reading job description..."):
        path = os.path.join(UPLOADS_DIR, uploaded_jd.name)
        with open(path, "wb") as f:
            f.write(uploaded_jd.getvalue())
        docs = convert_to_documents(path)
        st.session_state.jd_text = "\n".join([d.page_content for d in docs])
        st.session_state.jd_filename = uploaded_jd.name
        st.session_state.analysis_done = False
        st.success(f"✅ JD loaded: `{uploaded_jd.name}`")

# === UPLOAD: RESUME ===
st.subheader("2. Upload Candidate Resume")
uploaded_resume = st.file_uploader(
    "Choose Resume (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    key="resume_uploader"
)

if uploaded_resume and (uploaded_resume.name != st.session_state.resume_filename) and st.session_state.jd_text:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_resume.name)[1]) as tmp:
        tmp.write(uploaded_resume.getvalue())
        tmp_path = tmp.name

    with st.spinner("Parsing resume..."):
        try:
            docs = convert_to_documents(tmp_path)
            st.session_state.resume_text = "\n".join([d.page_content for d in docs])
            st.session_state.resume_filename = uploaded_resume.name
            st.session_state.analysis_done = False
            st.success(f"✅ Resume loaded: `{uploaded_resume.name}`")
        finally:
            os.unlink(tmp_path)

# === START ANALYSIS ONCE BOTH FILES ARE READY AND BUTTON CLICKED ===
if st.session_state.start_clicked and st.session_state.jd_text and st.session_state.resume_text and not st.session_state.analysis_done:
    # Perform analysis first (no empty AI symbol)
    with st.spinner("Analyzing resume against job description..."):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert HR assistant. Be concise, professional, and insightful."),
            ("human", """
            Analyze this resume against the job description.
            Focus on: experience match, skills alignment, technical fit.
            End with a **Fit Score (0–10)** and one-sentence verdict.
            Keep analysis very short , concise and easy to interpret all under 100 to 200 words.
            After the analysis is done further provide as single follow up question that user can ask you to assit on based on the data the you have
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

    # Add AI analysis message
    st.session_state.messages.append({"role": "assistant", "content": analysis})
    st.session_state.analysis_done = True

# === DISPLAY FULL CHAT HISTORY ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === CHAT INPUT (Only after analysis) ===
if st.session_state.analysis_done:
    user_prompt = st.chat_input("Ask follow-up questions (e.g., 'Why the low score?', 'Suggest improvements')")
    if user_prompt:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context_msg = {
                    "role": "user",
                    "content": f"Context (do not repeat):\nJob Description:\n{st.session_state.jd_text}\n\nResume:\n{st.session_state.resume_text}"
                }
                full_history = [
                    {"role": "system", "content": "You are a helpful HR assistant. Use the provided context. Keep answers concise and actionable."},
                    context_msg
                ] + st.session_state.messages

                response = llm.invoke(full_history)
                reply = response.content
            st.markdown(reply)

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": reply})

else:
    if st.session_state.jd_text and st.session_state.resume_text:
        st.info("💡 Click **Start** in the sidebar to begin analysis.")
    elif st.session_state.jd_text:
        st.info("⏳ Please upload a resume to start analysis.")
    else:
        st.info("⬆️ Start by uploading a job description.")
