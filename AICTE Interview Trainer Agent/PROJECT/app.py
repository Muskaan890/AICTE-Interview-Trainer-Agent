"""
app.py – Streamlit Web UI for the AICTE Interview Trainer Agent
Powered by IBM Granite (watsonx.ai) + RAG
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from pathlib import Path
import tempfile

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Interview Trainer Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Lazy imports (avoid blocking on startup) ─────────────────────────────────
@st.cache_resource(show_spinner="Initialising IBM Granite + RAG pipeline…")
def get_agent():
    from src.agent import InterviewTrainerAgent
    agent = InterviewTrainerAgent()
    agent.setup(ingest_kb=True)
    return agent


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin-top: 0.4rem; }

    /* IBM badge */
    .ibm-badge {
        display: inline-block;
        background: #0062ff;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* Section cards */
    .section-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.6rem 0;
    }

    /* Output box */
    .output-box {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-left: 4px solid #0062ff;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 0.8rem;
        white-space: pre-wrap;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* Metric card */
    .metric-card {
        background: #e7f3ff;
        border: 1px solid #b3d7ff;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }

    /* Status pill */
    .status-live   { color: #28a745; font-weight: 700; }
    .status-mock   { color: #fd7e14; font-weight: 700; }

    /* Tab content spacing */
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎯 Interview Trainer Agent</h1>
    <p>Your AI-powered interview preparation coach — personalised questions, model answers & strategies</p>
    <span class="ibm-badge">⚡ Powered by IBM Granite · watsonx.ai · RAG</span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar — Agent Status & Profile Input ───────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=80)
    st.title("Candidate Profile")
    st.divider()

    with st.form("profile_form"):
        st.subheader("👤 Basic Info")
        name = st.text_input("Full Name", value="Alex Johnson", placeholder="Your full name")
        job_role = st.text_input("Target Job Role", value="Software Engineer", placeholder="e.g. Data Scientist")
        experience_level = st.selectbox(
            "Experience Level",
            ["Fresher", "Junior", "Mid-Level", "Senior", "Lead / Principal"],
            index=2,
        )
        years_exp = st.slider("Years of Experience", 0, 30, 3)
        target_company = st.text_input("Target Company (optional)", placeholder="e.g. IBM, Google")
        education = st.text_input("Education", placeholder="e.g. B.Tech Computer Science")

        st.subheader("🎯 Interview Focus")
        interview_type = st.selectbox(
            "Interview Type",
            ["Full Stack", "Technical Only", "Behavioral Only", "HR Only"],
        )
        num_questions = st.slider("Number of Questions", 5, 20, 10)

        st.subheader("📄 Resume (optional)")
        resume_text = st.text_area(
            "Paste Resume Text",
            placeholder="Paste your resume content here for personalized questions…",
            height=120,
        )
        resume_file = st.file_uploader("Or Upload Resume (PDF/DOCX)", type=["pdf", "docx", "txt"])

        st.subheader("🛠️ Skills")
        skills_input = st.text_input(
            "Key Skills (comma-separated)",
            placeholder="Python, Machine Learning, AWS, SQL",
        )

        submitted = st.form_submit_button("🚀 Start Interview Prep", use_container_width=True)

    st.divider()

    # Agent status
    try:
        agent = get_agent()
        granite_status = "🟢 LIVE" if agent.llm.is_live else "🟡 DEMO MODE"
        st.markdown(f"**IBM Granite:** {granite_status}")
        st.markdown(f"**Model:** `{agent.status['model_id'].split('/')[-1]}`")
        vs = agent.status["vectorstore_stats"]
        st.markdown(f"**KB Chunks:** {vs.get('collection_count', vs.get('in_memory_docs', 0))}")
    except Exception as e:
        st.error(f"Agent init error: {e}")


# ── Session State Defaults ────────────────────────────────────────────────────
for key in ["profile", "results", "eval_result", "tips", "behavioral_result"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── Main Content — Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Question Set",
    "✅ Answer Evaluator",
    "📅 Prep Strategy",
    "🎭 Behavioral Coach",
    "💡 Quick Tips",
])


# ── Trigger on Form Submit ────────────────────────────────────────────────────
if submitted:
    with st.spinner("Building your personalised interview prep package…"):
        try:
            agent = get_agent()
            skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]

            # Handle uploaded resume
            temp_resume_path = None
            if resume_file is not None:
                suffix = Path(resume_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(resume_file.read())
                    temp_resume_path = tmp.name

            profile = agent.create_profile(
                name=name,
                job_role=job_role,
                experience_level=experience_level,
                years_of_experience=years_exp,
                skills=skills_list,
                resume_text=resume_text,
                resume_file=temp_resume_path,
                target_company=target_company,
                interview_type=interview_type,
                education=education,
            )
            st.session_state.profile = profile

            results = agent.run_full_preparation(profile, num_questions=num_questions)
            st.session_state.results = results

            # Clean up temp file
            if temp_resume_path:
                try:
                    os.unlink(temp_resume_path)
                except Exception:
                    pass

            st.success(f"✅ Interview prep ready for **{name}** applying as **{job_role}**!")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)


# ── Tab 1: Question Set ───────────────────────────────────────────────────────
with tab1:
    st.header("📋 Tailored Interview Questions")

    if st.session_state.results:
        results = st.session_state.results
        profile = st.session_state.profile

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Candidate", profile.name)
        with col2:
            st.metric("Target Role", profile.job_role)
        with col3:
            st.metric("Experience", profile.experience_level)
        with col4:
            mode = "🟢 IBM Granite (Live)" if results.get("ibm_granite_live") else "🟡 Demo Mode"
            st.metric("AI Engine", mode)

        st.divider()

        st.markdown("### 🔍 Generated Questions")
        st.markdown(
            f'<div class="output-box">{results["question_set"]}</div>',
            unsafe_allow_html=True,
        )

        # Download button
        st.download_button(
            label="⬇️ Download Questions (TXT)",
            data=results["question_set"],
            file_name=f"interview_questions_{profile.name.replace(' ', '_')}.txt",
            mime="text/plain",
        )

    else:
        st.info("👈 Fill in your profile in the sidebar and click **Start Interview Prep** to generate questions.")
        st.markdown("""
        ### 📌 What you'll get:
        - **Technical questions** tailored to your specific role and skill stack
        - **Behavioral questions** using the STAR method framework
        - **Situational questions** testing decision-making
        - **HR questions** for culture-fit and motivation assessment
        - Each question includes: *difficulty level*, *assessment criteria*, and a *model answer hint*
        """)


# ── Tab 2: Answer Evaluator ───────────────────────────────────────────────────
with tab2:
    st.header("✅ Answer Evaluator")
    st.markdown("Practice answering interview questions and get AI-powered feedback from IBM Granite.")

    if st.session_state.profile is None:
        st.info("👈 Please generate your interview prep first using the sidebar.")
    else:
        profile = st.session_state.profile

        with st.form("eval_form"):
            eval_question = st.text_area(
                "Interview Question",
                placeholder="Paste or type the interview question here…",
                height=80,
            )
            eval_answer = st.text_area(
                "Your Answer",
                placeholder="Write your answer here. Try to use the STAR format for behavioral questions.",
                height=180,
            )
            eval_submitted = st.form_submit_button("📊 Evaluate My Answer", use_container_width=True)

        if eval_submitted and eval_question and eval_answer:
            with st.spinner("IBM Granite is evaluating your answer…"):
                try:
                    agent = get_agent()
                    evaluation = agent.evaluate_answer(profile, eval_question, eval_answer)
                    st.session_state.eval_result = evaluation
                except Exception as e:
                    st.error(f"Evaluation error: {e}")

        if st.session_state.eval_result:
            ev = st.session_state.eval_result
            st.divider()
            st.subheader("📊 Evaluation Results")

            if ev.score > 0:
                score_color = "green" if ev.score >= 7 else ("orange" if ev.score >= 5 else "red")
                st.markdown(
                    f"### Score: <span style='color:{score_color};font-size:1.5rem'>"
                    f"**{ev.score}/10**</span>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="output-box">{ev.raw_feedback}</div>',
                unsafe_allow_html=True,
            )

            st.download_button(
                label="⬇️ Download Feedback",
                data=ev.raw_feedback,
                file_name="answer_evaluation.txt",
                mime="text/plain",
            )


# ── Tab 3: Preparation Strategy ───────────────────────────────────────────────
with tab3:
    st.header("📅 7-Day Preparation Strategy")

    if st.session_state.results:
        strategy = st.session_state.results.get("preparation_strategy", "")
        st.markdown(
            f'<div class="output-box">{strategy}</div>',
            unsafe_allow_html=True,
        )
        st.download_button(
            label="⬇️ Download Strategy (TXT)",
            data=strategy,
            file_name="prep_strategy.txt",
            mime="text/plain",
        )
    else:
        st.info("👈 Generate your interview prep package first.")
        st.markdown("""
        ### 📌 Your strategy will include:
        - Day-by-day preparation timeline
        - Topic-specific study guides
        - Mock interview scheduling
        - Confidence-building exercises
        - Common pitfalls to avoid
        - Curated resource recommendations
        """)


# ── Tab 4: Behavioral Coach ───────────────────────────────────────────────────
with tab4:
    st.header("🎭 Behavioral Interview Coach")
    st.markdown("Generate STAR-based behavioral questions for any competency scenario.")

    if st.session_state.profile is None:
        st.info("👈 Please generate your interview prep first using the sidebar.")
    else:
        profile = st.session_state.profile

        BEHAVIORAL_SCENARIOS = [
            "Leadership and conflict resolution",
            "Teamwork and collaboration",
            "Handling failure and learning",
            "Working under pressure / tight deadlines",
            "Innovation and creative problem solving",
            "Customer / stakeholder management",
            "Mentoring and knowledge sharing",
            "Adapting to change",
            "Ethical dilemma",
            "Custom scenario…",
        ]

        scenario_choice = st.selectbox("Select a Behavioral Scenario", BEHAVIORAL_SCENARIOS)

        if scenario_choice == "Custom scenario…":
            scenario = st.text_input("Describe your custom scenario")
        else:
            scenario = scenario_choice

        if st.button("🎭 Generate Behavioral Questions", use_container_width=True):
            if scenario:
                with st.spinner("Generating behavioral assessment…"):
                    try:
                        agent = get_agent()
                        result = agent.get_behavioral_assessment(profile, scenario)
                        st.session_state.behavioral_result = result
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.behavioral_result:
            st.markdown(
                f'<div class="output-box">{st.session_state.behavioral_result}</div>',
                unsafe_allow_html=True,
            )

        # Always show the generated behavioral from full prep too
        elif st.session_state.results:
            behavioral = st.session_state.results.get("behavioral_assessment", "")
            if behavioral:
                st.subheader("📌 Behavioral Assessment from Your Prep Package")
                st.markdown(
                    f'<div class="output-box">{behavioral}</div>',
                    unsafe_allow_html=True,
                )


# ── Tab 5: Quick Tips ─────────────────────────────────────────────────────────
with tab5:
    st.header("💡 Quick Improvement Tips")

    if st.session_state.profile is None:
        st.info("👈 Please generate your interview prep first using the sidebar.")
    else:
        profile = st.session_state.profile

        if st.button("💡 Get Personalized Improvement Tips", use_container_width=True):
            with st.spinner("IBM Granite is generating tips for you…"):
                try:
                    agent = get_agent()
                    tips = agent.get_improvement_tips(profile)
                    st.session_state.tips = tips
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.tips:
            st.markdown(
                f'<div class="output-box">{st.session_state.tips}</div>',
                unsafe_allow_html=True,
            )

    # Static tips always visible
    st.divider()
    st.markdown("### 📌 Universal Interview Success Tips")
    tips_cols = st.columns(2)
    universal_tips = [
        ("🎯 Research Deeply", "Know the company's products, culture, recent news, and the team you're joining."),
        ("⭐ Use STAR Method", "Structure behavioral answers: Situation → Task → Action → Result with metrics."),
        ("📊 Quantify Impact", "Replace 'improved performance' with 'improved latency by 40% serving 1M users'."),
        ("❓ Ask Questions", "Prepare 3–5 thoughtful questions for the interviewer. Silence signals disinterest."),
        ("🧘 Manage Nerves", "Pause before answering. Interviewers respect thoughtful responses over rushed ones."),
        ("💼 Tailor Each Application", "Customize examples for each company — generic answers rarely impress."),
    ]
    for i, (title, desc) in enumerate(universal_tips):
        with tips_cols[i % 2]:
            st.markdown(f"**{title}**\n\n{desc}")
            st.markdown("")
