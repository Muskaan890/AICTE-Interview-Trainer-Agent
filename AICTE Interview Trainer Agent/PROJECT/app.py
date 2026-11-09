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
import json
from datetime import datetime

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AICTE Interview Trainer Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://cloud.ibm.com/docs/watsonx",
        "About": "AICTE Interview Trainer Agent — Powered by IBM Granite + RAG",
    },
)


# ── Lazy imports ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Initialising IBM Granite + RAG pipeline…")
def get_agent():
    from src.agent import InterviewTrainerAgent
    agent = InterviewTrainerAgent()
    agent.setup(ingest_kb=True)
    return agent


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* --- Reset & Base --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* --- Main header --- */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 { font-size: 2.4rem; margin: 0; font-weight: 700; letter-spacing: -0.5px; }
    .main-header p  { font-size: 1rem; opacity: 0.82; margin-top: 0.5rem; font-weight: 300; }

    /* --- IBM badge --- */
    .ibm-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #0062ff;
        color: white;
        padding: 0.28rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.6rem;
        letter-spacing: 0.3px;
    }
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #198754;
        color: white;
        padding: 0.28rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.6rem;
        margin-left: 6px;
        letter-spacing: 0.3px;
    }
    .demo-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #fd7e14;
        color: white;
        padding: 0.28rem 0.9rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.6rem;
        margin-left: 6px;
        letter-spacing: 0.3px;
    }

    /* --- Output box --- */
    .output-box {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-left: 4px solid #0062ff;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 0.8rem;
        white-space: pre-wrap;
        font-size: 0.875rem;
        line-height: 1.75;
        color: #1f2328;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    /* --- Score bar --- */
    .score-container {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    .score-bar-track {
        background: #e9ecef;
        border-radius: 50px;
        height: 12px;
        margin-top: 6px;
        overflow: hidden;
    }

    /* --- Profile card --- */
    .profile-card {
        background: linear-gradient(135deg, #e7f3ff 0%, #f0f4ff 100%);
        border: 1px solid #b3d7ff;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }

    /* --- Feature cards --- */
    .feature-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: #0062ff; }
    .feature-icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
    .feature-title { font-weight: 600; font-size: 0.9rem; color: #1f2328; }
    .feature-desc  { font-size: 0.78rem; color: #57606a; margin-top: 0.2rem; }

    /* --- Status indicators --- */
    .status-live  { color: #198754; font-weight: 700; }
    .status-demo  { color: #fd7e14; font-weight: 700; }
    .status-error { color: #dc3545; font-weight: 700; }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.875rem;
        font-weight: 500;
    }

    /* --- Divider --- */
    hr { border: none; border-top: 1px solid #e9ecef; margin: 1.2rem 0; }

    /* --- Hide Streamlit branding --- */
    #MainMenu  { visibility: hidden; }
    footer     { visibility: hidden; }

    /* --- Sidebar styling --- */
    .sidebar-section {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
try:
    _agent_check = get_agent()
    _is_live = _agent_check.llm.is_live
    _model_name = _agent_check.llm.model_name
    _mode_badge = (
        '<span class="live-badge">🟢 IBM Granite LIVE</span>'
        if _is_live
        else '<span class="demo-badge">🟡 Demo Mode</span>'
    )
except Exception:
    _mode_badge = '<span class="demo-badge">🟡 Initializing…</span>'
    _is_live = False
    _model_name = "granite-3-8b-instruct"

st.markdown(f"""
<div class="main-header">
    <h1>🎯 AICTE Interview Trainer Agent</h1>
    <p>Your AI-powered interview coach — personalised questions, model answers, prep strategies & behavioral coaching</p>
    <span class="ibm-badge">⚡ IBM Granite · watsonx.ai · RAG</span>
    {_mode_badge}
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=72)
    except Exception:
        st.markdown("**IBM** watsonx.ai")

    st.title("🎯 Candidate Profile")
    st.caption("Fill in your details and click Start to generate your personalised interview prep package.")
    st.divider()

    with st.form("profile_form"):
        st.subheader("👤 Basic Information")
        name = st.text_input("Full Name", value="Alex Johnson", placeholder="Your full name")
        job_role = st.text_input(
            "Target Job Role",
            value="Software Engineer",
            placeholder="e.g. Data Scientist, DevOps Engineer",
        )
        experience_level = st.selectbox(
            "Experience Level",
            ["Fresher", "Junior", "Mid-Level", "Senior", "Lead / Principal"],
            index=2,
        )
        years_exp = st.slider("Years of Experience", 0, 30, 3)
        target_company = st.text_input(
            "Target Company (optional)",
            placeholder="e.g. IBM, Google, TCS",
        )
        education = st.text_input(
            "Education",
            placeholder="e.g. B.Tech Computer Science, MBA",
        )

        st.subheader("🎯 Interview Focus")
        interview_type = st.selectbox(
            "Interview Type",
            ["Full Stack", "Technical Only", "Behavioral Only", "HR Only"],
        )
        num_questions = st.slider("Number of Questions to Generate", 5, 20, 10)

        st.subheader("🛠️ Skills")
        skills_input = st.text_input(
            "Key Skills (comma-separated)",
            placeholder="Python, Machine Learning, AWS, SQL, Docker",
        )

        st.subheader("📄 Resume Input")
        resume_text = st.text_area(
            "Paste Resume Text",
            placeholder="Paste your resume content here for hyper-personalized questions…",
            height=100,
        )
        resume_file = st.file_uploader(
            "Or Upload Resume (PDF / DOCX / TXT)",
            type=["pdf", "docx", "txt"],
        )

        submitted = st.form_submit_button(
            "🚀 Generate Interview Prep Package",
            use_container_width=True,
            type="primary",
        )

    st.divider()

    # ── Agent Status Panel ─────────────────────────────────────────────
    st.subheader("⚙️ Agent Status")
    try:
        agent_s = get_agent()
        vs = agent_s.status["vectorstore_stats"]
        kb_count = vs.get("collection_count") or vs.get("in_memory_docs", 0)

        if agent_s.llm.is_live:
            st.markdown("**IBM Granite:** 🟢 LIVE")
        else:
            st.markdown("**IBM Granite:** 🟡 Demo Mode")

        st.markdown(f"**Model:** `{agent_s.llm.model_name}`")
        st.markdown(f"**KB Chunks:** {kb_count}")
        st.markdown(f"**ChromaDB:** {'✅' if vs.get('chroma_available') else '⚠️ In-Memory'}")
        st.markdown(f"**Embeddings:** {'✅' if vs.get('embedder_available') else '⚠️ Keyword Search'}")

        if not agent_s.llm.is_live:
            st.info(
                "Set `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` in your `.env` file "
                "to enable live IBM Granite AI responses.\n\n"
                "[Get free IBM Cloud account →](https://cloud.ibm.com/registration)"
            )
    except Exception as e:
        st.error(f"Agent init error: {e}")


# ── Session State Defaults ─────────────────────────────────────────────────
for key in ["profile", "results", "eval_result", "tips", "behavioral_result", "prep_started"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "prep_started" not in st.session_state:
    st.session_state.prep_started = False


# ── Trigger on Form Submit ─────────────────────────────────────────────────
if submitted:
    with st.spinner("⚡ IBM Granite is building your personalised interview prep package…"):
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
            st.session_state.eval_result = None
            st.session_state.tips = None
            st.session_state.behavioral_result = None

            results = agent.run_full_preparation(profile, num_questions=num_questions)
            st.session_state.results = results
            st.session_state.prep_started = True

            if temp_resume_path:
                try:
                    os.unlink(temp_resume_path)
                except Exception:
                    pass

            mode_label = "IBM Granite (Live)" if results.get("ibm_granite_live") else "Demo Mode"
            st.success(
                f"✅ Interview prep ready for **{name}** applying as **{job_role}** "
                f"· AI Engine: **{mode_label}**"
            )

        except Exception as e:
            st.error(f"❌ Error generating prep: {e}")
            st.exception(e)


# ── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Question Set",
    "✅ Answer Evaluator",
    "📅 Prep Strategy",
    "🎭 Behavioral Coach",
    "💡 Improvement Tips",
    "📊 Session Summary",
])


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Question Set
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("📋 Tailored Interview Question Set")

    if st.session_state.results and st.session_state.profile:
        results = st.session_state.results
        profile = st.session_state.profile

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👤 Candidate", profile.name)
        col2.metric("🎯 Target Role", profile.job_role)
        col3.metric("📊 Level", profile.experience_level)
        col4.metric(
            "🤖 AI Engine",
            "IBM Granite (Live)" if results.get("ibm_granite_live") else "Demo Mode",
        )

        if profile.target_company:
            st.info(f"🏢 Questions tailored for target company: **{profile.target_company}**")

        st.divider()
        st.markdown("### 🔍 Generated Questions")

        st.markdown(
            f'<div class="output-box">{results["question_set"]}</div>',
            unsafe_allow_html=True,
        )

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="⬇️ Download Questions (TXT)",
                data=results["question_set"],
                file_name=f"questions_{profile.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_dl2:
            q_json = json.dumps({
                "candidate": profile.name,
                "role": profile.job_role,
                "level": profile.experience_level,
                "generated_at": datetime.now().isoformat(),
                "questions": results["question_set"],
            }, indent=2)
            st.download_button(
                label="⬇️ Download as JSON",
                data=q_json,
                file_name=f"questions_{profile.name.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True,
            )

    else:
        # Feature showcase when no prep yet
        st.info("👈 Fill in your profile in the sidebar and click **Generate Interview Prep Package** to start.")
        st.markdown("### 📌 Your Personalised Question Set Will Include:")

        feat_cols = st.columns(4)
        features = [
            ("🔧", "Technical Questions", "Role-specific, skill-matched to your stack and level"),
            ("🤝", "Behavioral (STAR)", "Scenario-based questions with STAR guidance"),
            ("🎯", "Situational", "Decision-making and crisis response scenarios"),
            ("💼", "HR & Culture-Fit", "Motivation, values, and culture alignment"),
        ]
        for col, (icon, title, desc) in zip(feat_cols, features):
            with col:
                st.markdown(
                    f'<div class="feature-card">'
                    f'<div class="feature-icon">{icon}</div>'
                    f'<div class="feature-title">{title}</div>'
                    f'<div class="feature-desc">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("""
        #### Each question includes:
        - ✅ **Difficulty level** (Easy / Medium / Hard)
        - 🔍 **What the interviewer is assessing**
        - 💡 **Model answer framework** to guide your response
        - 📊 **Category tag** for organized practice
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Answer Evaluator
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("✅ AI Answer Evaluator")
    st.markdown("Practice answering any interview question and get instant, scored feedback from IBM Granite.")

    if st.session_state.profile is None:
        st.info("👈 Please generate your interview prep package first using the sidebar.")
        st.markdown("""
        The Answer Evaluator will:
        - **Score your answer 1–10** with detailed justification
        - Identify **specific strengths** in your response
        - Point out **exact areas to improve** with actionable guidance
        - Provide a **model answer** that demonstrates ideal structure
        - Generate **likely follow-up questions** to anticipate
        """)
    else:
        profile = st.session_state.profile

        with st.form("eval_form"):
            eval_question = st.text_area(
                "🔍 Interview Question",
                placeholder="Paste or type the interview question you want to practice…",
                height=80,
            )
            eval_answer = st.text_area(
                "💬 Your Answer",
                placeholder=(
                    "Write your answer here.\n\n"
                    "💡 Tip: For behavioral questions, use STAR format:\n"
                    "Situation → Task → Action → Result (with metrics)"
                ),
                height=200,
            )
            eval_btn = st.form_submit_button(
                "📊 Evaluate My Answer with IBM Granite",
                use_container_width=True,
                type="primary",
            )

        if eval_btn:
            if not eval_question or not eval_answer:
                st.warning("⚠️ Please enter both a question and your answer to evaluate.")
            else:
                with st.spinner("⚡ IBM Granite is evaluating your answer…"):
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
                score_pct = int(ev.score * 10)
                if ev.score >= 8:
                    bar_color, label, emoji = "#198754", "Excellent", "🌟"
                elif ev.score >= 6:
                    bar_color, label, emoji = "#0062ff", "Good", "👍"
                elif ev.score >= 4:
                    bar_color, label, emoji = "#fd7e14", "Needs Work", "📝"
                else:
                    bar_color, label, emoji = "#dc3545", "Significant Improvement Needed", "🔧"

                st.markdown(
                    f"""
                    <div class="score-container">
                        <div style="display:flex;align-items:center;justify-content:space-between;">
                            <div>
                                <span style="font-size:2rem;font-weight:700;color:{bar_color}">{ev.score}/10</span>
                                <span style="font-size:0.9rem;color:#57606a;margin-left:10px">{emoji} {label}</span>
                            </div>
                            <span style="font-size:1.8rem">{emoji}</span>
                        </div>
                        <div class="score-bar-track">
                            <div style="background:{bar_color};height:100%;width:{score_pct}%;border-radius:50px;transition:width 0.5s;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="output-box">{ev.raw_feedback}</div>',
                unsafe_allow_html=True,
            )

            st.download_button(
                label="⬇️ Download Evaluation Report",
                data=f"Question:\n{ev.question}\n\nYour Answer:\n{ev.candidate_answer}\n\nFeedback:\n{ev.raw_feedback}",
                file_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Preparation Strategy
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📅 7-Day Preparation Strategy")
    st.markdown("Your personalized, day-by-day interview preparation roadmap generated by IBM Granite.")

    if st.session_state.results:
        strategy = st.session_state.results.get("preparation_strategy", "")
        if strategy:
            st.markdown(
                f'<div class="output-box">{strategy}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                label="⬇️ Download 7-Day Strategy (TXT)",
                data=strategy,
                file_name=f"prep_strategy_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
            )
        else:
            st.warning("Preparation strategy not available. Please regenerate your prep package.")
    else:
        st.info("👈 Generate your interview prep package first using the sidebar.")
        st.markdown("""
        ### 📌 Your 7-Day Strategy Will Include:
        - 📅 **Day-by-day tasks** with morning, afternoon, and evening breakdowns
        - 📚 **Role-specific study topics** with priority levels
        - 🎭 **Mock interview scheduling** guidance
        - 🛠️ **Curated resources** for your exact role and level
        - ⚠️ **Common pitfalls** specific to your target role
        - 💪 **Confidence-building exercises** to reduce interview anxiety
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Behavioral Coach
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🎭 Behavioral Interview Coach")
    st.markdown("Generate deep STAR-method behavioral questions for any competency scenario.")

    if st.session_state.profile is None:
        st.info("👈 Please generate your interview prep package first.")
        st.markdown("""
        The Behavioral Coach generates:
        - **3 STAR-based questions** per competency scenario
        - **Competency breakdown** showing what each question assesses
        - **Strong answer guidance** with bullet-point structure
        - **Red flag awareness** — mistakes to avoid
        - **STAR answer template** specific to each scenario
        - **Scoring rubric** comparing 5/10 vs 9/10 answers
        """)
    else:
        profile = st.session_state.profile

        BEHAVIORAL_SCENARIOS = [
            "Leadership, ownership, and taking initiative",
            "Conflict resolution and difficult team dynamics",
            "Handling failure, setbacks, and learning from mistakes",
            "Working under pressure and tight deadlines",
            "Innovation, creativity, and challenging the status quo",
            "Stakeholder management and cross-functional collaboration",
            "Mentoring, coaching, and knowledge sharing",
            "Adapting to change and ambiguity",
            "Ethical dilemmas and integrity under pressure",
            "Customer focus and going beyond expectations",
            "Data-driven decision making",
            "Custom scenario…",
        ]

        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            scenario_choice = st.selectbox(
                "Select a Behavioral Competency Scenario",
                BEHAVIORAL_SCENARIOS,
                help="Select a scenario to generate 3 STAR-based behavioral questions",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_beh = st.button("🎭 Generate", use_container_width=True, type="primary")

        if scenario_choice == "Custom scenario…":
            scenario = st.text_input(
                "Describe your custom scenario",
                placeholder="e.g. Managing a remote cross-cultural team through a critical delivery",
            )
        else:
            scenario = scenario_choice

        if generate_beh:
            if not scenario:
                st.warning("Please enter a scenario.")
            else:
                with st.spinner("⚡ Generating behavioral assessment with IBM Granite…"):
                    try:
                        agent = get_agent()
                        result = agent.get_behavioral_assessment(profile, scenario)
                        st.session_state.behavioral_result = result
                    except Exception as e:
                        st.error(f"Error: {e}")

        # Display latest behavioral result
        display_beh = st.session_state.behavioral_result
        if not display_beh and st.session_state.results:
            display_beh = st.session_state.results.get("behavioral_assessment", "")

        if display_beh:
            if st.session_state.behavioral_result:
                st.markdown(f"**Scenario**: *{scenario}*")
            else:
                st.markdown("📌 **Default behavioral assessment from your prep package:**")

            st.markdown(
                f'<div class="output-box">{display_beh}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                label="⬇️ Download Behavioral Assessment",
                data=display_beh,
                file_name=f"behavioral_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Improvement Tips
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("💡 Interview Improvement Tips")

    if st.session_state.profile is None:
        st.info("👈 Please generate your interview prep package first.")
    else:
        profile = st.session_state.profile

        col_tip1, col_tip2 = st.columns([2, 1])
        with col_tip1:
            st.markdown(
                f"Get **7 high-impact, personalized tips** for **{profile.name}** "
                f"targeting **{profile.job_role}** ({profile.experience_level} level)"
            )
        with col_tip2:
            get_tips_btn = st.button(
                "💡 Generate Personalized Tips",
                use_container_width=True,
                type="primary",
            )

        if get_tips_btn:
            with st.spinner("⚡ IBM Granite is crafting your personalized tips…"):
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
            st.download_button(
                label="⬇️ Download Tips",
                data=st.session_state.tips,
                file_name=f"improvement_tips_{profile.name.replace(' ', '_')}.txt",
                mime="text/plain",
            )

    # Universal tips — always visible
    st.divider()
    st.markdown("### 📌 Universal Interview Success Principles")
    tips_data = [
        ("🎯", "Research Deeply", "Know the company's products, culture, tech stack, recent news, and your interviewer's background. Most candidates don't — this alone sets you apart."),
        ("⭐", "Master STAR Method", "Situation (15%) → Task (15%) → Action (55%) → Result with metrics (15%). Every behavioral answer should follow this distribution."),
        ("📊", "Quantify Everything", "Replace 'improved performance' with 'improved latency by 40%, reducing error rate from 8% to 0.3%, impacting 2M users.' Numbers close offers."),
        ("❓", "Ask Great Questions", "Prepare 5 specific, research-backed questions. 'What does the on-call rotation look like?' beats 'What does success look like?'"),
        ("🧘", "Pause Before Answering", "3–5 seconds of silent thinking reads as confidence and intelligence. Rushed answers often reveal weak preparation."),
        ("💼", "Tailor Every Answer", "Generic answers fail. Reference the company's products, engineering blog, or recent news — it shows genuine interest and research depth."),
    ]

    tip_cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(tips_data):
        with tip_cols[i % 2]:
            with st.expander(f"{icon} {title}"):
                st.markdown(desc)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 6 — Session Summary
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("📊 Session Summary & Export")

    if not st.session_state.results or not st.session_state.profile:
        st.info("👈 Generate your interview prep package to see the session summary.")
        st.markdown("""
        After generating your prep package, this tab will show:
        - 📋 Complete session overview with candidate profile
        - 📊 What was generated (questions, strategy, behavioral assessment)
        - 📥 Full session export (all outputs in one file)
        - 🔄 Option to start a new session
        """)
    else:
        profile = st.session_state.profile
        results = st.session_state.results

        # Session overview card
        st.markdown("### 👤 Candidate Profile")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"""
            <div class="profile-card">
                <b>Name:</b> {profile.name}<br>
                <b>Target Role:</b> {profile.job_role}<br>
                <b>Level:</b> {profile.experience_level} ({profile.years_of_experience} years)<br>
                <b>Interview Type:</b> {profile.interview_type}
            </div>
            """, unsafe_allow_html=True)
        with col_p2:
            st.markdown(f"""
            <div class="profile-card">
                <b>Education:</b> {profile.education or "Not specified"}<br>
                <b>Target Company:</b> {profile.target_company or "Not specified"}<br>
                <b>Skills:</b> {profile.skills_str[:80] + "…" if len(profile.skills_str) > 80 else profile.skills_str}<br>
                <b>AI Engine:</b> {"IBM Granite (Live)" if results.get("ibm_granite_live") else "Demo Mode"}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📋 Generated Content Summary")
        content_cols = st.columns(4)
        content_items = [
            ("📋", "Question Set", bool(results.get("question_set"))),
            ("📅", "Prep Strategy", bool(results.get("preparation_strategy"))),
            ("🎭", "Behavioral Assessment", bool(results.get("behavioral_assessment"))),
            ("💡", "Improvement Tips", bool(st.session_state.tips)),
        ]
        for col, (icon, label, done) in zip(content_cols, content_items):
            with col:
                status = "✅ Generated" if done else "⏳ Not Yet"
                st.metric(f"{icon} {label}", status)

        # Full session export
        st.divider()
        st.markdown("### 📥 Full Session Export")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        full_export = f"""# AICTE Interview Trainer Agent — Session Report
# Generated: {now_str}
# AI Engine: {"IBM Granite (Live)" if results.get("ibm_granite_live") else "Demo Mode (set WATSONX_API_KEY for live)"}

{'=' * 70}
CANDIDATE PROFILE
{'=' * 70}
{profile.to_context_string()}

{'=' * 70}
INTERVIEW QUESTION SET
{'=' * 70}
{results.get("question_set", "Not generated")}

{'=' * 70}
7-DAY PREPARATION STRATEGY
{'=' * 70}
{results.get("preparation_strategy", "Not generated")}

{'=' * 70}
BEHAVIORAL ASSESSMENT
{'=' * 70}
{results.get("behavioral_assessment", "Not generated")}
"""
        if st.session_state.tips:
            full_export += f"\n{'=' * 70}\nIMPROVEMENT TIPS\n{'=' * 70}\n{st.session_state.tips}\n"

        if st.session_state.eval_result:
            ev = st.session_state.eval_result
            full_export += (
                f"\n{'=' * 70}\nANSWER EVALUATION\n{'=' * 70}\n"
                f"Question: {ev.question}\n\nYour Answer: {ev.candidate_answer}\n\n"
                f"Score: {ev.score}/10\n\nFeedback:\n{ev.raw_feedback}\n"
            )

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📥 Download Full Session Report (TXT)",
                data=full_export,
                file_name=f"interview_prep_{profile.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True,
                type="primary",
            )
        with dl_col2:
            session_json = json.dumps({
                "generated_at": now_str,
                "candidate": {
                    "name": profile.name,
                    "job_role": profile.job_role,
                    "experience_level": profile.experience_level,
                    "years": profile.years_of_experience,
                    "skills": profile.skills,
                    "target_company": profile.target_company,
                    "interview_type": profile.interview_type,
                    "education": profile.education,
                },
                "ai_engine": "IBM Granite (Live)" if results.get("ibm_granite_live") else "Demo Mode",
                "outputs": {
                    "question_set": results.get("question_set", ""),
                    "preparation_strategy": results.get("preparation_strategy", ""),
                    "behavioral_assessment": results.get("behavioral_assessment", ""),
                    "improvement_tips": st.session_state.tips or "",
                },
            }, indent=2)
            st.download_button(
                label="📥 Download Session Data (JSON)",
                data=session_json,
                file_name=f"interview_prep_{profile.name.replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True,
            )

        # New session button
        st.divider()
        if st.button("🔄 Start New Session (Clear All Data)", use_container_width=True):
            for key in ["profile", "results", "eval_result", "tips", "behavioral_result", "prep_started"]:
                st.session_state[key] = None
            st.rerun()
