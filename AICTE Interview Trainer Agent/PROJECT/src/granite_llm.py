"""
granite_llm.py – IBM Granite LLM wrapper via watsonx.ai (IBM Cloud Lite).

Supports:
  - Text generation with IBM Granite 3.x / 13b-instruct models
  - Role-specific prompt templating using native Granite chat format
  - Graceful fallback to rich structured mock responses when credentials absent
"""

import os
from typing import Optional
from loguru import logger

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    logger.warning("ibm-watsonx-ai not installed. Running in demo mode.")

from src.config import config


# ──────────────────────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Interview Trainer AI powered by IBM Granite (watsonx.ai).
Your mission is to maximise every candidate's interview success rate by:
1. Generating targeted, role-specific interview questions matched to the candidate's level
2. Providing model answer frameworks with evaluation criteria
3. Scoring candidate responses objectively (1-10) with specific, actionable feedback
4. Creating day-by-day interview preparation strategies
5. Generating STAR-based behavioral questions for any competency scenario
6. Offering concise, impactful improvement tips

Always be encouraging yet honest. Tailor every response to the candidate's exact
experience level, target role, and skills. Use markdown formatting for clarity.
Never be generic — every question and tip must be specific to the candidate."""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Templates (IBM Granite Native Chat Format)
# ──────────────────────────────────────────────────────────────────────────────

QUESTION_GENERATION_TEMPLATE = """<|system|>
{system_prompt}
<|user|>
## Retrieved Context from Knowledge Base
{context}

## Candidate Profile
- **Name**: {name}
- **Target Role**: {job_role}
- **Experience Level**: {experience_level}
- **Skills**: {skills}
- **Target Company**: {target_company}
- **Interview Type**: {interview_type}

## Task
Generate exactly {num_questions} tailored interview questions for {name} applying as **{job_role}**.

Distribution based on interview type "{interview_type}":
- Technical/Domain questions (role-specific, skills-based)
- Behavioral questions (STAR method framework)
- Situational/Scenario questions (decision-making)
- HR/Culture-fit questions

For EACH question provide:
**Q[N]. [Difficulty: Easy/Medium/Hard] — [Category]**
*Question text here*

📌 **What the interviewer assesses**: [specific competency]
💡 **Model answer framework**: [concise guidance on ideal answer structure]

Tailor difficulty to {experience_level} level. Use role-specific technical terminology.
<|assistant|>"""


ANSWER_EVALUATION_TEMPLATE = """<|system|>
{system_prompt}
<|user|>
## Context
{context}

## Candidate
- **Name**: {name}
- **Target Role**: {job_role}
- **Experience Level**: {experience_level}

## Interview Question
{question}

## Candidate's Answer
{answer}

## Task
Evaluate this answer with professional rigour. Provide:

### 🎯 Score: [X]/10
**Justification**: [1-2 sentences explaining the score]

### ✅ Strengths (what worked well)
- [Strength 1]
- [Strength 2]

### 🔧 Areas for Improvement
- [Specific improvement 1 with actionable guidance]
- [Specific improvement 2]

### 💡 Improved Model Answer
[Write a significantly stronger version of the answer, incorporating specific metrics,
the STAR structure where appropriate, and concrete technical details]

### 🔮 Follow-up Questions an Interviewer Would Ask
1. [Follow-up question 1]
2. [Follow-up question 2]

Be specific to the role and experience level. Score strictly but fairly.
<|assistant|>"""


PREPARATION_STRATEGY_TEMPLATE = """<|system|>
{system_prompt}
<|user|>
## Knowledge Base Context
{context}

## Candidate Profile
- **Name**: {name}
- **Target Role**: {job_role}
- **Experience Level**: {experience_level}
- **Skills**: {skills}
- **Target Company**: {target_company}
- **Resume Summary**: {resume_summary}

## Task
Create a comprehensive **7-Day Interview Preparation Strategy** for {name} targeting **{job_role}**.

Structure the plan as follows:

### 📅 Day-by-Day Plan
For each day (Day 1–7), specify:
- Focus theme for the day
- Morning task (1 hour)
- Afternoon task (1.5 hours)
- Evening task (30 min)
- Daily goal/milestone

### 📚 Key Topics to Study (role-specific for {job_role})
List 8-10 specific topics with priority level (High/Medium/Low)

### 🛠️ Resources & Tools
Curated resources specific to {job_role} and {experience_level} level

### 🎭 Mock Interview Schedule
When and how to run mock interviews across the 7 days

### ⚠️ Common Pitfalls for {job_role} Candidates
5 specific mistakes candidates at {experience_level} level make

### 💪 Confidence-Building Exercises
3-4 practical exercises to build interview confidence
<|assistant|>"""


BEHAVIORAL_ASSESSMENT_TEMPLATE = """<|system|>
{system_prompt}
<|user|>
## Context
{context}

## Candidate
- **Name**: {name}
- **Target Role**: {job_role}
- **Experience Level**: {experience_level}

## Behavioral Scenario
**{scenario}**

## Task
Generate a comprehensive behavioral assessment for this scenario:

### 🎭 Scenario Overview
Brief explanation of why this competency matters for {job_role}

### Question 1 — [Specific sub-competency]
*[STAR behavioral question]*
- **Competencies assessed**: [list]
- **Strong answer guidance**: [3-4 bullet points]
- **Red flags to avoid**: [2-3 specific warning signs]

### Question 2 — [Specific sub-competency]
*[STAR behavioral question]*
- **Competencies assessed**: [list]
- **Strong answer guidance**: [3-4 bullet points]
- **Red flags to avoid**: [2-3 specific warning signs]

### Question 3 — [Specific sub-competency]
*[STAR behavioral question]*
- **Competencies assessed**: [list]
- **Strong answer guidance**: [3-4 bullet points]
- **Red flags to avoid**: [2-3 specific warning signs]

### 💡 STAR Answer Template for This Scenario
Provide a fill-in-the-blanks STAR template specific to {scenario}

### 📊 Scoring Rubric
What separates a 5/10 answer from a 9/10 answer for this competency
<|assistant|>"""


IMPROVEMENT_TIPS_TEMPLATE = """<|system|>
{system_prompt}
<|user|>
## Context
{context}

## Candidate
- **Name**: {name}
- **Target Role**: {job_role}
- **Experience Level**: {experience_level}
- **Skills**: {skills}

## Task
Generate **7 high-impact, personalized interview improvement tips** for {name} targeting {job_role}.

Each tip must be:
- Specific to {experience_level} level candidates applying for {job_role}
- Immediately actionable (can be implemented today)
- Backed by a concrete technique or framework

Format each tip as:

### Tip [N]: [Catchy Title]
**The Problem**: [1 sentence describing the common mistake]
**The Fix**: [2-3 sentences with specific, actionable guidance]
**Quick Practice**: [A 5-minute exercise to reinforce this tip]

End with a brief motivational closing paragraph.
<|assistant|>"""


# ──────────────────────────────────────────────────────────────────────────────
# IBM Granite Client
# ──────────────────────────────────────────────────────────────────────────────

class GraniteLLM:
    """
    Wrapper around IBM watsonx.ai Granite model for interview coaching tasks.
    Falls back to rich structured mock responses when credentials are not configured.
    """

    def __init__(self):
        self.cfg = config.watsonx
        self._model: Optional[object] = None
        self._mock_mode = False
        self._initialize()

    def _initialize(self):
        if not IBM_AVAILABLE:
            logger.warning("ibm-watsonx-ai SDK not available. Running in demo mode.")
            self._mock_mode = True
            return

        if not self.cfg.api_key or self.cfg.api_key in ("your_ibm_cloud_api_key_here", ""):
            logger.warning("WATSONX_API_KEY not configured. Running in demo mode.")
            self._mock_mode = True
            return

        try:
            credentials = Credentials(
                url=self.cfg.url,
                api_key=self.cfg.api_key,
            )
            params = {
                GenParams.MAX_NEW_TOKENS:      self.cfg.max_new_tokens,
                GenParams.TEMPERATURE:         self.cfg.temperature,
                GenParams.TOP_P:               self.cfg.top_p,
                GenParams.TOP_K:               self.cfg.top_k,
                GenParams.REPETITION_PENALTY:  self.cfg.repetition_penalty,
                GenParams.STOP_SEQUENCES:      ["<|user|>", "<|system|>", "<|endoftext|>"],
            }
            self._model = ModelInference(
                model_id=self.cfg.model_id,
                credentials=credentials,
                project_id=self.cfg.project_id,
                params=params,
            )
            logger.info(f"IBM Granite initialized: {self.cfg.model_id}")
        except Exception as exc:
            logger.error(f"IBM Granite init failed: {exc}")
            self._mock_mode = True

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using IBM Granite."""
        if self._mock_mode:
            return self._mock_response(prompt)
        try:
            response = self._model.generate_text(prompt=prompt)
            return response.strip() if response else self._mock_response(prompt)
        except Exception as exc:
            logger.error(f"IBM Granite generation error: {exc}")
            return self._mock_response(prompt)

    def generate_questions(
        self,
        name: str,
        job_role: str,
        experience_level: str,
        skills: str,
        context: str,
        num_questions: int = 10,
        target_company: str = "",
        interview_type: str = "Full Stack",
    ) -> str:
        prompt = QUESTION_GENERATION_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            name=name,
            experience_level=experience_level,
            job_role=job_role,
            skills=skills,
            num_questions=num_questions,
            target_company=target_company or "Not specified",
            interview_type=interview_type,
        )
        return self.generate(prompt)

    def evaluate_answer(
        self,
        name: str,
        job_role: str,
        experience_level: str,
        question: str,
        answer: str,
        context: str,
    ) -> str:
        prompt = ANSWER_EVALUATION_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            name=name,
            job_role=job_role,
            experience_level=experience_level,
            question=question,
            answer=answer,
        )
        return self.generate(prompt)

    def create_preparation_strategy(
        self,
        name: str,
        job_role: str,
        experience_level: str,
        skills: str,
        resume_summary: str,
        context: str,
        target_company: str = "",
    ) -> str:
        prompt = PREPARATION_STRATEGY_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            name=name,
            experience_level=experience_level,
            job_role=job_role,
            skills=skills,
            resume_summary=resume_summary,
            target_company=target_company or "Not specified",
        )
        return self.generate(prompt)

    def behavioral_assessment(
        self,
        name: str,
        job_role: str,
        experience_level: str,
        scenario: str,
        context: str,
    ) -> str:
        prompt = BEHAVIORAL_ASSESSMENT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            name=name,
            job_role=job_role,
            experience_level=experience_level,
            scenario=scenario,
        )
        return self.generate(prompt)

    def get_improvement_tips(
        self,
        name: str,
        job_role: str,
        experience_level: str,
        skills: str,
        context: str,
    ) -> str:
        prompt = IMPROVEMENT_TIPS_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            name=name,
            job_role=job_role,
            experience_level=experience_level,
            skills=skills,
        )
        return self.generate(prompt)

    # ── Mock fallback ──────────────────────────────────────────────────────

    def _mock_response(self, prompt: str) -> str:
        """Rich, structured mock responses for demo / offline use."""
        p = prompt.lower()
        if "generate" in p and "interview questions" in p.replace("\n", " "):
            return _MOCK_QUESTIONS
        elif "evaluate" in p and "candidate's answer" in p.replace("\n", " "):
            return _MOCK_EVALUATION
        elif "preparation strategy" in p or "7-day" in p:
            return _MOCK_STRATEGY
        elif "behavioral" in p or "star" in p:
            return _MOCK_BEHAVIORAL
        elif "improvement tips" in p or "improve" in p:
            return _MOCK_TIPS
        return (
            "### IBM Granite Demo Mode\n\n"
            "To enable live AI responses, configure your IBM watsonx.ai credentials in `.env`:\n\n"
            "```\nWATSONX_API_KEY=your_ibm_cloud_api_key\n"
            "WATSONX_PROJECT_ID=your_project_id\n```\n\n"
            "Get free IBM Cloud credentials at: https://cloud.ibm.com/registration"
        )

    @property
    def is_live(self) -> bool:
        return not self._mock_mode

    @property
    def model_name(self) -> str:
        return self.cfg.model_id.split("/")[-1]


# ──────────────────────────────────────────────────────────────────────────────
# Mock Responses (rich, structured output for demo / testing)
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_QUESTIONS = """
## 📋 Interview Question Set — Generated by IBM Granite

---

### 🔧 Technical Questions

**Q1. [Medium] — System Design**
Can you walk me through how you would design a scalable microservices architecture for a high-traffic e-commerce platform handling 1M+ daily active users?

📌 **What the interviewer assesses**: System design thinking, scalability mindset, trade-off analysis, knowledge of distributed systems
💡 **Model answer framework**: Decompose into services (auth, catalog, cart, payment, notification), API gateway, load balancer, Redis caching, Kafka for async events, PostgreSQL with read replicas, CDN for static assets.

---

**Q2. [Hard] — Performance Optimization**
Describe a situation where you diagnosed and resolved a critical performance bottleneck in a production system. What was your systematic approach?

📌 **What the interviewer assesses**: Debugging methodology, profiling tools knowledge, root cause analysis, production awareness
💡 **Model answer framework**: Use APM tools (Datadog/New Relic), identify top latency endpoints, profile DB queries with EXPLAIN ANALYZE, check N+1 issues, add indexes, introduce caching, measure impact.

---

**Q3. [Medium] — Core Concepts**
Explain the difference between synchronous and asynchronous programming. When would you choose one over the other in a real production scenario?

📌 **What the interviewer assesses**: Foundational programming depth, practical application understanding, I/O vs CPU-bound task awareness
💡 **Model answer framework**: Sync = blocking execution, good for CPU-bound; Async = non-blocking, ideal for I/O-bound (DB queries, HTTP calls). Give concrete language example (Python async/await, Node.js Promises).

---

**Q4. [Medium] — Security**
How do you secure a REST API? Walk me through the security layers you would implement.

📌 **What the interviewer assesses**: Security mindset, awareness of OWASP top 10, API design maturity
💡 **Model answer framework**: HTTPS/TLS, JWT/OAuth2 authentication, rate limiting, input validation, CORS policies, API gateway, audit logging, secrets management (HashiCorp Vault).

---

### 🤝 Behavioral Questions (STAR Method)

**Q5. [Medium] — Leadership & Initiative**
Tell me about a time you identified a critical problem that no one else on your team had noticed. How did you respond?

📌 **What the interviewer assesses**: Proactiveness, ownership mindset, problem identification skills, communication
💡 **Model answer framework**: Describe the gap/risk you spotted → explain your decision to act vs escalate → concrete actions taken → quantified outcome + team impact.

---

**Q6. [Medium] — Conflict Resolution**
Describe a time you disagreed with a team member's technical decision on an important project. How did you handle it?

📌 **What the interviewer assesses**: Conflict resolution maturity, technical conviction, collaborative communication, ego management
💡 **Model answer framework**: State the technical disagreement → your approach (data-driven argument, seek peer input) → resolution method → outcome and relationship preserved.

---

### 🎯 Situational Questions

**Q7. [Hard] — Crisis Management**
You've just joined our team. On your second week, you discover a critical security vulnerability in a production system affecting 500,000 users. What are the immediate steps you take?

📌 **What the interviewer assesses**: Crisis management, prioritisation under pressure, communication skills, security awareness
💡 **Model answer framework**: Immediately notify security & engineering lead → assess exploit risk → contain (rate limiting, feature flag off) → patch → post-mortem → communicate to affected users per policy.

---

**Q8. [Medium] — Prioritisation**
You have three competing deadlines this week and your manager has become unavailable. How do you decide what to work on?

📌 **What the interviewer assesses**: Time management, stakeholder communication, autonomous decision-making, escalation judgment
💡 **Model answer framework**: Classify by urgency × impact matrix → communicate with stakeholders of each task → document decision → deliver highest priority → update manager asynchronously.

---

### 💼 HR / Culture-Fit Questions

**Q9. [Easy] — Motivation**
Why are you interested in this specific role and company? What excites you most about joining?

📌 **What the interviewer assesses**: Research depth, genuine interest, alignment with company values, self-awareness
💡 **Model answer framework**: Company-specific research (products, mission, recent news) + role alignment to your strengths + growth opportunity + contribution you'd make.

---

**Q10. [Easy] — Career Vision**
Where do you see yourself professionally in 3–5 years, and how does this role fit that trajectory?

📌 **What the interviewer assesses**: Ambition, self-awareness, fit with role growth path, retention potential
💡 **Model answer framework**: Realistic progression tied to role → specific skills you'll develop → value you'll add at each stage → show you've thought through the path.

---

*💡 **IBM Granite Tip**: For behavioral questions, allocate — 15% Situation, 15% Task, 55% Action (YOUR specific steps), 15% Result with metrics. Practice timing: each answer should be 90–120 seconds.*
"""

_MOCK_EVALUATION = """
## 📊 Answer Evaluation — IBM Granite Analysis

---

### 🎯 Score: **7.5 / 10**

**Justification**: The answer demonstrates solid foundational knowledge and relevant real-world experience. The candidate shows clear understanding of the concept and uses appropriate terminology. However, the response lacks quantifiable outcomes and the STAR structure could be tightened for greater impact.

---

### ✅ Strengths

1. **Logical Structure** — The answer follows a clear, coherent progression that is easy to follow
2. **Relevant Experience** — Good use of past experience to contextualize the technical response
3. **Technical Accuracy** — Core concepts are correctly identified and the approach is sound
4. **Self-Awareness** — The candidate acknowledges tradeoffs rather than presenting a one-size-fits-all solution

---

### 🔧 Areas for Improvement

1. **Quantify Your Impact** — Replace vague claims with hard metrics: "reduced API latency by 40%", "handled 50K concurrent requests", "reduced costs by $12K/month"
2. **Deeper Technical Specificity** — Name the exact tools, libraries, and configuration decisions made — interviewers reward specificity
3. **Explain the Why** — Articulate *why* you chose this approach over alternatives; this demonstrates engineering judgment
4. **Stronger STAR Completion** — The "Result" section needs more emphasis on business/user impact, not just technical outcome
5. **Anticipate Follow-ups** — Proactively mention edge cases or limitations you considered

---

### 💡 Improved Model Answer

*"In my role at [Company], our payment processing service was experiencing 8-second P99 latency during peak hours, directly causing a 12% cart abandonment rate (Situation/Task).*

*I began by instrumenting the service with distributed tracing using Jaeger, which pinpointed three bottlenecks: N+1 database queries fetching payment methods, synchronous calls to an external fraud-check API, and missing composite indexes on the transactions table (Analysis).*

*I resolved the N+1 issue by refactoring to a single batched query with JOINs, moved the fraud-check API call to an async Celery task using Redis as the broker, and added three covering indexes after running EXPLAIN ANALYZE on slow query logs. I also introduced a 5-minute Redis cache for frequently-accessed payment configurations (Action).*

*The changes reduced P99 latency from 8 seconds to 380ms — a 95% improvement. Cart abandonment dropped by 23% within two weeks, adding an estimated $180K in monthly recovered revenue. The architecture was documented and became our team's standard pattern for external API integration (Result)."*

---

### 🔮 Likely Follow-Up Questions

1. How did you decide which optimizations to implement first? What was your prioritization framework?
2. How did you test the performance improvements before deploying to production?
3. What monitoring did you put in place to ensure the improvements held over time?

---

*⭐ **Score Breakdown**: Structure (7/10) · Technical Depth (8/10) · Quantified Impact (6/10) · Communication Clarity (8/10)*
"""

_MOCK_STRATEGY = """
## 📅 7-Day Interview Preparation Strategy — IBM Granite

---

### 🎯 Strategy Overview
This personalized plan is designed to systematically build your technical mastery,
behavioral storytelling, and interview confidence within 7 days.

---

### Day 1 — Know Your Story 📖
**Theme**: Personal Narrative & Company Research

- **Morning (1 hr)**: Craft your 2-minute personal introduction (elevator pitch). Structure: Current role → Key achievements with metrics → Why this role/company
- **Afternoon (1.5 hr)**: Deep-dive company research: mission, recent product launches, tech stack, engineering blog, Glassdoor reviews
- **Evening (30 min)**: Map 10 career achievements to the STAR framework; note specific metrics for each
- ✅ **Daily Goal**: Have a compelling, authentic narrative and know the company better than any other candidate

---

### Day 2 — Technical Foundation 🛠️
**Theme**: Core Technical Concepts

- **Morning (1 hr)**: Review data structures & algorithms fundamentals — focus on the top 10 patterns most common for your role
- **Afternoon (1.5 hr)**: Solve 5–8 LeetCode/HackerRank problems at Medium difficulty level
- **Evening (30 min)**: Review system design fundamentals (if applicable): load balancing, caching, databases
- ✅ **Daily Goal**: Solve 5 problems fluently and explain your approach out loud

---

### Day 3 — Role-Specific Deep Dive 🎯
**Theme**: Mastering the Job Description

- **Morning (1 hr)**: Dissect the job description line-by-line — map every requirement to a specific experience or project
- **Afternoon (1.5 hr)**: Study the top 5 technical areas specific to your target role using documentation and articles
- **Evening (30 min)**: Prepare 5 intelligent, research-backed questions to ask the interviewer
- ✅ **Daily Goal**: Become the subject-matter expert in your target role's requirements

---

### Day 4 — Behavioral Mastery 🤝
**Theme**: STAR Stories & Soft Skills

- **Morning (1 hr)**: Practice 15 behavioral questions using the STAR format — write out each answer
- **Afternoon (1.5 hr)**: Record yourself answering 5 key behavioral questions — review delivery, pacing, and body language
- **Evening (30 min)**: Refine the 3 weakest answers with stronger, more quantified results
- ✅ **Daily Goal**: Have 8+ polished STAR stories ready to deploy in any order

---

### Day 5 — Mock Interviews 🎭
**Theme**: Simulation Under Pressure

- **Morning (1 hr)**: Full 45-minute mock technical interview — use Pramp, Interviewing.io, or a peer
- **Afternoon (1.5 hr)**: Full mock HR + behavioral interview; debrief every question
- **Evening (30 min)**: Write down the top 3 weaknesses identified in today's mocks
- ✅ **Daily Goal**: Identify and address your biggest gaps with 2 days remaining

---

### Day 6 — Gap Closing & Polish 🔧
**Theme**: Eliminating Remaining Weaknesses

- **Morning (1 hr)**: Target your Day 5 identified gaps — study, practice, and refine those specific areas
- **Afternoon (1.5 hr)**: Research salary benchmarks (Glassdoor, Levels.fyi) — prepare negotiation talking points
- **Evening (30 min)**: Prepare logistics: attire, printed resume copies, Zoom/video setup test, commute route
- ✅ **Daily Goal**: Eliminate every known gap and remove all logistical surprises

---

### Day 7 — Confidence Day 🌟
**Theme**: Mental Preparation

- **Morning (1 hr)**: Light review of key notes only — no heavy studying
- **Afternoon (1.5 hr)**: Light exercise, proper nutrition, relaxation (meditation, walk)
- **Evening (30 min)**: Positive visualization, early sleep (8+ hours critical for cognitive performance)
- ✅ **Daily Goal**: Arrive mentally sharp, energized, and genuinely confident

---

### 📚 Top Resources

| Resource | Purpose |
|---|---|
| LeetCode (Medium) | Algorithm practice |
| Pramp | Free mock interviews |
| Grokking System Design | System design mastery |
| IBM SkillsBuild | IBM-specific learning |
| Glassdoor / Levels.fyi | Salary research |
| "Cracking the Coding Interview" | Technical prep book |

---

### ⚠️ 5 Critical Pitfalls to Avoid

1. **Vague Impact Claims** — "Improved performance" is not enough. Always have the number.
2. **Memorized Answers** — Interviewers detect scripts. Know your points, not your words.
3. **Badmouthing Past Employers** — Always reframe negatives constructively.
4. **Not Asking Questions** — Silence at the end signals disengagement; prepare 3+ smart questions.
5. **Underselling Yourself** — Quantify every achievement; modesty doesn't close offers.

---

*🏆 Powered by IBM Granite · Configure WATSONX_API_KEY for personalized AI-generated strategy*
"""

_MOCK_BEHAVIORAL = """
## 🎭 Behavioral Assessment — STAR Framework Questions

---

### Scenario Overview
Leadership and conflict resolution is one of the most heavily assessed competencies
across all roles and levels. Interviewers are evaluating your emotional intelligence,
your ability to influence without authority, and how you behave under social pressure.

---

### Question 1 — Initiative & Proactive Leadership
*"Describe a time when you identified a problem that no one else noticed and took the initiative to solve it without being formally asked."*

**Competencies assessed**: Ownership, analytical thinking, proactiveness, risk management
**Strong answer guidance**:
- Clearly describe the gap/risk you observed using specific data signals
- Explain your decision process: act vs. escalate, who you consulted
- Walk through 3-4 concrete steps you took, using "I" not "we"
- Quantify the impact: time saved, cost avoided, incidents prevented

**Red flags to avoid**:
- Taking sole credit for what was a team effort
- Describing an outcome with no measurable result
- Acting unilaterally without any communication or buy-in

---

### Question 2 — Managing Difficult Relationships
*"Tell me about a time you had to collaborate with someone whose working style conflicted significantly with yours to deliver a critical project."*

**Competencies assessed**: Emotional intelligence, conflict resolution, adaptability, team cohesion
**Strong answer guidance**:
- Acknowledge the challenge without making the other person the villain
- Focus entirely on YOUR actions, not their failings
- Show the specific steps you took to build the working relationship
- End with what you learned about yourself and how you now approach similar situations

**Red flags to avoid**:
- Blaming the other person entirely for the conflict
- Showing no evidence of self-reflection or adaptation
- Resolution that was avoidance, not genuine collaboration

---

### Question 3 — Owning Failure & Demonstrating Growth
*"Describe your most significant professional failure — one that had real consequences. What happened, and how did it change how you work?"*

**Competencies assessed**: Self-awareness, accountability, growth mindset, resilience, maturity
**Strong answer guidance**:
- Choose a REAL failure with real stakes — not a disguised success story
- Take clear ownership; avoid blame-shifting even when others contributed
- Use a structured root cause analysis: what went wrong and why
- Describe the concrete behavioural changes you made as a result — show growth is permanent

**Red flags to avoid**:
- Picking a trivial "failure" (interviewers see through it immediately)
- Excessive self-deprecation that creates doubt about your capability
- Failure story with no evidence of learning, change, or recovery

---

### 💡 STAR Answer Template for Leadership Scenarios

**Situation** (15%): "In [time period/context], our team/project faced [specific challenge or opportunity]..."
**Task** (15%): "I was responsible for [your specific role], which required me to [key responsibility]..."
**Action** (55%): "First, I [specific action 1 with reasoning]. Then I [specific action 2]. I also [specific action 3] because [rationale]. Throughout, I [communication/stakeholder management]..."
**Result** (15%): "As a result, [quantified outcome]. This led to [business impact]. I learned [key insight] which I now apply by [specific changed behavior]."

---

### 📊 Scoring Rubric: 5/10 vs 9/10

| Dimension | 5/10 Answer | 9/10 Answer |
|---|---|---|
| Specificity | Vague, generic story | Specific dates, names, context |
| Ownership | "We decided to..." | "I specifically..." |
| Quantification | "It improved a lot" | "Reduced by 34%, saving $50K" |
| Reflection | Thin or absent | Deep insight + behavior change |
| Delivery | Rambling, unstructured | Crisp STAR, 90-120 seconds |

---

*💡 Practice these questions by recording a 2-minute video response. Watch it back — you'll improve 40% just from self-review.*
"""

_MOCK_TIPS = """
## 💡 7 High-Impact Interview Improvement Tips — IBM Granite

---

### Tip 1: Quantify Everything with the "So What?" Test
**The Problem**: Most candidates describe what they did but not the impact — "I optimized the database" tells an interviewer almost nothing.
**The Fix**: After every achievement claim, ask yourself "So what?" until you reach a business metric. "Optimized the database → reduced query time by 60% → API response improved from 2s to 0.8s → user engagement increased 18% → estimated $200K additional annual revenue."
**Quick Practice**: Take your top 3 resume bullets and apply the "So What?" chain to each for 5 minutes. If you can't get to a number, dig deeper.

---

### Tip 2: Replace "We" with "I" Without Sounding Arrogant
**The Problem**: Excessive use of "we" makes it impossible for interviewers to understand your individual contribution — they're hiring you, not your team.
**The Fix**: Use "I" to describe your specific actions and decisions. Acknowledge the team for the outcome: "I designed the caching architecture, which the team then implemented together. The result was a 3x throughput improvement."
**Quick Practice**: Record yourself answering "Tell me about a project you're proud of" — count how many times you say "we" vs. "I." Aim for 70% I, 30% we.

---

### Tip 3: Master the 90-Second Answer Rule
**The Problem**: Candidates either ramble for 4+ minutes (exhausting the interviewer) or give 20-second non-answers that lack substance.
**The Fix**: Every behavioral answer should be 90–120 seconds. Use STAR as your clock: 15s Situation, 15s Task, 60s Action (the meat), 15s Result. Set a timer and practice until it feels natural.
**Quick Practice**: Pick any behavioral question and time your answer. Adjust until it lands between 90–120 seconds without feeling rushed.

---

### Tip 4: Prepare a "Conflict Without Casualties" Story
**The Problem**: When asked about conflict or disagreement, candidates either describe a non-conflict (to seem agreeable) or reveal they handled it poorly.
**The Fix**: Prepare one story where you disagreed respectfully, used data/logic to make your case, and reached a productive resolution — ideally one where you were right AND maintained the relationship. This is the gold standard.
**Quick Practice**: Write out the story in STAR format. Ensure it shows intellectual courage + emotional maturity + positive outcome.

---

### Tip 5: Use the "Pause-and-Frame" Technique for Hard Questions
**The Problem**: Candidates panic during unexpected or ambiguous questions and either go silent awkwardly or answer hastily with a poor response.
**The Fix**: Buy yourself thinking time professionally: "That's a great question — let me think through this properly." Then pause 3–5 seconds silently. Interviewers respect thoughtful pauses far more than rushed answers.
**Quick Practice**: Ask a friend to give you surprise questions. Practice pausing 3 seconds before answering every time — it will feel uncomfortable at first but reads as composed confidence.

---

### Tip 6: Research the Interviewer's LinkedIn Profile
**The Problem**: Generic "company research" is obvious and forgettable — every candidate does it. Personalised research creates memorable connection.
**The Fix**: Look up your interviewer on LinkedIn. Note their background, shared experiences, or projects they've worked on. Weave 1-2 genuine connections into conversation naturally: "I noticed you worked on [X] — I actually faced a similar challenge when..."
**Quick Practice**: Find the LinkedIn profile of 3 people likely to interview you for your target role and prepare one natural connection point for each.

---

### Tip 7: Prepare 5 Brilliant "Reverse Questions" to Ask the Interviewer
**The Problem**: "Do you have any questions for us?" is met with either silence or generic questions ("What does success look like?") — a missed opportunity to stand out.
**The Fix**: Prepare 5 specific, thoughtful questions that show you've researched deeply and think strategically. Examples: "I read about your recent migration to [technology] — what were the biggest technical challenges the team faced?" or "What does the on-call rotation look like, and how has the team improved response times recently?"
**Quick Practice**: Write 5 questions specific to your target company right now. Prioritise questions that show you've done real research.

---

*🎯 Mastering even 3 of these tips will put you in the top 10% of candidates. Powered by IBM Granite — configure WATSONX_API_KEY for personalized AI tips.*
"""
