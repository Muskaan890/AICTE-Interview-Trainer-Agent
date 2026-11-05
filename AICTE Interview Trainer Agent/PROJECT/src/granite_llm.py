"""
granite_llm.py – IBM Granite LLM wrapper via watsonx.ai (IBM Cloud Lite).

Supports:
  - Text generation with IBM Granite models
  - Streaming responses
  - Prompt templating with role-specific context
"""

import os
from typing import Optional, Iterator
from loguru import logger

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False
    logger.warning("ibm-watsonx-ai not installed. Running in mock mode.")

from src.config import config


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Interview Trainer AI powered by IBM Granite.
Your role is to help candidates prepare for job interviews by:
1. Generating targeted, role-specific interview questions
2. Providing model answers and evaluation criteria
3. Offering improvement tips based on candidate responses
4. Assessing both technical competence and soft skills
5. Boosting candidate confidence and readiness

Always be encouraging, constructive, and precise. Tailor your responses to the
candidate's experience level and target job role."""


QUESTION_GENERATION_TEMPLATE = """<|system|>
{system_prompt}
<|user|>
Context from knowledge base:
{context}

Candidate Profile:
- Name: {name}
- Experience Level: {experience_level}
- Target Role: {job_role}
- Skills: {skills}

Task: Generate {num_questions} interview questions for this candidate.
Include a mix of:
- Technical/domain questions (based on the role)
- Behavioral questions (STAR method)
- Situational questions
- HR/culture-fit questions

For each question, provide:
1. The question
2. What the interviewer is assessing
3. A model answer framework
4. Difficulty level (Easy/Medium/Hard)

<|assistant|>"""


ANSWER_EVALUATION_TEMPLATE = """<|system|>
{system_prompt}
<|user|>
Context:
{context}

Candidate: {name} | Role: {job_role} | Experience: {experience_level}

Interview Question: {question}

Candidate's Answer: {answer}

Task: Evaluate this answer and provide:
1. Score (1-10) with justification
2. Strengths in the answer
3. Areas for improvement
4. A model/improved version of the answer
5. Follow-up questions an interviewer might ask

<|assistant|>"""


PREPARATION_STRATEGY_TEMPLATE = """<|system|>
{system_prompt}
<|user|>
Context from knowledge base:
{context}

Candidate Profile:
- Name: {name}
- Experience Level: {experience_level}
- Target Role: {job_role}
- Skills: {skills}
- Resume Summary: {resume_summary}

Task: Create a comprehensive 7-day interview preparation strategy that includes:
1. Day-wise preparation plan
2. Key topics to study/revise
3. Resources and materials
4. Mock interview schedule
5. Confidence-building exercises
6. Common pitfalls to avoid for this role

<|assistant|>"""


BEHAVIORAL_ASSESSMENT_TEMPLATE = """<|system|>
{system_prompt}
<|user|>
Context:
{context}

Candidate: {name} | Role: {job_role}

Behavioral Scenario: {scenario}

Task: 
1. Generate 3 STAR (Situation-Task-Action-Result) behavioral questions related to this scenario
2. Explain what competencies each question assesses
3. Provide guidance on structuring strong answers
4. Give example red flags interviewers watch for

<|assistant|>"""


# ──────────────────────────────────────────────────────────────────────────────
# IBM Granite Client
# ──────────────────────────────────────────────────────────────────────────────

class GraniteLLM:
    """
    Wrapper around IBM watsonx.ai Granite model for interview coaching tasks.
    Falls back to a descriptive mock when credentials are not set.
    """

    def __init__(self):
        self.cfg = config.watsonx
        self._model: Optional[object] = None
        self._mock_mode = False
        self._initialize()

    def _initialize(self):
        if not IBM_AVAILABLE:
            logger.warning("IBM watsonx-ai SDK not available. Using mock mode.")
            self._mock_mode = True
            return

        if not self.cfg.api_key or self.cfg.api_key == "your_ibm_cloud_api_key_here":
            logger.warning("WATSONX_API_KEY not set. Using mock mode.")
            self._mock_mode = True
            return

        try:
            credentials = Credentials(
                url=self.cfg.url,
                api_key=self.cfg.api_key,
            )
            self._model = ModelInference(
                model_id=self.cfg.model_id,
                credentials=credentials,
                project_id=self.cfg.project_id,
                params={
                    GenParams.MAX_NEW_TOKENS: self.cfg.max_new_tokens,
                    GenParams.TEMPERATURE: self.cfg.temperature,
                    GenParams.TOP_P: self.cfg.top_p,
                    GenParams.STOP_SEQUENCES: ["<|user|>", "<|system|>"],
                },
            )
            logger.info(f"IBM Granite model initialized: {self.cfg.model_id}")
        except Exception as exc:
            logger.error(f"Failed to initialize IBM Granite: {exc}")
            self._mock_mode = True

    # ── Public API ──────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using IBM Granite."""
        if self._mock_mode:
            return self._mock_response(prompt)
        try:
            response = self._model.generate_text(prompt=prompt)
            return response.strip()
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
    ) -> str:
        prompt = QUESTION_GENERATION_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            name=name,
            experience_level=experience_level,
            job_role=job_role,
            skills=skills,
            num_questions=num_questions,
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
    ) -> str:
        prompt = PREPARATION_STRATEGY_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            name=name,
            experience_level=experience_level,
            job_role=job_role,
            skills=skills,
            resume_summary=resume_summary,
        )
        return self.generate(prompt)

    def behavioral_assessment(
        self,
        name: str,
        job_role: str,
        scenario: str,
        context: str,
    ) -> str:
        prompt = BEHAVIORAL_ASSESSMENT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            name=name,
            job_role=job_role,
            scenario=scenario,
        )
        return self.generate(prompt)

    # ── Mock fallback ───────────────────────────────────────────────────────

    def _mock_response(self, prompt: str) -> str:
        """
        Structured mock response used when IBM credentials are absent.
        Provides realistic, useful output for demonstration/testing.
        """
        if "Generate" in prompt and "interview questions" in prompt:
            return _MOCK_QUESTIONS
        elif "Evaluate" in prompt and "Candidate's Answer" in prompt:
            return _MOCK_EVALUATION
        elif "preparation strategy" in prompt:
            return _MOCK_STRATEGY
        elif "behavioral" in prompt.lower() or "STAR" in prompt:
            return _MOCK_BEHAVIORAL
        return "IBM Granite response: Please configure WATSONX_API_KEY to enable live AI responses."

    @property
    def is_live(self) -> bool:
        return not self._mock_mode


# ──────────────────────────────────────────────────────────────────────────────
# Mock Responses (for demo / offline use)
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_QUESTIONS = """
## 📋 Interview Questions — Generated by IBM Granite

---

### 🔧 Technical Questions

**Q1. [Medium] Architecture Design**
Can you walk me through how you would design a scalable microservices architecture for a high-traffic e-commerce platform?

*Assessing: System design skills, scalability thinking, trade-off analysis*
*Model Answer Framework: Discuss services decomposition, API gateway, load balancing, caching, message queues (Kafka/RabbitMQ), database sharding.*

---

**Q2. [Hard] Problem Solving**
Describe a situation where you optimized a slow database query that was impacting production performance. What was your approach?

*Assessing: Debugging skills, database knowledge, performance optimization*
*Model Answer Framework: Identify query with EXPLAIN, add indexes, denormalize if needed, consider caching layer.*

---

**Q3. [Medium] Core Concepts**
Explain the difference between synchronous and asynchronous programming. When would you choose one over the other?

*Assessing: Fundamentals depth, real-world application understanding*
*Model Answer Framework: Define both, give language-specific examples (async/await, Promises, threads), discuss I/O-bound vs CPU-bound tasks.*

---

### 🤝 Behavioral Questions (STAR Method)

**Q4. [Medium] Teamwork**
Tell me about a time you disagreed with a team member's technical decision. How did you handle it?

*Assessing: Conflict resolution, collaboration, maturity*
*Model Answer Framework: Set the Situation → describe the Task → explain Actions taken (listening, proposing alternatives, escalation if needed) → share the Result.*

---

**Q5. [Medium] Leadership**
Describe a project where you took ownership despite it not being your direct responsibility. What drove you to do it?

*Assessing: Initiative, ownership mindset, accountability*

---

### 🎯 Situational Questions

**Q6. [Hard] Pressure Handling**
If you joined our team and discovered a critical security vulnerability in production on your first week — what would you do?

*Assessing: Crisis management, communication, prioritization*

---

### 💼 HR / Culture-Fit Questions

**Q7. [Easy] Motivation**
Why are you interested in this specific role at our company? What excites you most?

**Q8. [Easy] Career Goals**
Where do you see yourself professionally in 3-5 years?

**Q9. [Medium] Weakness**
What is your most significant professional weakness, and what steps are you actively taking to improve?

**Q10. [Easy] Cultural Contribution**
How do you contribute to a positive team culture and work environment?

---
*💡 Tip: Practice answers using the STAR method (Situation → Task → Action → Result) for maximum impact.*
"""

_MOCK_EVALUATION = """
## 📊 Answer Evaluation — IBM Granite Analysis

---

### 🎯 Overall Score: **7.5 / 10**

**Justification:** The answer demonstrates solid foundational knowledge and relevant experience. The candidate shows clear understanding of the concept but could strengthen their response with more quantifiable outcomes and specific technical details.

---

### ✅ Strengths

1. **Clear Structure** — The answer follows a logical progression, making it easy to follow
2. **Relevant Experience** — Good use of past experience to contextualize the response
3. **Technical Accuracy** — Core concepts are correctly identified and explained
4. **Communication** — Articulate and confident delivery (implied by written response)

---

### 🔧 Areas for Improvement

1. **Quantify Results** — Add metrics: "reduced latency by 40%", "handled 10K concurrent users"
2. **Deeper Technical Depth** — Elaborate on the specific tools/technologies used
3. **Contrast Alternatives** — Explain why you chose this approach over alternatives
4. **STAR Completion** — The "Result" portion needs more emphasis on business impact

---

### 💡 Improved Model Answer

"In my previous role at [Company], I was tasked with optimizing our payment processing pipeline that was experiencing 8-second response times during peak load (Situation/Task). 

I began by profiling the system using New Relic and identified three bottlenecks: N+1 database queries, missing indexes on the transactions table, and synchronous external API calls. I refactored the queries using eager loading, added composite indexes, and moved the external API calls to an async queue using Celery + Redis (Action).

As a result, response times dropped from 8 seconds to under 400ms — a 95% improvement — directly reducing cart abandonment by 23% and increasing monthly revenue by $180K (Result)."

---

### 🔮 Likely Follow-Up Questions

1. What monitoring tools do you use in production systems?
2. How did you communicate the technical trade-offs to non-technical stakeholders?
3. What would you do differently if you approached this problem today?
"""

_MOCK_STRATEGY = """
## 📅 7-Day Interview Preparation Strategy — IBM Granite

---

### 🎯 Overview
This personalized plan is designed to maximize your interview readiness within one week through structured study, practice, and confidence building.

---

### Day 1 — Know Your Story 📖
- **Morning:** Craft your 2-minute personal introduction (elevator pitch)
- **Afternoon:** Map your experience to the STAR method for 10 key scenarios
- **Evening:** Research the target company (mission, products, recent news, competitors)
- **Goal:** Have a compelling, authentic narrative ready

---

### Day 2 — Technical Foundation 🛠️
- **Morning:** Review core technical concepts for your role (data structures, algorithms, or domain-specific tools)
- **Afternoon:** Practice 5-8 technical problems on LeetCode/HackerRank (Medium level)
- **Evening:** Review system design fundamentals if applicable
- **Goal:** Sharpen technical fundamentals

---

### Day 3 — Role-Specific Deep Dive 🎯
- **Morning:** Study job description in depth — map every requirement to your experience
- **Afternoon:** Research industry trends relevant to the role
- **Evening:** Prepare 5 intelligent questions to ask the interviewer
- **Goal:** Become a domain expert in your target role

---

### Day 4 — Behavioral Mastery 🤝
- **Morning:** Practice 15 behavioral questions using STAR format
- **Afternoon:** Record yourself answering — review body language and pacing
- **Evening:** Refine weak answers, focus on quantifiable results
- **Goal:** Smooth, confident behavioral responses

---

### Day 5 — Mock Interviews 🎭
- **Morning:** Full mock technical interview (45 min) with a peer or online platform
- **Afternoon:** Full mock HR/behavioral interview
- **Evening:** Review feedback, identify gaps
- **Goal:** Simulate real interview pressure

---

### Day 6 — Gap Closing + Refinement 🔧
- **Morning:** Address top 3 weaknesses identified in mock interviews
- **Afternoon:** Practice salary negotiation talking points
- **Evening:** Prepare logistics (attire, documents, location/Zoom setup)
- **Goal:** Close remaining gaps and eliminate surprises

---

### Day 7 — Confidence Day 🌟
- **Morning:** Light review of key notes (no heavy studying)
- **Afternoon:** Relaxation techniques — meditation, light exercise
- **Evening:** Early sleep, positive visualization
- **Goal:** Arrive mentally sharp and confident

---

### ⚠️ Common Pitfalls to Avoid
1. **Vague answers** — Always support claims with specific examples
2. **Badmouthing past employers** — Stay professional and constructive
3. **Not asking questions** — Shows disengagement
4. **Memorized answers** — Sound natural, not rehearsed
5. **Underselling yourself** — Quantify your impact consistently

---

### 📚 Recommended Resources
- **Books:** "Cracking the Coding Interview" by G. McDowell, "The STAR Interview" by C. Miller
- **Platforms:** LeetCode, Pramp, Interviewing.io
- **IBM Learning:** IBM SkillsBuild, Coursera IBM courses
"""

_MOCK_BEHAVIORAL = """
## 🎭 Behavioral Assessment — STAR Questions

---

### Scenario Analysis

**Competencies Being Assessed:**
- Leadership under ambiguity
- Cross-functional collaboration  
- Decision-making with incomplete information
- Resilience and adaptability

---

### Question 1: Initiative & Ownership
*"Describe a time when you identified a problem that no one else noticed and took steps to address it without being asked."*

**Competencies:** Proactiveness, ownership, problem identification
**Strong Answer Guidance:** 
- Clearly describe the gap/problem you observed
- Explain your decision process to act vs. escalate
- Detail the concrete steps you took
- Quantify the positive outcome

**Red Flags Interviewers Watch For:**
- Taking credit for team efforts
- No measurable outcome mentioned
- Acting unilaterally without any communication

---

### Question 2: Conflict & Collaboration
*"Tell me about a time you had to collaborate with a difficult team member to deliver a critical project. How did you manage the relationship?"*

**Competencies:** Emotional intelligence, conflict resolution, team dynamics
**Strong Answer Guidance:**
- Acknowledge the challenge without blaming
- Focus on your actions, not the other person's faults
- Show empathy and mature communication
- End with a positive resolution and learning

**Red Flags:**
- Blaming the other person entirely
- No evidence of self-reflection
- Outcome was avoidance, not resolution

---

### Question 3: Failure & Learning
*"Describe your most significant professional failure. What happened, and what did you learn from it?"*

**Competencies:** Self-awareness, growth mindset, accountability
**Strong Answer Guidance:**
- Choose a real failure (not a disguised success)
- Take clear ownership — avoid blame-shifting
- Show structured reflection on root causes
- Demonstrate concrete changes you made afterward

**Red Flags:**
- Inability to identify a genuine failure
- Excessive self-deprecation
- No evidence of learning or behavior change

---

*💡 Pro Tip: Spend 30% of answer on S+T (context), 50% on A (your specific actions), 20% on R (measurable results + learnings).*
"""
