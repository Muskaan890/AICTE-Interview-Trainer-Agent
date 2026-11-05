"""
interview_engine.py – Core interview question generation and assessment engine.

Orchestrates:
  - Question set generation (Technical, Behavioral, HR, Situational)
  - Per-answer evaluation and scoring
  - Session management
  - Improvement tip generation
"""

import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger

from src.granite_llm import GraniteLLM
from src.rag_pipeline import RAGPipeline
from src.user_profile import UserProfile


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

class InterviewQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question: str
    category: str  # Technical | Behavioral | HR | Situational
    difficulty: str  # Easy | Medium | Hard
    skill_assessed: str = ""
    model_answer_hint: str = ""


class QuestionSet(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    candidate_name: str
    job_role: str
    experience_level: str
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    raw_output: str = ""
    questions: List[InterviewQuestion] = Field(default_factory=list)
    total_questions: int = 0


class AnswerEvaluation(BaseModel):
    question: str
    candidate_answer: str
    score: float = 0.0
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    model_answer: str = ""
    follow_up_questions: List[str] = Field(default_factory=list)
    raw_feedback: str = ""


class InterviewSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    profile: UserProfile
    question_set: Optional[QuestionSet] = None
    evaluations: List[AnswerEvaluation] = Field(default_factory=list)
    preparation_strategy: str = ""
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    overall_score: float = 0.0

    def compute_overall_score(self) -> float:
        if not self.evaluations:
            return 0.0
        scores = [e.score for e in self.evaluations if e.score > 0]
        self.overall_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        return self.overall_score


# ──────────────────────────────────────────────────────────────────────────────
# Interview Engine
# ──────────────────────────────────────────────────────────────────────────────

class InterviewEngine:
    """
    Core engine that ties together the RAG pipeline and IBM Granite LLM
    to deliver a full interview coaching experience.
    """

    def __init__(self, llm: GraniteLLM, rag: RAGPipeline):
        self.llm = llm
        self.rag = rag
        logger.info(
            f"InterviewEngine ready | IBM Granite: {'LIVE' if llm.is_live else 'MOCK'}"
        )

    # ── Question Generation ──────────────────────────────────────────────────

    def generate_question_set(
        self,
        profile: UserProfile,
        num_questions: int = 10,
        categories: Optional[List[str]] = None,
    ) -> QuestionSet:
        """
        Generate a tailored question set for the candidate using RAG + Granite.
        """
        logger.info(f"Generating {num_questions} questions for {profile.name} — {profile.job_role}")

        # Retrieve relevant context from the knowledge base
        query = f"{profile.job_role} {profile.experience_level} interview questions {profile.skills_str}"
        context = self.rag.build_context(query=query, job_role=profile.job_role)

        # Generate via IBM Granite
        raw_output = self.llm.generate_questions(
            name=profile.name,
            job_role=profile.job_role,
            experience_level=profile.experience_level,
            skills=profile.skills_str,
            context=context,
            num_questions=num_questions,
        )

        question_set = QuestionSet(
            candidate_name=profile.name,
            job_role=profile.job_role,
            experience_level=profile.experience_level,
            raw_output=raw_output,
            total_questions=num_questions,
        )
        return question_set

    # ── Answer Evaluation ────────────────────────────────────────────────────

    def evaluate_answer(
        self,
        profile: UserProfile,
        question: str,
        answer: str,
    ) -> AnswerEvaluation:
        """
        Evaluate a candidate's answer to a specific question.
        Returns a scored AnswerEvaluation with feedback.
        """
        logger.info(f"Evaluating answer for: {profile.name}")

        context = self.rag.build_context(
            query=f"{question} {profile.job_role} model answer evaluation",
            job_role=profile.job_role,
        )

        raw_feedback = self.llm.evaluate_answer(
            name=profile.name,
            job_role=profile.job_role,
            experience_level=profile.experience_level,
            question=question,
            answer=answer,
            context=context,
        )

        evaluation = AnswerEvaluation(
            question=question,
            candidate_answer=answer,
            raw_feedback=raw_feedback,
        )

        # Attempt to parse score from response
        evaluation.score = self._extract_score(raw_feedback)
        return evaluation

    # ── Preparation Strategy ─────────────────────────────────────────────────

    def generate_preparation_strategy(self, profile: UserProfile) -> str:
        """
        Generate a personalized multi-day interview preparation plan.
        """
        logger.info(f"Generating preparation strategy for {profile.name}")

        context = self.rag.build_context(
            query=f"{profile.job_role} interview preparation strategy tips resources",
            job_role=profile.job_role,
        )

        return self.llm.create_preparation_strategy(
            name=profile.name,
            job_role=profile.job_role,
            experience_level=profile.experience_level,
            skills=profile.skills_str,
            resume_summary=profile.resume_summary or "Not provided.",
            context=context,
        )

    # ── Behavioral Assessment ────────────────────────────────────────────────

    def behavioral_assessment(
        self,
        profile: UserProfile,
        scenario: str,
    ) -> str:
        """
        Generate behavioral STAR questions for a specific competency scenario.
        """
        context = self.rag.build_context(
            query=f"behavioral interview STAR {scenario} {profile.job_role}",
            job_role=profile.job_role,
        )

        return self.llm.behavioral_assessment(
            name=profile.name,
            job_role=profile.job_role,
            scenario=scenario,
            context=context,
        )

    # ── Full Session ─────────────────────────────────────────────────────────

    def start_session(self, profile: UserProfile) -> InterviewSession:
        """Initialize a new interview session for a candidate."""
        session = InterviewSession(profile=profile)
        logger.info(f"New session started: {session.session_id} for {profile.name}")
        return session

    def run_full_preparation(
        self,
        profile: UserProfile,
        num_questions: int = 10,
    ) -> Dict[str, Any]:
        """
        Run the complete interview preparation pipeline and return all results.
        """
        session = self.start_session(profile)

        # 1. Generate questions
        question_set = self.generate_question_set(profile, num_questions=num_questions)
        session.question_set = question_set

        # 2. Generate preparation strategy
        strategy = self.generate_preparation_strategy(profile)
        session.preparation_strategy = strategy

        # 3. Behavioral assessment (common competency)
        behavioral = self.behavioral_assessment(
            profile=profile,
            scenario="leadership and conflict resolution",
        )

        return {
            "session_id": session.session_id,
            "profile": profile.to_context_string(),
            "question_set": question_set.raw_output,
            "preparation_strategy": strategy,
            "behavioral_assessment": behavioral,
            "ibm_granite_live": self.llm.is_live,
        }

    # ── Utilities ────────────────────────────────────────────────────────────

    def _extract_score(self, text: str) -> float:
        """Extract a numerical score (1-10) from LLM feedback text."""
        import re
        # Pattern: "Score: 7.5/10" or "7/10" or "Score: 8"
        patterns = [
            r'[Ss]core[:\s]+(\d+(?:\.\d+)?)\s*(?:/\s*10)?',
            r'(\d+(?:\.\d+)?)\s*/\s*10',
            r'[Rr]ating[:\s]+(\d+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                val = float(match.group(1))
                return min(max(val, 0.0), 10.0)
        return 0.0

    def get_improvement_tips(self, profile: UserProfile) -> str:
        """Generate quick improvement tips based on profile and role."""
        context = self.rag.build_context(
            query=f"{profile.job_role} interview tips improvement common mistakes",
        )
        prompt = (
            f"Based on a {profile.experience_level} {profile.job_role} candidate named {profile.name} "
            f"with skills: {profile.skills_str}, provide 5 concise, actionable interview improvement tips. "
            f"Context: {context[:1000]}"
        )
        return self.llm.generate(prompt)
