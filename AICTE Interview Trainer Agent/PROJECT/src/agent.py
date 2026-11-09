"""
agent.py – Main Interview Trainer Agent orchestrator.

The Agent class is the top-level entry point that:
  1. Initialises all subsystems (RAG, IBM Granite LLM, Interview Engine)
  2. Manages the end-to-end user interaction flow
  3. Exposes a clean API consumed by both the Streamlit UI and CLI
"""

from typing import Optional, Dict, Any, List
from loguru import logger

from src.config import config
from src.granite_llm import GraniteLLM
from src.rag_pipeline import RAGPipeline
from src.user_profile import ProfileBuilder, UserProfile
from src.interview_engine import InterviewEngine, AnswerEvaluation


class InterviewTrainerAgent:
    """
    IBM Granite-powered Interview Trainer Agent.

    Workflow:
        agent = InterviewTrainerAgent()
        agent.setup()
        profile = agent.create_profile(name="Alice", job_role="Data Scientist", ...)
        results = agent.run_full_preparation(profile)
        evaluation = agent.evaluate_answer(profile, question="...", answer="...")
    """

    _instance: Optional["InterviewTrainerAgent"] = None

    def __init__(self):
        self.llm = GraniteLLM()
        self.rag = RAGPipeline()
        self.engine = InterviewEngine(llm=self.llm, rag=self.rag)
        self.profile_builder = ProfileBuilder()
        self._ready = False

    # ── Singleton factory ────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "InterviewTrainerAgent":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.setup()
        return cls._instance

    # ── Initialization ───────────────────────────────────────────────────────

    def setup(self, ingest_kb: bool = True) -> Dict[str, Any]:
        """
        Bootstrap the agent:
          - Ingest knowledge base documents into the vector store
          - Verify IBM Granite connectivity
        Returns a status dict.
        """
        logger.info("=== Interview Trainer Agent — Startup ===")

        kb_chunks = 0
        if ingest_kb:
            try:
                kb_chunks = self.rag.ingest_knowledge_base()
                logger.info(f"Knowledge base ingested: {kb_chunks} chunks")
            except Exception as e:
                logger.warning(f"KB ingestion warning: {e}")

        vs_stats = self.rag.stats()
        self._ready = True

        status = {
            "ready": True,
            "ibm_granite_live": self.llm.is_live,
            "model": config.watsonx.model_id,
            "kb_chunks_ingested": kb_chunks,
            "vectorstore": vs_stats,
        }
        logger.info(f"Agent ready | IBM Granite live: {self.llm.is_live}")
        return status

    # ── Profile Management ───────────────────────────────────────────────────

    def create_profile(
        self,
        name: str,
        job_role: str,
        experience_level: Optional[str] = None,
        years_of_experience: int = 0,
        skills: Optional[List[str]] = None,
        resume_text: str = "",
        resume_file: Optional[str] = None,
        target_company: str = "",
        interview_type: str = "Full Stack",
        education: str = "",
    ) -> UserProfile:
        """Build a UserProfile from user inputs, enriched by resume parsing."""
        profile = self.profile_builder.build_from_inputs(
            name=name,
            job_role=job_role,
            experience_level=experience_level,
            years_of_experience=years_of_experience,
            skills=skills or [],
            resume_text=resume_text,
            resume_file=resume_file,
            target_company=target_company,
            interview_type=interview_type,
            education=education,
        )

        # Ingest the resume into the vector store for RAG context
        if profile.resume_text:
            self.rag.ingest_text(
                text=profile.resume_text,
                doc_id=f"resume_{profile.name.replace(' ', '_')}",
                metadata={"type": "resume", "candidate": profile.name, "role": profile.job_role},
            )

        logger.info(f"Profile created: {profile.name} | {profile.job_role} | {profile.experience_level}")
        return profile

    # ── Core Capabilities ────────────────────────────────────────────────────

    def generate_questions(
        self,
        profile: UserProfile,
        num_questions: int = 10,
    ) -> str:
        """Generate a tailored interview question set for the candidate."""
        result = self.engine.generate_question_set(profile, num_questions=num_questions)
        return result.raw_output

    def evaluate_answer(
        self,
        profile: UserProfile,
        question: str,
        answer: str,
    ) -> AnswerEvaluation:
        """Evaluate a candidate's answer and return scored feedback."""
        return self.engine.evaluate_answer(profile, question, answer)

    def get_preparation_strategy(self, profile: UserProfile) -> str:
        """Generate a personalized multi-day interview preparation plan."""
        return self.engine.generate_preparation_strategy(profile)

    def get_behavioral_assessment(
        self, profile: UserProfile, scenario: str = "leadership and conflict resolution"
    ) -> str:
        """Generate STAR-based behavioral interview questions for a scenario."""
        return self.engine.behavioral_assessment(profile, scenario)

    def get_improvement_tips(self, profile: UserProfile) -> str:
        """Return 5 actionable improvement tips for the candidate."""
        return self.engine.get_improvement_tips(profile)

    def run_full_preparation(
        self, profile: UserProfile, num_questions: int = 10
    ) -> Dict[str, Any]:
        """
        Run the full preparation pipeline and return all results in one call.
        Returns a dict with: question_set, preparation_strategy, behavioral_assessment.
        """
        return self.engine.run_full_preparation(profile, num_questions=num_questions)

    def ingest_resume_file(self, file_path: str) -> int:
        """Manually ingest a resume file into the RAG vector store."""
        return self.rag.ingest_resume(file_path)

    # ── Status ───────────────────────────────────────────────────────────────

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "ready": self._ready,
            "ibm_granite_live": self.llm.is_live,
            "model_id": config.watsonx.model_id,
            "vectorstore_stats": self.rag.stats(),
        }
