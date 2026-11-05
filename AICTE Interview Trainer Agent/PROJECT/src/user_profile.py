"""
user_profile.py – Candidate profile management and resume parsing.

Handles:
  - Structured UserProfile data model
  - Resume text extraction (PDF, DOCX, plain text)
  - Skill extraction using keyword matching
  - Experience level classification
"""

import re
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from loguru import logger

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    name: str = Field(default="Candidate", description="Candidate's full name")
    job_role: str = Field(default="Software Engineer", description="Target job role")
    experience_level: str = Field(default="Mid-Level", description="Fresher / Junior / Mid-Level / Senior / Lead")
    years_of_experience: int = Field(default=0, description="Total years of experience")
    skills: List[str] = Field(default_factory=list, description="Technical and soft skills")
    education: str = Field(default="", description="Highest education qualification")
    resume_text: str = Field(default="", description="Full resume text")
    resume_summary: str = Field(default="", description="Short resume summary (auto-generated)")
    target_company: str = Field(default="", description="Company being interviewed at")
    interview_type: str = Field(default="Full Stack", description="Technical / Behavioral / HR / Full Stack")

    @property
    def skills_str(self) -> str:
        return ", ".join(self.skills) if self.skills else "General skills"

    def to_context_string(self) -> str:
        return (
            f"Candidate: {self.name}\n"
            f"Target Role: {self.job_role}\n"
            f"Experience Level: {self.experience_level} ({self.years_of_experience} years)\n"
            f"Skills: {self.skills_str}\n"
            f"Education: {self.education}\n"
            f"Target Company: {self.target_company or 'Not specified'}\n"
            f"Interview Type: {self.interview_type}\n"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Skill Taxonomy (role-specific keyword banks)
# ──────────────────────────────────────────────────────────────────────────────

SKILL_TAXONOMY: Dict[str, List[str]] = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "kotlin", "swift", "ruby", "php", "scala", "r", "matlab", "perl",
    ],
    "web_frameworks": [
        "react", "angular", "vue", "next.js", "django", "flask", "fastapi",
        "spring", "express", "node.js", "laravel", "rails", "asp.net",
    ],
    "databases": [
        "mysql", "postgresql", "mongodb", "redis", "cassandra", "dynamodb",
        "oracle", "sqlite", "elasticsearch", "neo4j", "db2",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "ibm cloud", "docker", "kubernetes", "terraform",
        "ansible", "jenkins", "github actions", "gitlab ci", "helm", "istio",
    ],
    "data_ai_ml": [
        "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn",
        "pandas", "numpy", "spark", "hadoop", "tableau", "power bi",
        "llm", "rag", "nlp", "computer vision", "watsonx", "langchain",
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem solving", "agile",
        "scrum", "project management", "mentoring", "collaboration", "critical thinking",
    ],
    "security": [
        "cybersecurity", "penetration testing", "owasp", "soc", "siem",
        "firewalls", "encryption", "iam", "zero trust",
    ],
}

ALL_SKILLS = [skill for skills in SKILL_TAXONOMY.values() for skill in skills]


# ──────────────────────────────────────────────────────────────────────────────
# Experience Level Classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_experience_level(years: int) -> str:
    if years == 0:
        return "Fresher"
    elif years <= 2:
        return "Junior"
    elif years <= 5:
        return "Mid-Level"
    elif years <= 9:
        return "Senior"
    else:
        return "Lead / Principal"


# ──────────────────────────────────────────────────────────────────────────────
# Resume Text Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    if not PYPDF_AVAILABLE:
        return ""
    try:
        import pypdf
        reader = pypdf.PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""


def extract_text_from_docx(path: str) -> str:
    if not DOCX_AVAILABLE:
        return ""
    try:
        doc = DocxDocument(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""


def extract_resume_text(file_path: str) -> str:
    """Extract text from a resume file (PDF, DOCX, or TXT)."""
    path = Path(file_path)
    if not path.exists():
        return ""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"Text read error: {e}")
            return ""


# ──────────────────────────────────────────────────────────────────────────────
# Skill Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_skills_from_text(text: str) -> List[str]:
    """Extract recognized skills from resume/profile text via keyword matching."""
    text_lower = text.lower()
    found = []
    for skill in ALL_SKILLS:
        # Use word-boundary aware matching
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill.title())
    return list(dict.fromkeys(found))  # deduplicate preserving order


def extract_years_of_experience(text: str) -> int:
    """Heuristically extract years of experience from resume text."""
    patterns = [
        r'(\d+)\+?\s+years?\s+of\s+experience',
        r'(\d+)\+?\s+years?\s+experience',
        r'experience[:\s]+(\d+)\+?\s+years?',
        r'(\d+)\+?\s+yrs?\s+of\s+experience',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return min(int(match.group(1)), 40)  # cap at 40 years
    return 0


def generate_resume_summary(text: str, max_sentences: int = 3) -> str:
    """Extract the first meaningful paragraph as a summary."""
    if not text:
        return "No resume provided."
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        summary = " ".join(paragraphs[:max_sentences])
        return summary[:500] + ("..." if len(summary) > 500 else "")
    return text[:300]


# ──────────────────────────────────────────────────────────────────────────────
# Profile Builder
# ──────────────────────────────────────────────────────────────────────────────

class ProfileBuilder:
    """
    Builds a UserProfile from user-provided inputs and optional resume file.
    """

    def build_from_inputs(
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
        """
        Construct a UserProfile, enriching it with auto-extracted data
        from the resume if provided.
        """
        # Extract resume text from file if provided
        if resume_file:
            extracted = extract_resume_text(resume_file)
            if extracted:
                resume_text = extracted
                logger.info(f"Resume extracted from file: {resume_file}")

        # Auto-extract missing fields from resume text
        auto_skills = extract_skills_from_text(resume_text) if resume_text else []
        auto_years = extract_years_of_experience(resume_text) if (years_of_experience == 0 and resume_text) else years_of_experience

        combined_skills = list(dict.fromkeys((skills or []) + auto_skills))

        resolved_level = experience_level or classify_experience_level(auto_years)

        summary = generate_resume_summary(resume_text)

        return UserProfile(
            name=name,
            job_role=job_role,
            experience_level=resolved_level,
            years_of_experience=auto_years,
            skills=combined_skills,
            education=education,
            resume_text=resume_text,
            resume_summary=summary,
            target_company=target_company,
            interview_type=interview_type,
        )

    def build_minimal(self, name: str, job_role: str, experience_level: str) -> UserProfile:
        """Build a minimal profile without a resume — uses role-based defaults."""
        role_skills = self._default_skills_for_role(job_role)
        return UserProfile(
            name=name,
            job_role=job_role,
            experience_level=experience_level,
            skills=role_skills,
        )

    def _default_skills_for_role(self, role: str) -> List[str]:
        role_lower = role.lower()
        if any(kw in role_lower for kw in ["software", "developer", "engineer", "fullstack", "backend", "frontend"]):
            return ["Python", "Java", "SQL", "REST APIs", "Git", "Agile"]
        elif any(kw in role_lower for kw in ["data scientist", "ml", "machine learning", "ai"]):
            return ["Python", "Machine Learning", "TensorFlow", "Pandas", "SQL", "Statistics"]
        elif any(kw in role_lower for kw in ["devops", "sre", "cloud"]):
            return ["Docker", "Kubernetes", "AWS", "CI/CD", "Linux", "Terraform"]
        elif any(kw in role_lower for kw in ["product manager", "pm"]):
            return ["Product Strategy", "Agile", "Roadmapping", "SQL", "Communication", "Leadership"]
        elif any(kw in role_lower for kw in ["data analyst", "analyst"]):
            return ["SQL", "Python", "Excel", "Tableau", "Statistics", "Communication"]
        else:
            return ["Communication", "Problem Solving", "Teamwork", "Adaptability"]
