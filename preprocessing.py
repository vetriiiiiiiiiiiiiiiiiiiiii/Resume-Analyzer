import re
from typing import Dict, List


DEFAULT_SKILLS = [
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "flask",
    "fastapi",
    "django",
    "streamlit",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",
    "matplotlib",
    "seaborn",
    "plotly",
    "tableau",
    "power bi",
    "excel",
    "git",
    "github",
    "linux",
    "terraform",
    "ansible",
    "ci/cd",
    "react",
    "angular",
    "vue",
    "node",
    "javascript",
    "typescript",
    "java",
    "c++",
    "c#",
    "api",
    "rest api",
    "microservices",
    "data analysis",
    "statistics",
    "cybersecurity",
    "spark",
    "hadoop",
    "airflow",
    "mlops",
    "mongodb",
    "postgresql",
    "mysql",
    "snowflake",
    "powerpoint",
    "communication",
    "leadership",
    "agile",
    "scrum",
    "jira",
    "product strategy",
    "user research",
]

SKILL_ALIASES = {
    "scikit-learn": ["sklearn", "scikit learn"],
    "natural language processing": ["nlp"],
    "ci/cd": ["ci cd", "continuous integration", "continuous delivery"],
    "power bi": ["powerbi"],
    "rest api": ["restful api", "rest apis"],
    "machine learning": ["ml"],
    "javascript": ["js"],
    "typescript": ["ts"],
    "postgresql": ["postgres"],
}

EXPECTED_SECTIONS = {
    "summary": ["summary", "profile", "objective"],
    "skills": ["skills", "technical skills", "core competencies"],
    "experience": ["experience", "work experience", "employment", "professional experience"],
    "projects": ["projects", "project experience"],
    "education": ["education", "academic"],
    "certifications": ["certifications", "certificates", "licenses"],
}


def clean_text(text: str) -> str:
    """Normalize text for TF-IDF and lightweight keyword matching."""
    if not text:
        return ""
    text = text.lower()
    text = text.replace("ci cd", "ci/cd")
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text: str, skills: List[str] | None = None) -> List[str]:
    normalized = clean_text(text)
    skill_list = skills or DEFAULT_SKILLS
    found = []

    for skill in skill_list:
        aliases = [skill.lower(), *SKILL_ALIASES.get(skill.lower(), [])]
        if any(
            re.search(r"(?<![a-z0-9+#])" + re.escape(alias) + r"(?![a-z0-9+#])", normalized)
            for alias in aliases
        ):
            found.append(skill)

    return sorted(set(found))


def compare_skills(resume_text: str, job_description: str) -> Dict[str, List[str]]:
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    matched = sorted(set(resume_skills).intersection(job_skills))
    missing = sorted(set(job_skills).difference(resume_skills))

    return {
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matched_skills": matched,
        "missing_skills": missing,
    }


def detect_sections(text: str) -> Dict[str, bool]:
    normalized = clean_text(text)
    detected = {}
    for section, aliases in EXPECTED_SECTIONS.items():
        detected[section] = any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases)
    return detected


def keyword_coverage(resume_text: str, job_description: str) -> float:
    comparison = compare_skills(resume_text, job_description)
    job_skills = comparison["job_skills"]
    if not job_skills:
        return 0.0
    return len(comparison["matched_skills"]) / len(job_skills)


def estimate_ats_score(
    similarity_score: float,
    skills: List[str],
    resume_text: str,
    job_description: str = "",
) -> int:
    sections = detect_sections(resume_text)
    section_score = sum(sections.values()) / len(sections)
    word_count = len(resume_text.split())
    length_score = 1.0 if 350 <= word_count <= 900 else 0.65 if word_count >= 180 else 0.35
    coverage = keyword_coverage(resume_text, job_description) if job_description else min(len(skills) / 12, 1)

    score = 0
    score += similarity_score * 35
    score += coverage * 30
    score += section_score * 20
    score += length_score * 10
    score += min(len(skills) / 15, 1) * 5
    return int(max(0, min(round(score), 100)))


def evidence_summary(resume_text: str, job_description: str) -> Dict[str, object]:
    comparison = compare_skills(resume_text, job_description)
    sections = detect_sections(resume_text)
    return {
        **comparison,
        "sections": sections,
        "word_count": len(resume_text.split()),
        "keyword_coverage": round(keyword_coverage(resume_text, job_description) * 100, 2)
        if job_description
        else 0.0,
    }
