import re
from typing import List


DEFAULT_SKILLS = [
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "flask",
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
    "tableau",
    "power bi",
    "excel",
    "git",
    "linux",
    "terraform",
    "ci/cd",
    "react",
    "javascript",
    "java",
    "api",
    "data analysis",
    "statistics",
    "cybersecurity",
]


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
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill.lower()) + r"(?![a-z0-9+#])"
        if re.search(pattern, normalized):
            found.append(skill)

    return sorted(set(found))


def estimate_ats_score(similarity_score: float, skills: List[str], resume_text: str) -> int:
    score = similarity_score * 70
    score += min(len(skills), 12) * 2
    if len(resume_text.split()) > 250:
        score += 6
    if re.search(r"\b(experience|projects|education|skills|certifications)\b", resume_text.lower()):
        score += 6
    return int(max(0, min(round(score), 100)))
