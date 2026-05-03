import os
from typing import Dict, List

import requests


SYSTEM_INSTRUCTIONS = """You are an expert resume reviewer and career coach.
Write in a professional tone.
Use concise bullet points.
Do not invent experience that is not present in the resume.
Structure the answer with: Resume Feedback, Skill Gap Analysis, ATS Score Estimate, Career Suggestions, Resume Improvements."""


def build_prompt(resume_text: str, job_description: str, analysis: Dict[str, object]) -> str:
    resume_excerpt = resume_text[:5000]
    jd_excerpt = (job_description or "No job description provided.")[:3000]
    return f"""
{SYSTEM_INSTRUCTIONS}

ML analysis:
- Predicted role: {analysis.get("predicted_role")}
- Extracted skills: {", ".join(analysis.get("skills", []))}
- Similarity score: {analysis.get("score")}
- ATS score estimate: {analysis.get("ats_score")}

Resume:
{resume_excerpt}

Job description:
{jd_excerpt}

Return structured bullet points only.
"""


def call_ollama(prompt: str, model: str = "llama3") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=45,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=45,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def fallback_feedback(analysis: Dict[str, object]) -> str:
    skills: List[str] = list(analysis.get("skills", []))
    skill_text = ", ".join(skills[:8]) if skills else "No explicit technical skills detected"
    role = analysis.get("predicted_role", "the target role")
    score = analysis.get("score", 0)
    ats_score = analysis.get("ats_score", 0)

    return f"""Resume Feedback
- The resume appears most aligned with {role}.
- Detected skills: {skill_text}.
- Resume-to-job match score is {score}%.

Skill Gap Analysis
- Add missing role-specific tools, measurable project outcomes, and domain keywords from the job description.
- Strengthen evidence for the highest-priority skills with project or work examples.

ATS Score Estimate
- Estimated ATS score: {ats_score}/100.
- Improve section headings, keyword coverage, and quantified impact statements.

Career Suggestions
- Target {role} roles and adjacent positions that match the detected skills.
- Build one portfolio project that mirrors the target job description.

Resume Improvements
- Use clear sections: Summary, Skills, Experience, Projects, Education, Certifications.
- Start bullets with action verbs and include numbers where possible.
- Keep formatting simple for applicant tracking systems."""


def generate_feedback(resume_text: str, job_description: str, analysis: Dict[str, object]) -> Dict[str, str]:
    prompt = build_prompt(resume_text, job_description, analysis)

    try:
        return {"provider": "ollama", "feedback": call_ollama(prompt)}
    except Exception as ollama_error:
        try:
            return {"provider": "openai", "feedback": call_openai(prompt)}
        except Exception as openai_error:
            return {
                "provider": "local_fallback",
                "feedback": fallback_feedback(analysis),
                "warning": f"LLM unavailable. Ollama error: {ollama_error}. OpenAI error: {openai_error}",
            }
