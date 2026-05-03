import csv
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from sklearn.exceptions import InconsistentVersionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

from preprocessing import clean_text, estimate_ats_score, evidence_summary, extract_skills


MODEL_PATH = Path("models/resume_role_model.pkl")
DATASET_PATH = Path("dataset/job_roles.csv")


@dataclass
class ResumeAnalysis:
    predicted_role: str
    confidence: float
    skills: List[str]
    matched_skills: List[str]
    missing_skills: List[str]
    sections: Dict[str, bool]
    keyword_coverage: float
    word_count: int
    score: float
    ats_score: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "predicted_role": self.predicted_role,
            "confidence": self.confidence,
            "skills": self.skills,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "sections": self.sections,
            "keyword_coverage": self.keyword_coverage,
            "word_count": self.word_count,
            "score": self.score,
            "ats_score": self.ats_score,
        }


def load_training_data(dataset_path: Path = DATASET_PATH) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            texts.append(clean_text(row["text"]))
            labels.append(row["role"])
    return texts, labels


def train_and_save_model(model_path: Path = MODEL_PATH) -> Pipeline:
    texts, labels = load_training_data()
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(texts, labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(pipeline, file)
    return pipeline


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    if not model_path.exists():
        return train_and_save_model(model_path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            with model_path.open("rb") as file:
                return pickle.load(file)
    except (AttributeError, InconsistentVersionWarning, ValueError, pickle.UnpicklingError):
        return train_and_save_model(model_path)


def calculate_similarity(resume_text: str, job_description: str) -> float:
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)
    if not resume_clean or not jd_clean:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([jd_clean, resume_clean])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def analyze_resume(resume_text: str, job_description: str = "") -> ResumeAnalysis:
    model = load_model()
    cleaned_resume = clean_text(resume_text)
    predicted_role = str(model.predict([cleaned_resume])[0])
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba([cleaned_resume])[0]))

    comparison_text = job_description or predicted_role
    similarity = calculate_similarity(resume_text, comparison_text)
    skills = extract_skills(resume_text)
    evidence = evidence_summary(resume_text, job_description)
    ats_score = estimate_ats_score(similarity, skills, resume_text, job_description)

    return ResumeAnalysis(
        predicted_role=predicted_role,
        confidence=round(confidence * 100, 2),
        skills=skills,
        matched_skills=evidence["matched_skills"],
        missing_skills=evidence["missing_skills"],
        sections=evidence["sections"],
        keyword_coverage=evidence["keyword_coverage"],
        word_count=evidence["word_count"],
        score=round(similarity * 100, 2),
        ats_score=ats_score,
    )


if __name__ == "__main__":
    path = os.fspath(MODEL_PATH)
    train_and_save_model()
    print(f"Saved model to {path}")
