import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict

import chromadb
from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "resume_profiles"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

TOP_K = 10

load_dotenv(BASE_DIR / ".env")

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# SKILL / TECHNOLOGY DICTIONARY
# ============================================================

SKILLS = [
    "Python",
    "Java",
    "C#",
    ".NET",
    "ASP.NET",
    "ASP.NET Core",
    "SQL",
    "SQL Server",
    "MySQL",
    "PostgreSQL",
    "Oracle",
    "JavaScript",
    "TypeScript",
    "React",
    "Angular",
    "Node.js",
    "Machine Learning",
    "Deep Learning",
    "TensorFlow",
    "PyTorch",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "AWS",
    "Azure",
    "GCP",
    "Docker",
    "Kubernetes",
    "Terraform",
    "Jenkins",
    "Git",
    "GitHub",
    "Kafka",
    "Spark",
    "Databricks",
    "Airflow",
    "Redis",
    "Microservices",
    "REST API",
    "REST APIs",
    "Django",
    "FastAPI",
    "Flask",
    "Spring",
    "Spring Boot",
    "Linux",
    "ETL",
    "ELT",
    "Hadoop",
    "Power BI",
    "Tableau",
    "AutoCAD",
    "Civil 3D",
    "CAD",
    "Civil Engineering",
    "Surveying",
    "Construction",
    "Excel",
    "Leadership",
    "Marketing",
    "Customer Service",
    "Documentation",
    "Project Management",
    "Agile",
    "Scrum",
]


# ============================================================
# HELPERS
# ============================================================

def normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower()
    ).strip()


def contains_term(term: str, text: str) -> bool:

    text_normalized = normalize(text)
    term_normalized = normalize(term)

    if term_normalized == "c#":
        return bool(
            re.search(
                r"(?<![a-z])c\s*#(?![a-z])",
                text_normalized
            )
        )

    if term_normalized == ".net":
        return bool(
            re.search(
                r"(?<![a-z])\.net(?![a-z])",
                text_normalized
            )
        )

    return bool(
        re.search(
            r"(?<!\w)"
            + re.escape(term_normalized)
            + r"(?!\w)",
            text_normalized
        )
    )


def extract_known_skills(text: str) -> List[str]:

    return sorted(
        {
            skill
            for skill in SKILLS
            if contains_term(skill, text)
        }
    )


# ============================================================
# JD ANALYSIS
# ============================================================

def extract_required_experience(
    job_description: str
) -> float:

    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+professional\s+experience",
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+experience",
    ]

    values = []

    for pattern in patterns:

        for match in re.findall(
            pattern,
            job_description.lower()
        ):

            try:
                values.append(
                    float(match)
                )
            except ValueError:
                pass

    return max(values) if values else 0.0


def extract_must_have_skills(
    job_description: str
) -> List[str]:

    text = job_description.lower()

    if "must-have requirements:" not in text:
        return extract_known_skills(
            job_description
        )

    section = text.split(
        "must-have requirements:",
        1
    )[1]

    if "nice-to-have:" in section:

        section = section.split(
            "nice-to-have:",
            1
        )[0]

    elif "responsibilities:" in section:

        section = section.split(
            "responsibilities:",
            1
        )[0]

    return extract_known_skills(
        section
    )


# ============================================================
# EMBEDDING
# ============================================================

def create_embedding(
    text: str
) -> List[float]:

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


# ============================================================
# VECTOR SEARCH
# ============================================================

def search_resumes(
    job_description: str,
    top_k: int = 50
):

    embedding = create_embedding(
        job_description
    )

    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


# ============================================================
# EXPERIENCE
# ============================================================

def extract_resume_experience(
    resume_text: str
) -> float:

    text = resume_text.lower()

    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+experience",
        r"(\d+(?:\.\d+)?)\+?\s*yrs?\s+experience",
    ]

    values = []

    for pattern in patterns:

        for match in re.findall(
            pattern,
            text
        ):

            try:
                values.append(
                    float(match)
                )
            except ValueError:
                pass

    if values:
        return max(values)

    # Calculate from employment ranges.
    ranges = re.findall(
        r"\b(19\d{2}|20\d{2})\s*[-–—]\s*"
        r"(19\d{2}|20\d{2}|present|current)\b",
        text
    )

    total_months = 0

    for start, end in ranges:

        try:

            start_year = int(start)

            end_year = (
                2026
                if end in ("present", "current")
                else int(end)
            )

            if end_year > start_year:

                total_months += (
                    end_year - start_year
                ) * 12

        except ValueError:
            continue

    if total_months:

        return round(
            total_months / 12,
            1
        )

    return 0.0


# ============================================================
# SCORING
# ============================================================

def skill_score(
    resume_text: str,
    required_skills: List[str]
) -> float:

    if not required_skills:
        return 1.0

    matched = sum(
        contains_term(
            skill,
            resume_text
        )
        for skill in required_skills
    )

    return matched / len(
        required_skills
    )


def experience_score(
    candidate_years: float,
    required_years: float
) -> float:

    if required_years <= 0:
        return 1.0

    if candidate_years >= required_years:
        return 1.0

    if candidate_years <= 0:
        return 0.0

    return candidate_years / required_years


def calculate_final_score(
    semantic_score: float,
    skills_score: float,
    exp_score: float,
    must_have_passed: bool
) -> float:

    score = (
        semantic_score * 0.40
        + skills_score * 0.45
        + exp_score * 0.15
    )

    # Candidates failing must-have requirements
    # should rank substantially lower.
    if not must_have_passed:
        score *= 0.40

    return round(
        max(
            0.0,
            min(
                score * 100,
                100.0
            )
        ),
        1
    )


# ============================================================
# MUST-HAVE CHECK
# ============================================================

def passes_must_have(
    resume_text: str,
    candidate_experience: float,
    must_have_skills: List[str],
    required_experience: float
) -> bool:

    # Every must-have skill must be present.
    for skill in must_have_skills:

        if not contains_term(
            skill,
            resume_text
        ):
            return False

    # Experience requirement.
    if (
        required_experience > 0
        and candidate_experience < required_experience
    ):
        return False

    return True


# ============================================================
# REASONING
# ============================================================

def build_reasoning(
    matched_skills: List[str],
    candidate_experience: float,
    required_experience: float,
    must_have_passed: bool
) -> str:

    parts = []

    if matched_skills:

        parts.append(
            "Matched skills: "
            + ", ".join(matched_skills)
            + "."
        )

    if required_experience > 0:

        if candidate_experience >= required_experience:

            parts.append(
                f"Candidate has approximately "
                f"{candidate_experience:.1f} years of "
                f"experience, meeting the "
                f"{required_experience:.1f}+ year requirement."
            )

        else:

            parts.append(
                f"Candidate has approximately "
                f"{candidate_experience:.1f} years of "
                f"experience, below the "
                f"{required_experience:.1f}+ year requirement."
            )

    if must_have_passed:

        parts.append(
            "All must-have requirements are satisfied."
        )

    else:

        parts.append(
            "One or more must-have requirements are not satisfied."
        )

    return " ".join(parts)


# ============================================================
# MAIN MATCHING ENGINE
# ============================================================

def match_job(
    job_description: str
) -> Dict:

    start = time.perf_counter()

    required_skills = extract_known_skills(
        job_description
    )

    must_have_skills = extract_must_have_skills(
        job_description
    )

    required_experience = extract_required_experience(
        job_description
    )

    search_results = search_resumes(
        job_description,
        top_k=50
    )

    documents = search_results[
        "documents"
    ][0]

    metadatas = search_results[
        "metadatas"
    ][0]

    distances = search_results[
        "distances"
    ][0]

    candidates = {}

    # --------------------------------------------------------
    # Combine all chunks belonging to the same resume.
    # --------------------------------------------------------

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        candidate_name = metadata.get(
            "candidate_name",
            "Unknown Candidate"
        )

        if candidate_name not in candidates:

            candidates[candidate_name] = {
                "candidate_name":
                    candidate_name,

                "resume_path":
                    metadata.get(
                        "resume_path",
                        ""
                    ),

                "chunks": [],

                "sections": set(),

                "semantic_scores": [],
            }

        candidate = candidates[
            candidate_name
        ]

        candidate["chunks"].append(
            document
        )

        candidate["sections"].add(
            metadata.get(
                "section",
                "general"
            )
        )

        similarity = max(
            0.0,
            min(
                1.0,
                1.0 - float(distance)
            )
        )

        candidate[
            "semantic_scores"
        ].append(
            similarity
        )

    ranked = []

    # --------------------------------------------------------
    # Score each candidate.
    # --------------------------------------------------------

    for candidate in candidates.values():

        resume_text = "\n".join(
            candidate["chunks"]
        )

        semantic = max(
            candidate["semantic_scores"]
        )

        candidate_skills = extract_known_skills(
            resume_text
        )

        matched_skills = [
            skill
            for skill in required_skills
            if contains_term(
                skill,
                resume_text
            )
        ]

        candidate_experience = (
            extract_resume_experience(
                resume_text
            )
        )

        skills = skill_score(
            resume_text,
            required_skills
        )

        exp = experience_score(
            candidate_experience,
            required_experience
        )

        must_have = passes_must_have(
            resume_text,
            candidate_experience,
            must_have_skills,
            required_experience
        )

        final_score = calculate_final_score(
            semantic,
            skills,
            exp,
            must_have
        )

        # ----------------------------------------------------
        # Find relevant excerpts.
        # ----------------------------------------------------

        excerpts = []

        chunks = candidate["chunks"]

        for chunk in chunks:

            if any(
                contains_term(
                    skill,
                    chunk
                )
                for skill in required_skills
            ):

                excerpts.append(
                    chunk[:700]
                )

        if not excerpts:

            excerpts = [
                chunk[:700]
                for chunk in chunks[:2]
            ]

        reasoning = build_reasoning(
            matched_skills,
            candidate_experience,
            required_experience,
            must_have
        )

        ranked.append(
            {
                "candidate_name":
                    candidate["candidate_name"],

                "resume_path":
                    candidate["resume_path"],

                "match_score":
                    final_score,

                "matched_skills":
                    sorted(
                        set(
                            matched_skills
                        )
                    ),

                "relevant_excerpts":
                    excerpts[:3],

                "reasoning":
                    reasoning,

                "must_have_requirements_met":
                    must_have,

                "experience_years":
                    candidate_experience,

                "matched_sections":
                    sorted(
                        candidate["sections"]
                    ),
            }
        )

    ranked.sort(
        key=lambda x: x["match_score"],
        reverse=True
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return {
        "job_description":
            job_description,

        "top_matches":
            ranked[:TOP_K],

        "metadata": {
            "required_skills":
                required_skills,

            "must_have_skills":
                must_have_skills,

            "required_experience_years":
                required_experience,

            "retrieval_latency_seconds":
                round(
                    elapsed,
                    4
                ),
        }
    }


# ============================================================
# CLI
# ============================================================

def main():

    print("=" * 60)
    print("RAG PROFILE MATCHING ENGINE")
    print("=" * 60)

    jd_directory = (
        BASE_DIR / "job_descriptions"
    )

    files = sorted(
        jd_directory.glob("*.txt")
    )

    print()
    print(
        "Available job descriptions:"
    )

    for index, file in enumerate(
        files,
        1
    ):

        print(
            f"{index}. {file.name}"
        )

    print()

    choice = input(
        "Enter JD number (1-5): "
    ).strip()

    try:

        selected = files[
            int(choice) - 1
        ]

    except (
        ValueError,
        IndexError
    ):

        print(
            "Invalid selection."
        )

        return

    job_description = (
        selected.read_text(
            encoding="utf-8"
        )
    )

    print()
    print(
        "Searching resumes..."
    )

    result = match_job(
        job_description
    )

    output_file = (
        BASE_DIR
        / "job_match_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("TOP MATCHES")
    print("=" * 60)

    for index, candidate in enumerate(
        result["top_matches"],
        1
    ):

        print(
            f"{index}. "
            f"{candidate['candidate_name']} "
            f"- "
            f"{candidate['match_score']}/100"
        )

        print(
            "   Matched skills: "
            + (
                ", ".join(
                    candidate[
                        "matched_skills"
                    ]
                )
                or "None"
            )
        )

        print(
            "   Experience: "
            f"{candidate['experience_years']} years"
        )

        print(
            "   Must-have: "
            f"{candidate['must_have_requirements_met']}"
        )

        print(
            "   Reasoning: "
            f"{candidate['reasoning']}"
        )

        print()

    print("=" * 60)

    print(
        "Retrieval latency: "
        f"{result['metadata']['retrieval_latency_seconds']} seconds"
    )

    print(
        "Results saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()