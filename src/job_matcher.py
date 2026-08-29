import os
import re
import json
import time
from pathlib import Path

import chromadb
from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
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
# REQUIREMENT DEFINITIONS
# ============================================================

REQUIREMENT_ALIASES = {

    "Python": [
        "python"
    ],

    "FastAPI or Django": [
        "fastapi",
        "django"
    ],

    "REST APIs": [
        "rest api",
        "rest apis",
        "restful api",
        "restful apis"
    ],

    "SQL": [
        "sql",
        "sql server",
        "mysql",
        "postgresql",
        "oracle"
    ],

    "Git": [
        "git",
        "github"
    ],

    "AWS": [
        "aws",
        "amazon web services"
    ],

    "Docker": [
        "docker"
    ],

    "Kubernetes": [
        "kubernetes",
        "k8s"
    ],

    "Microservices": [
        "microservices",
        "microservice"
    ],

    "CI/CD": [
        "ci/cd",
        "continuous integration",
        "continuous delivery",
        "continuous deployment",
        "jenkins"
    ],

    "Software Development": [
        "software development",
        "software engineering",
        "application development",
        "backend development",
        "web development"
    ],
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):

    text = text.lower()

    text = text.replace(
        "\u200b",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TERM MATCHING
# ============================================================

def contains_any(
    text,
    terms
):

    text = normalize(text)

    for term in terms:

        term = normalize(term)

        if term in text:
            return True

    return False


# ============================================================
# EXTRACT YEARS
# ============================================================

def extract_year_requirements(
    job_description
):

    text = normalize(
        job_description
    )

    # General professional experience.
    professional_match = re.search(
        r"(\d+)\+?\s*years?\s+of\s+professional\s+software\s+development",
        text
    )

    professional_years = (
        int(professional_match.group(1))
        if professional_match
        else 0
    )

    # Python experience.
    python_match = re.search(
        r"(\d+)\+?\s*years?\s+of\s+python\s+experience",
        text
    )

    python_years = (
        int(python_match.group(1))
        if python_match
        else 0
    )

    return (
        professional_years,
        python_years
    )


# ============================================================
# JD REQUIREMENTS
# ============================================================

def extract_requirements(
    job_description
):

    text = normalize(
        job_description
    )

    requirements = []

    if contains_any(
        text,
        ["python"]
    ):
        requirements.append(
            "Python"
        )

    if (
        "fastapi" in text
        or "django" in text
    ):
        requirements.append(
            "FastAPI or Django"
        )

    if (
        "rest api" in text
        or "rest apis" in text
        or "restful api" in text
    ):
        requirements.append(
            "REST APIs"
        )

    if "sql" in text:
        requirements.append(
            "SQL"
        )

    if "git" in text:
        requirements.append(
            "Git"
        )

    if (
        "software development best practices"
        in text
        or "software development"
        in text
    ):
        requirements.append(
            "Software Development"
        )

    return requirements


# ============================================================
# EXPERIENCE EXTRACTION
# ============================================================

def extract_resume_experience(
    resume_text
):

    text = normalize(
        resume_text
    )

    # Explicit statements.
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+experience",
        r"(\d+(?:\.\d+)?)\+?\s*yrs?\s+experience",
    ]

    values = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        for match in matches:

            try:
                values.append(
                    float(match)
                )
            except ValueError:
                pass

    if values:
        return max(values)

    # Employment date ranges.
    ranges = re.findall(
        r"\b(19\d{2}|20\d{2})\s*[-–—]\s*"
        r"(19\d{2}|20\d{2}|present|current)\b",
        text
    )

    total_years = 0

    for start, end in ranges:

        try:

            start_year = int(start)

            if end in (
                "present",
                "current"
            ):
                end_year = 2026
            else:
                end_year = int(end)

            if end_year > start_year:

                total_years += (
                    end_year - start_year
                )

        except ValueError:
            pass

    return float(
        total_years
    )


# ============================================================
# REQUIREMENT MATCHING
# ============================================================

def check_requirement(
    requirement,
    resume_text
):

    aliases = REQUIREMENT_ALIASES.get(
        requirement,
        []
    )

    return contains_any(
        resume_text,
        aliases
    )


# ============================================================
# PYTHON EXPERIENCE
# ============================================================

def estimate_python_experience(
    resume_text
):

    text = normalize(
        resume_text
    )

    patterns = [
        r"(\d+)\+?\s*years?\s+of\s+python",
        r"(\d+)\+?\s*years?\s+python",
        r"python\s+.*?(\d+)\+?\s*years?",
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

    return 0.0


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    semantic_score,
    required_matches,
    total_requirements,
    experience_ok,
    python_experience_ok
):

    if total_requirements == 0:

        requirement_score = 0

    else:

        requirement_score = (
            required_matches
            / total_requirements
        )

    score = (
        requirement_score * 65
        + semantic_score * 20
        + (10 if experience_ok else 0)
        + (5 if python_experience_ok else 0)
    )

    return round(
        min(score, 100),
        1
    )


# ============================================================
# MATCH JOB
# ============================================================

def match_job(
    job_description
):

    start_time = time.perf_counter()

    requirements = extract_requirements(
        job_description
    )

    professional_years, python_years = (
        extract_year_requirements(
            job_description
        )
    )

    # --------------------------------------------------------
    # Semantic retrieval.
    # --------------------------------------------------------

    embedding_response = (
        client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=job_description
        )
    )

    query_embedding = (
        embedding_response
        .data[0]
        .embedding
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=50,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    distances = results[
        "distances"
    ][0]

    # --------------------------------------------------------
    # Combine chunks by resume.
    # --------------------------------------------------------

    candidates = {}

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        resume_path = metadata.get(
            "resume_path",
            ""
        )

        candidate_name = metadata.get(
            "candidate_name",
            Path(resume_path).stem
        )

        key = resume_path or candidate_name

        if key not in candidates:

            candidates[key] = {
                "candidate_name":
                    candidate_name,

                "resume_path":
                    resume_path,

                "chunks": [],

                "distances": [],

                "sections": set(),
            }

        candidates[key]["chunks"].append(
            document
        )

        candidates[key]["distances"].append(
            float(distance)
        )

        candidates[key]["sections"].add(
            metadata.get(
                "section",
                "general"
            )
        )

    ranked = []

    # --------------------------------------------------------
    # Candidate scoring.
    # --------------------------------------------------------

    for candidate in candidates.values():

        resume_text = "\n".join(
            candidate["chunks"]
        )

        normalized_resume = normalize(
            resume_text
        )

        # Chroma cosine distance:
        # lower distance = better match.
        best_distance = min(
            candidate["distances"]
        )

        semantic_score = max(
            0.0,
            min(
                1.0,
                1.0 - best_distance
            )
        )

        matched_requirements = []

        for requirement in requirements:

            if check_requirement(
                requirement,
                normalized_resume
            ):

                matched_requirements.append(
                    requirement
                )

        candidate_experience = (
            extract_resume_experience(
                normalized_resume
            )
        )

        python_experience = (
            estimate_python_experience(
                normalized_resume
            )
        )

        experience_ok = (
            candidate_experience
            >= professional_years
        )

        python_experience_ok = (
            python_experience
            >= python_years
        )

        # ----------------------------------------------------
        # MUST HAVE FILTER
        # ----------------------------------------------------

        # For this Python JD, Python, REST API, SQL,
        # Git and FastAPI/Django are critical.
        critical_requirements = [
            "Python",
            "FastAPI or Django",
            "REST APIs",
            "SQL",
            "Git",
        ]

        critical_missing = [
            requirement
            for requirement
            in critical_requirements
            if requirement in requirements
            and requirement
            not in matched_requirements
        ]

        must_have_passed = (
            len(critical_missing) == 0
            and experience_ok
            and python_experience_ok
        )

        score = calculate_score(
            semantic_score,
            len(matched_requirements),
            len(requirements),
            experience_ok,
            python_experience_ok
        )

        # Strong penalty for failing critical requirements.
        if critical_missing:

            score *= 0.35

        # Strong penalty for insufficient experience.
        if not experience_ok:

            score *= 0.70

        score = round(
            score,
            1
        )

        # ----------------------------------------------------
        # Excerpts
        # ----------------------------------------------------

        excerpts = []

        for chunk in candidate["chunks"]:

            chunk_normalized = normalize(
                chunk
            )

            if any(
                check_requirement(
                    requirement,
                    chunk_normalized
                )
                for requirement
                in matched_requirements
            ):

                excerpts.append(
                    chunk[:700]
                )

        if not excerpts:

            excerpts = [
                chunk[:700]
                for chunk
                in candidate["chunks"][:2]
            ]

        # ----------------------------------------------------
        # Reasoning
        # ----------------------------------------------------

        reasoning_parts = []

        if matched_requirements:

            reasoning_parts.append(
                "Matched requirements: "
                + ", ".join(
                    matched_requirements
                )
                + "."
            )

        if critical_missing:

            reasoning_parts.append(
                "Missing critical requirements: "
                + ", ".join(
                    critical_missing
                )
                + "."
            )

        if experience_ok:

            reasoning_parts.append(
                f"Estimated professional experience: "
                f"{candidate_experience:.1f} years."
            )

        else:

            reasoning_parts.append(
                f"Estimated professional experience: "
                f"{candidate_experience:.1f} years, "
                f"below the required "
                f"{professional_years}+ years."
            )

        if python_years > 0:

            if python_experience_ok:

                reasoning_parts.append(
                    f"Python experience requirement "
                    f"appears satisfied."
                )

            else:

                reasoning_parts.append(
                    f"Could not verify the required "
                    f"{python_years}+ years of Python experience."
                )

        if must_have_passed:

            reasoning_parts.append(
                "All critical must-have requirements passed."
            )

        else:

            reasoning_parts.append(
                "Candidate does not satisfy all must-have requirements."
            )

        ranked.append(
            {
                "candidate_name":
                    candidate["candidate_name"],

                "resume_path":
                    candidate["resume_path"],

                "match_score":
                    score,

                "matched_skills":
                    matched_requirements,

                "relevant_excerpts":
                    excerpts[:3],

                "reasoning":
                    " ".join(
                        reasoning_parts
                    ),

                "must_have_requirements_met":
                    must_have_passed,

                "experience_years":
                    candidate_experience,

                "python_experience_years":
                    python_experience,

                "matched_sections":
                    sorted(
                        candidate["sections"]
                    ),
            }
        )

    # --------------------------------------------------------
    # Valid candidates first.
    # --------------------------------------------------------

    ranked.sort(
        key=lambda candidate: (
            candidate[
                "must_have_requirements_met"
            ],
            candidate[
                "match_score"
            ]
        ),
        reverse=True
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return {
        "job_description":
            job_description,

        "top_matches":
            ranked[:TOP_K],

        "metadata": {
            "required_skills":
                requirements,

            "required_professional_experience":
                professional_years,

            "required_python_experience":
                python_years,

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
    print(
        "RAG PROFILE MATCHING ENGINE"
    )
    print("=" * 60)

    jd_directory = (
        BASE_DIR
        / "job_descriptions"
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

        selected_file = files[
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
        selected_file.read_text(
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
            "   Matched requirements: "
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
            "   Python experience: "
            f"{candidate['python_experience_years']} years"
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
        "Results saved to:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()