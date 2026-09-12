from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from resume_rag import search_resumes as rag_search_resumes


class AgentState(TypedDict):
    conversation_history: list
    job_description: str
    requirements: dict
    candidates: list
    reasoning: dict
    final_report: str
    feedback: str


def parse_jd(state: AgentState):
    print("1. Parse JD")

    return {
        "job_description": state["job_description"],
        "conversation_history": state["conversation_history"]
        + ["JD parsed"]
    }


def extract_requirements(state: AgentState):
    print("2. Extract Requirements")

    jd = state["job_description"].lower()

    known_skills = [
        ".net", "c#", "asp.net", "asp.net core", "mvc",
        "python", "java", "javascript", "typescript",
        "react", "angular", "sql", "sql server",
        "mongodb", "azure", "aws", "docker", "kubernetes",
        "microservices", "rest api", "git", "ci/cd",
        "fastapi", "django", "html", "css"
    ]

    detected_skills = []

    for skill in known_skills:
        if skill in jd:
            detected_skills.append(skill)

    # Detect experience requirement
    import re

    experience = None
    match = re.search(r"(\d+)\s*\+?\s*years?", jd)

    if match:
        experience = int(match.group(1))

    # For now, explicitly detected skills are treated as must-have.
    requirements = {
        "must_have": detected_skills,
        "nice_to_have": [],
        "experience": experience
    }

    print(f"   Must-have skills: {detected_skills}")
    print(f"   Experience: {experience}")

    return {
        "requirements": requirements,
        "conversation_history": (
            state["conversation_history"]
            + ["Requirements extracted"]
        )
    }


def search_resumes(state: AgentState):
    print("3. Search Resumes")

    jd = state["job_description"]

    # Use the existing Milestone 2 RAG
    rag_results = rag_search_resumes(
        jd,
        top_k=50
    )

    documents = rag_results.get("documents", [[]])[0]
    metadatas = rag_results.get("metadatas", [[]])[0]
    distances = rag_results.get("distances", [[]])[0]

    candidates = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        candidates.append({
            "candidate_name": metadata.get(
                "candidate_name",
                "Unknown"
            ),
            "resume_path": metadata.get(
                "resume_path",
                ""
            ),
            "chunk": document,
            "distance": distance
        })

    print(
        f"   RAG retrieved {len(candidates)} resume chunks"
    )

    return {
        "candidates": candidates,
        "conversation_history": (
            state["conversation_history"]
            + [
                f"RAG retrieved {len(candidates)} chunks"
            ]
        )
    }


def rank_candidates(state: AgentState):
    print("4. Rank Candidates")

    candidates = state["candidates"]
    requirements = state["requirements"]

    must_have = requirements.get("must_have", [])
    nice_to_have = requirements.get("nice_to_have", [])
    required_experience = requirements.get("experience")

    # Group chunks by candidate
    grouped = {}

    for candidate in candidates:
        name = candidate["candidate_name"]

        if name not in grouped:
            grouped[name] = {
                "candidate_name": name,
                "resume_path": candidate["resume_path"],
                "chunks": [],
                "best_distance": candidate["distance"]
            }

        grouped[name]["chunks"].append(candidate["chunk"])

        if candidate["distance"] < grouped[name]["best_distance"]:
            grouped[name]["best_distance"] = candidate["distance"]

    ranked = []

    for name, candidate in grouped.items():

        resume_text = " ".join(candidate["chunks"]).lower()

        # Skill matching
        matched_must = [
            skill for skill in must_have
            if skill.lower() in resume_text
        ]

        matched_nice = [
            skill for skill in nice_to_have
            if skill.lower() in resume_text
        ]

        missing_must = [
            skill for skill in must_have
            if skill.lower() not in resume_text
        ]

        # Scores
        must_score = (
            len(matched_must) / len(must_have) * 45
            if must_have else 45
        )

        nice_score = (
            len(matched_nice) / len(nice_to_have) * 10
            if nice_to_have else 0
        )

        # Semantic score
        semantic_score = max(
            0,
            min(30, (1 - candidate["best_distance"]) * 50)
        )

        # Experience score
        experience_score = 0

        if required_experience:
            import re

            exp_matches = re.findall(
                r"(\d+)\+?\s*years?",
                resume_text
            )

            resume_experience = max(
                [int(x) for x in exp_matches],
                default=0
            )

            if resume_experience >= required_experience:
                experience_score = 15
            elif resume_experience > 0:
                experience_score = (
                    15 * resume_experience / required_experience
                )
            else:
                experience_score = 0


        final_score = round(
            must_score
            + nice_score
            + semantic_score
            + experience_score,
            2
        )

        reasoning = {
            "matched_must_have": matched_must,
            "missing_must_have": missing_must,
            "matched_nice_to_have": matched_nice,
            "semantic_score": round(semantic_score, 2),
            "skill_score": round(must_score + nice_score, 2),
            "experience_score": round(experience_score, 2),
            "final_score": final_score,
            "strengths": matched_must + matched_nice,
            "gaps": missing_must
        }

        ranked.append({
            "candidate_name": name,
            "resume_path": candidate["resume_path"],
            "score": final_score,
            "reasoning": reasoning,
            "evidence": candidate["chunks"][:3]
        })

    # Highest score first
    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Top 10
    ranked = ranked[:10]

    reasoning = {
        candidate["candidate_name"]: candidate["reasoning"]
        for candidate in ranked
    }

    print(f"   Ranked {len(ranked)} candidates")

    for i, candidate in enumerate(ranked, 1):
        print(
            f"   {i}. {candidate['candidate_name']} "
            f"-> {candidate['score']}/100"
        )

    return {
        "candidates": ranked,
        "reasoning": reasoning,
        "conversation_history": (
            state["conversation_history"]
            + ["Candidates ranked"]
        )
    }


def generate_report(state: AgentState):
    print("5. Generate Report")

    candidates = state["candidates"]
    requirements = state["requirements"]

    lines = []

    lines.append("=" * 70)
    lines.append("RESUME MATCHING REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Must-have skills: {', '.join(requirements.get('must_have', [])) or 'None'}")
    lines.append(
        f"Experience required: "
        f"{requirements.get('experience') or 'Not specified'}"
    )
    lines.append("")
    lines.append("TOP CANDIDATES")
    lines.append("-" * 70)

    for index, candidate in enumerate(candidates, 1):
        reasoning = candidate["reasoning"]

        lines.append(
            f"{index}. {candidate['candidate_name']} "
            f"({candidate['score']}/100)"
        )

        lines.append(
            f"   Resume: {candidate['resume_path']}"
        )

        lines.append(
            f"   Strengths: "
            f"{', '.join(reasoning['strengths']) or 'None identified'}"
        )

        lines.append(
            f"   Gaps: "
            f"{', '.join(reasoning['gaps']) or 'None identified'}"
        )

        lines.append(
            f"   Semantic score: "
            f"{reasoning['semantic_score']}/30"
        )

        lines.append(
            f"   Skill score: "
            f"{reasoning['skill_score']}/55"
        )

        lines.append(
            f"   Experience score: "
            f"{reasoning['experience_score']}/15"
        )

        lines.append("")

    report = "\n".join(lines)

    print(report)

    return {
        "final_report": report,
        "conversation_history": (
            state["conversation_history"]
            + ["Report generated"]
        )
    }


# Create LangGraph
graph = StateGraph(AgentState)

graph.add_node("parse_jd", parse_jd)
graph.add_node("extract_requirements", extract_requirements)
graph.add_node("search_resumes", search_resumes)
graph.add_node("rank_candidates", rank_candidates)
graph.add_node("generate_report", generate_report)

# Workflow
graph.add_edge(START, "parse_jd")
graph.add_edge("parse_jd", "extract_requirements")
graph.add_edge("extract_requirements", "search_resumes")
graph.add_edge("search_resumes", "rank_candidates")
graph.add_edge("rank_candidates", "generate_report")
graph.add_edge("generate_report", END)

agent = graph.compile()

# ============================================================
# AGENT TOOLS
# ============================================================

def extract_requirements(jd: str):
    """Extract must-have skills, nice-to-have skills and experience."""

    jd_lower = jd.lower()

    known_skills = [
        ".net", "c#", "asp.net", "asp.net core", "mvc",
        "python", "java", "javascript", "typescript",
        "react", "angular", "sql", "sql server",
        "mongodb", "azure", "aws", "docker", "kubernetes",
        "microservices", "rest api", "git", "ci/cd",
        "fastapi", "django", "html", "css"
    ]

    must_have = [
        skill for skill in known_skills
        if skill in jd_lower
    ]

    nice_to_have = []

    import re

    match = re.search(r"(\d+)\s*\+?\s*years?", jd_lower)
    experience = int(match.group(1)) if match else None

    return {
        "must_have": must_have,
        "nice_to_have": nice_to_have,
        "experience": experience
    }


def compare_candidates(candidate_ids: list, candidates: list = None):
    """Compare candidates side by side."""

    if candidates is None:
        return {"error": "Candidate data is required"}

    selected = []

    for candidate in candidates:
        name = candidate.get("candidate_name", "")

        if (
            name in candidate_ids
            or candidate.get("resume_path") in candidate_ids
        ):
            selected.append(candidate)

    if not selected:
        return {"error": "No matching candidates found"}

    comparison = []

    for candidate in selected:
        reasoning = candidate.get("reasoning", {})

        comparison.append({
            "candidate": candidate.get("candidate_name"),
            "score": candidate.get("score"),
            "strengths": reasoning.get("strengths", []),
            "gaps": reasoning.get("gaps", []),
            "matched_skills": reasoning.get("matched_must_have", []),
            "semantic_score": reasoning.get("semantic_score", 0),
            "experience_score": reasoning.get("experience_score", 0)
        })

    comparison.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return comparison


def generate_interview_questions(candidate_id: str, candidates=None):
    candidate = None

    if candidates:
        for c in candidates:
            if c["candidate_name"].lower() == candidate_id.lower():
                candidate = c
                break

    if not candidate:
        return [
            "What projects or responsibilities demonstrate your experience?",
            "Can you explain the technologies listed in your resume?",
            "Describe a challenging technical problem you solved.",
            "How do you approach debugging and testing?",
            "What would you improve in your previous implementation?"
        ]

    reasoning = candidate.get("reasoning", {})
    skills = reasoning.get("matched_must_have", [])

    questions = []

    for skill in skills:
        questions.append(
            f"Can you explain your hands-on experience with {skill} "
            "and describe a project where you used it?"
        )

    questions.extend([
        "Walk me through your most relevant project for this role.",
        "How did you troubleshoot a difficult technical issue?",
        "How do you ensure code quality and maintainability?",
        "Describe a situation where you had to learn a new technology quickly.",
        "What would you improve in your previous project?"
    ])

    return questions[:5]
    
    """Generate screening questions for a candidate."""

    if candidates is None:
        return {"error": "Candidate data is required"}

    candidate = next(
        (
            c for c in candidates
            if c.get("candidate_name") == candidate_id
            or c.get("resume_path") == candidate_id
        ),
        None
    )

    if not candidate:
        return {"error": f"Candidate '{candidate_id}' not found"}

    reasoning = candidate.get("reasoning", {})

    strengths = reasoning.get("strengths", [])
    gaps = reasoning.get("gaps", [])

    questions = []

    for skill in strengths[:3]:
        questions.append(
            f"Can you describe a project where you used {skill}?"
        )

    for skill in gaps[:3]:
        questions.append(
            f"What is your experience with {skill}, and how would you "
            f"apply it in this role?"
        )

    questions.extend([
        "Describe your most challenging technical project.",
        "How do you troubleshoot a production issue?",
        "How do you approach designing a scalable application?"
    ])

    return {
        "candidate": candidate.get("candidate_name"),
        "questions": questions[:8]
    }


# ============================================================
# HUMAN FEEDBACK
# ============================================================

def apply_human_feedback(state: AgentState):
    """Apply recruiter feedback and re-rank candidates."""

    feedback = state.get("feedback", "").lower()

    if not feedback:
        return {
            "conversation_history": (
                state["conversation_history"] +
                ["No human feedback provided"]
            )
        }

    requirements = state["requirements"].copy()

    # Add skills mentioned in feedback
    known_skills = [
        ".net", "c#", "asp.net", "mvc",
        "python", "java", "javascript", "typescript",
        "react", "angular", "sql", "azure", "aws",
        "docker", "kubernetes", "microservices",
        "rest api", "git", "fastapi", "django"
    ]

    for skill in known_skills:
        if skill in feedback and skill not in requirements["must_have"]:
            requirements["must_have"].append(skill)

    state["requirements"] = requirements

    return {
        "requirements": requirements,
        "conversation_history": (
            state["conversation_history"] +
            ["Human feedback applied"]
        )
    }


# ============================================================
# MULTI-ROUND SCREENING
# ============================================================

def deep_analysis(state: AgentState):
    """Second-round analysis of shortlisted candidates."""

    candidates = state.get("candidates", [])

    analyzed = []

    for candidate in candidates[:10]:
        reasoning = candidate.get("reasoning", {})

        strengths = reasoning.get("strengths", [])
        gaps = reasoning.get("gaps", [])
        score = candidate.get("score", 0)

        if score >= 70 and not gaps:
            recommendation = "Strong Hire"
        elif score >= 50:
            recommendation = "Consider"
        else:
            recommendation = "No Hire"

        candidate["deep_analysis"] = {
            "strength_count": len(strengths),
            "gap_count": len(gaps),
            "recommendation": recommendation
        }

        analyzed.append(candidate)

    return {
        "candidates": analyzed,
        "conversation_history": (
            state["conversation_history"] +
            ["Second-round deep analysis completed"]
        )
    }


def generate_final_decision(state: AgentState):
    """Generate final hire/no-hire recommendation."""

    candidates = state.get("candidates", [])

    if not candidates:
        decision = "NO HIRE - No suitable candidates found."
    else:
        top = candidates[0]
        score = top.get("score", 0)
        gaps = top.get("reasoning", {}).get("gaps", [])

        if score >= 70 and len(gaps) == 0:
            decision = (
                f"HIRE RECOMMENDATION: {top['candidate_name']} "
                f"({score}/100)"
            )
        elif score >= 50:
            decision = (
                f"REVIEW RECOMMENDATION: {top['candidate_name']} "
                f"({score}/100). Additional screening recommended."
            )
        else:
            decision = (
                f"NO HIRE: No candidate currently meets the required "
                f"threshold. Top candidate: {top['candidate_name']} "
                f"({score}/100)."
            )

    return {
        "final_report": state.get("final_report", "") +
        "\n\nFINAL DECISION\n" +
        "=" * 70 +
        "\n" +
        decision,
        "conversation_history": (
            state["conversation_history"] +
            ["Final hire/no-hire decision generated"]
        )
    }


# ============================================================
# CONVERSATIONAL CLI
# ============================================================

def run_conversation():
    print("=" * 70)
    print("RAG PROFILE MATCHING AGENT")
    print("=" * 70)
    print("Type a job requirement or question.")
    print("Examples:")
    print("  Find .NET developers with Azure experience")
    print("  Find React developers with 3+ years experience")
    print("  compare top 3")
    print("  interview questions for <candidate>")
    print("  exit")
    print("=" * 70)

    current_result = None

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        # EXIT
        if user_input.lower() == "exit":
            print("Agent: Goodbye!")
            break

        # COMPARE
        if user_input.lower().startswith("compare"):
            if not current_result or not current_result.get("candidates"):
                print("Agent: Run a job search first.")
                continue

            candidates = current_result["candidates"][:3]

            comparison = compare_candidates(
                [c["candidate_name"] for c in candidates],
                candidates
            )

            print("\nAgent:")
            print("=" * 70)
            print("TOP 3 CANDIDATE COMPARISON")
            print("=" * 70)

            for candidate in candidates:
                reasoning = candidate["reasoning"]

                print(
                    f"{candidate['candidate_name']}: "
                    f"{candidate['score']}/100"
                )

                print(
                    "   Matched skills: "
                    f"{', '.join(reasoning['matched_must_have']) or 'None'}"
                )

                print(
                    "   Gaps: "
                    f"{', '.join(reasoning['missing_must_have']) or 'None'}"
                )

            print("\nHead-to-head analysis:")
            print(comparison)

            continue

        # INTERVIEW QUESTIONS
        if user_input.lower().startswith("interview questions"):
            if not current_result or not current_result.get("candidates"):
                print("Agent: Run a job search first.")
                continue

            prefix = "interview questions for"
            candidate_name = user_input[len(prefix):].strip()

            if not candidate_name:
                candidate_name = current_result["candidates"][0]["candidate_name"]

            candidate = None

            for c in current_result["candidates"]:
                if c["candidate_name"].lower() == candidate_name.lower():
                    candidate = c
                    break

            if not candidate:
                print(
                    f"Agent: Candidate '{candidate_name}' "
                    "was not found."
                )
                print("Available candidates:")

                for c in current_result["candidates"][:10]:
                    print(f"  - {c['candidate_name']}")

                continue

            questions = generate_interview_questions(
                candidate["candidate_name"],
                current_result["candidates"]
            )

            print("\nAgent:")
            print("=" * 70)
            print(
                f"INTERVIEW QUESTIONS - "
                f"{candidate['candidate_name']}"
            )
            print("=" * 70)

            for i, question in enumerate(questions, 1):
                print(f"{i}. {question}")

            continue

        # NORMAL JOB SEARCH
        initial_state = {
            "conversation_history": [user_input],
            "job_description": user_input,
            "requirements": {},
            "candidates": [],
            "reasoning": {},
            "final_report": "",
            "feedback": ""
        }

        current_result = agent.invoke(initial_state)

        # Multi-round analysis
        current_result.update(deep_analysis(current_result))
        current_result.update(generate_final_decision(current_result))

        print("\nAgent:")
        print(current_result["final_report"])
        
if __name__ == "__main__":
    run_conversation()