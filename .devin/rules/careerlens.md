---
trigger: always_on
---

---
trigger: always
---

# CareerLens AI — Workspace Rules

## Project Overview
CareerLens AI is a full-stack AI-powered career intelligence platform
for freshers and job seekers. 13 modules, built step by step.

## Tech Stack
BACKEND: Python 3.11, FastAPI (async), Supabase (PostgreSQL + Auth +
Storage), PyMuPDF, sentence-transformers (BAAI/bge-m3), faiss-cpu,
FlagEmbedding (BAAI/bge-reranker-base), google-generativeai (Gemini
2.0 Flash), WeasyPrint + Jinja2, APScheduler, spaCy en_core_web_sm,
Resend

FRONTEND: React 18 + Vite, Tailwind CSS, Axios, Recharts, React
Router v6

## Folder Structure
careerlens/
├── backend/
│   ├── main.py
│   ├── routers/
│   │   ├── cv.py
│   │   ├── jobs.py
│   │   ├── match.py
│   │   ├── optimizer.py
│   │   ├── builder.py
│   │   ├── ats.py
│   │   ├── roadmap.py
│   │   ├── interview.py
│   │   ├── copilot.py
│   │   ├── tracker.py
│   │   ├── apply.py
│   │   └── profile.py
│   ├── services/
│   │   ├── cv_parser.py
│   │   ├── embedder.py
│   │   ├── matcher.py
│   │   ├── reranker.py
│   │   ├── gemini.py
│   │   ├── pdf_generator.py
│   │   ├── ats_scorer.py
│   │   ├── scraper_manager.py
│   │   ├── scheduler.py
│   │   └── scrapers/
│   │       ├── base.py
│   │       ├── adzuna.py
│   │       ├── jsearch.py
│   │       ├── remoteok.py
│   │       └── arbeitnow.py
│   ├── db/
│   │   └── supabase_client.py
│   ├── models/
│   │   └── schemas.py
│   └── templates/
│       └── resume_template.html
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Dashboard.jsx
│       │   ├── FindJobs.jsx
│       │   ├── ResumeCenter.jsx
│       │   ├── AtsChecker.jsx
│       │   ├── CareerRoadmap.jsx
│       │   ├── InterviewCoach.jsx
│       │   ├── CoPilot.jsx
│       │   ├── JobTracker.jsx
│       │   ├── EasyApply.jsx
│       │   └── Profile.jsx
│       ├── components/
│       │   ├── Sidebar.jsx
│       │   ├── MetricCard.jsx
│       │   ├── JobCard.jsx
│       │   ├── MatchBadge.jsx
│       │   ├── AlertBar.jsx
│       │   └── SkillBadge.jsx
│       └── api/
│           └── client.js
├── logs/
├── .env
└── requirements.txt

## Database Tables (Supabase PostgreSQL)
cvs: id, user_id, raw_text, structured_json, file_url,
     version, created_at
jobs: id, title, company, location, description, source,
      experience_level, redirect_url, date_posted,
      date_fetched, embedding
sessions: id, user_id, job_id, match_score, ats_score,
          skill_gap, rewritten_cv_url, created_at
interview_sessions: id, user_id, job_id, questions_answers,
                    dimension_scores, overall_score,
                    difficulty, created_at
tracker: id, user_id, job_id, status, notes, updated_at
roadmaps: id, user_id, target_role, milestones,
          completion, created_at
profiles: id, user_id, full_name, email, phone, location,
          linkedin_url, github_url, portfolio_url,
          target_title, years_experience, expected_salary,
          work_preference, education, experience, skills,
          certifications, cover_letter_templates

## Environment Variables
GEMINI_API_KEY, ADZUNA_APP_ID, ADZUNA_APP_KEY,
JSEARCH_API_KEY, SUPABASE_URL, SUPABASE_KEY, RESEND_API_KEY

## Architecture Rules — Never Violate These
- BGE-M3 and BGE-Reranker load ONCE at startup, never per request
- FAISS index lives in memory, rebuilt from Supabase on startup
- Gemini receives top 20 jobs only, never the full list
- Gemini always outputs JSON, never markdown or plain text
- WeasyPrint always receives structured JSON, never raw LLM text
- Gemini never fabricates skills not present in the original CV
- RAG is used ONLY in copilot.py, nowhere else in the project
- Every scraper inherits from BaseJobScraper in scrapers/base.py
- redirect_url is mandatory in every scraper output schema
- Job embeddings generated at ingestion time by scheduler
- Resume embedding generated at upload time only

## ATS Score Formula
keyword_score    = matched JD noun phrases / total * 40
section_score    = present sections / 6 * 20
parsability      = no tables/images/columns detected * 20
chronology       = consistent date format * 10
quantification   = bullets with numbers / total bullets * 10
TOTAL MAX = 100. Use spaCy for noun phrase extraction.
Call it "CareerLens Resume Readiness Score" not "ATS Score"

## Module Build Order
Build in this exact sequence, test each before moving on:
1. Project setup (folder structure, main.py, schemas, DB client)
2. CV Parser (PyMuPDF + Gemini validation)
3. Job Collection Engine (4 scrapers + scheduler)
4. Semantic Matching Engine (BGE-M3 + FAISS + Reranker)
5. Resume Optimizer (Gemini JSON → WeasyPrint PDF)
6. Resume Builder (form → Gemini → WeasyPrint PDF)
7. ATS Checker (standalone, no login required)
8. Career Roadmap (skill gaps → weekly plan)
9. Interview Coach (questions + evaluation + difficulty)
10. Career Co-pilot (RAG over user career data)
11. Job Tracker + Easy Apply + Profile
12. Dashboard (reads from all modules, no hardcoded values)
13. Email Digest + Deployment

## Sprint Execution Strategy

Some modules are intentionally divided into multiple
implementation sprints.

Each sprint must be completed, tested, reviewed, and
committed before beginning the next sprint.

Never implement multiple sprints together unless
explicitly instructed.

Current sprint breakdown:

Module 1 — CV Intelligence Engine

Sprint 1.1
CV Parser Foundation

Sprint 1.2.1
Canonical Candidate Profile

Sprint 1.2.2
Document Analyzer

Sprint 1.2.3
Gemini Structured Extraction

Sprint 1.2.4
Schema Validation

Sprint 1.2.5
Normalization Engine

Sprint 1.3
Semantic Embeddings (BGE-M3)

Sprint 1.4
Skill Ontology

Sprint 1.2.6
Pipeline Integration

Continue to later modules only after Module 1
has been fully completed.

## Sprint Rules

Each sprint must satisfy ALL of the following before
the next sprint begins:

✓ Code implemented
✓ Unit tests passing
✓ Logging added
✓ Error handling implemented
✓ Manual review completed
✓ Git committed
✓ Git pushed
✓ Sprint approved

Do not begin the next sprint until the previous sprint
is fully approved.

## Gemini Prompt Rules
- Always end prompts with: "Return JSON only. No markdown.
  No explanation. No code blocks."
- Always validate Gemini response is valid JSON before using it
- Always have a fallback if Gemini call fails
- Never send more than 20 job descriptions in one Gemini call

## Logging Standard
import logging
logger = logging.getLogger(__name__)
# Every service file must have this at the top
# Log: startup events, scraper results, errors, API calls
# Log files go in logs/ folder, one per service
THIS IS A REAL PRODUCT NOT A DEMO.

Standards this implies:

ACCURACY: Never show a match score that is not 
calibrated to reality. Better to show 55% honestly 
than 90% incorrectly. Real freshers are making 
job application decisions based on these scores.

SAFETY: Never output a modified CV without showing 
an explicit diff of every change. The user must 
consciously approve every modification before download.
Never fabricate anything. If unsure, omit.

INDIA-SPECIFIC: Job sources must cover Indian market 
properly. Naukri RSS, Freshersworld, Internshala, 
Hirist are mandatory. CGPA fields are mandatory in 
education. Salary shown in LPA not USD. Location 
options include Indian cities by default.

PRIVACY: Supabase RLS must be enabled on every table.
Users can only ever read and write their own data.
CV content never sent to any third party except 
Gemini for processing. Gemini calls use the minimum 
data needed for each task.

RELIABILITY: Every scraper failure must be logged 
with full error details. FAISS index staleness must 
be detectable and auto-recoverable. Every Gemini call 
has a timeout of 30 seconds and a fallback response.
PDF generation failure returns a clear error, never 
a malformed file.

HONESTY: Placement likelihood must be conservative 
not optimistic. If a fresher has 0 relevant skills 
for a role, show low likelihood clearly with specific 
reasons. Do not inflate scores to make users feel good.
A user who gets false hope and wastes 20 applications 
is worse off than a user who gets honest assessment 
and applies to 5 well-matched roles.

ERROR MESSAGES: Every error shown to users must be 
human readable and actionable. Never show a stack 
trace or a generic "something went wrong". Tell the 
user what happened and what they can do about it.

## Architecture Updates (Latest)

SCHEDULER: Three APScheduler jobs not one:
- Every 2 hours: RemoteOK, Arbeitnow (no rate limits)
- Every 6 hours: Adzuna, JSearch (API rate limited)
- Every 24 hours: RSS feeds
Each job runs independently. One failure never stops others.

FAISS: Incremental updates only.
New jobs: faiss_index.add(embedding) directly.
Full rebuild: only on server restart from Supabase.
Never rebuild index mid-operation.

ELIGIBILITY FILTER: Runs before FAISS search.
required_exp > candidate_years + 2: multiply score by 0.4
required_exp > candidate_years: multiply score by 0.7
fresher-tagged jobs: never penalized regardless.

PLACEMENT INTELLIGENCE: Every match result includes:
placement_likelihood (high/medium/low)
placement_reason (skills driving assessment)
blocking_skill (most impactful missing skill)
recommended_action (one specific next step)

SKILL ONTOLOGY: All skills normalized before matching.
skill_ontology.py handles synonyms:
JS → JavaScript, PyTorch/Torch → pytorch,
ML → machine learning, k8s → kubernetes etc.

COMPANY INTELLIGENCE: company_intel.json in backend/data/
150+ companies with tech_stack, hires_freshers,
interview_difficulty, work_mode, avg_salary_fresher.
Automatically enriches job match results.

CERTIFICATION RECOMMENDER: cert_recommender.py service.
certifications.json in backend/data/ with 30+ entries.
Ranked by: gap coverage (40pts), role relevance (30pts),
issuer value (20pts), completely free (10pts).
Sources: Google, Microsoft, IBM, AWS, NVIDIA, Kaggle,
Infosys Springboard, NPTEL, Cisco, MongoDB, Meta.

NOTIFICATIONS: When scraper finds job with 
estimated_match > 90% for active user:
trigger immediate Resend email.
Daily digest at 8 AM for all active users.

NEW FILES TO CREATE:
backend/services/eligibility.py
backend/services/skill_ontology.py
backend/services/placement_intel.py
backend/services/cert_recommender.py
backend/services/company_intel.py
backend/services/candidate_profile.py
backend/data/certifications.json
backend/data/company_intel.json
## Engineering Standards

API VERSIONING
- All backend endpoints use /api/v1/.
- Never expose unversioned production APIs.

API RESPONSE FORMAT
Every endpoint returns structured JSON.

Success:
{
  "success": true,
  "message": "...",
  "data": { ... }
}

Validation Error:
{
  "success": false,
  "message": "...",
  "errors": [...]
}

Internal Error:
{
  "success": false,
  "message": "...",
  "error_code": "..."
}

Never expose stack traces to users.

SERVICE DEPENDENCIES
Routers may call Services.
Services may call Database and Utilities.
Services must never call Routers.
Database layer never imports Services.
Avoid circular imports.

DATABASE
Database schema and Pydantic models must remain synchronized.
Never manually edit production schema.
Schema changes require migrations.

UTILITIES
Reusable code belongs in backend/utils/.
Avoid duplicating helper logic across services.

PERFORMANCE
Prioritize correctness over optimization.
Only optimize after measuring bottlenecks.
Do not introduce caching or concurrency without need.

AI VALIDATION
Never trust LLM output directly.
Always:
- Validate JSON
- Validate required fields
- Validate data types
- Log malformed responses
- Use fallback behavior on failure

DOCUMENTATION
Every service should document:
Purpose
Inputs
Outputs
Dependencies
Side effects

Comments explain WHY, not WHAT.