# CareerLens AI

AI-powered career intelligence platform for freshers and job seekers.

## Tech Stack

### Backend
- Python 3.11
- FastAPI (async)
- Supabase (PostgreSQL + Auth + Storage)
- PyMuPDF (PDF parsing)
- sentence-transformers (BAAI/bge-m3)
- faiss-cpu (vector search)
- FlagEmbedding (BAAI/bge-reranker-base)
- google-generativeai (Gemini 2.0 Flash)
- WeasyPrint + Jinja2 (PDF generation)
- APScheduler (job scheduling)
- spaCy en_core_web_sm (NLP)
- Resend (email)

### Frontend
- React 18 + Vite
- Tailwind CSS
- Axios
- Recharts
- React Router v6
- Lucide React (icons)

## Project Structure

```
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
│       ├── components/
│       └── api/
├── logs/
├── .env
└── requirements.txt
```

## Setup Instructions

### Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:
```
GEMINI_API_KEY=your_gemini_api_key_here
ADZUNA_APP_ID=your_adzuna_app_id_here
ADZUNA_APP_KEY=your_adzuna_app_key_here
JSEARCH_API_KEY=your_jsearch_api_key_here
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_KEY=your_supabase_anon_key_here
RESEND_API_KEY=your_resend_api_key_here
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

5. Run the backend:
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Run the frontend:
```bash
npm run dev
```

## Database Tables (Supabase PostgreSQL)

- **cvs**: id, user_id, raw_text, structured_json, file_url, version, created_at
- **jobs**: id, title, company, location, description, source, experience_level, redirect_url, date_posted, date_fetched, embedding
- **sessions**: id, user_id, job_id, match_score, ats_score, skill_gap, rewritten_cv_url, created_at
- **interview_sessions**: id, user_id, job_id, questions_answers, dimension_scores, overall_score, difficulty, created_at
- **tracker**: id, user_id, job_id, status, notes, updated_at
- **roadmaps**: id, user_id, target_role, milestones, completion, created_at
- **profiles**: id, user_id, full_name, email, phone, location, linkedin_url, github_url, portfolio_url, target_title, years_experience, expected_salary, work_preference, education, experience, skills, certifications, cover_letter_templates

## Modules

1. **CV Parser** - Upload and parse resumes using PyMuPDF + Gemini validation
2. **Job Collection Engine** - 4 scrapers (Adzuna, JSearch, RemoteOK, ArbeitNow) + scheduler
3. **Semantic Matching Engine** - BGE-M3 + FAISS + Reranker for job matching
4. **Resume Optimizer** - Gemini JSON → WeasyPrint PDF generation
5. **Resume Builder** - Form-based resume creation with AI assistance
6. **ATS Checker** - CareerLens Resume Readiness Score (standalone, no login required)
7. **Career Roadmap** - Skill gap analysis → weekly learning plan
8. **Interview Coach** - AI-generated questions + evaluation + difficulty adjustment
9. **Career Co-pilot** - RAG over user career data for personalized advice
10. **Job Tracker** - Track application status across all jobs
11. **Easy Apply** - One-click job application
12. **Profile** - Comprehensive career profile management
13. **Dashboard** - Unified view of all modules with metrics

## Architecture Rules

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

```
keyword_score    = matched JD noun phrases / total * 40
section_score    = present sections / 6 * 20
parsability      = no tables/images/columns detected * 20
chronology       = consistent date format * 10
quantification   = bullets with numbers / total bullets * 10
TOTAL MAX = 100
```

Called "CareerLens Resume Readiness Score" not "ATS Score". Uses spaCy for noun phrase extraction.

## AI Agents

CareerLens AI is designed around specialized AI experiences that
share common backend services, candidate data, and controlled tools.

### 1. Career Copilot
The central career assistant. Helps users understand their career
position, coordinate tasks across CareerLens, and decide what to
focus on next.

### 2. Job Search Agent
Supports job discovery, candidate-job matching, skill-gap analysis,
eligibility assessment, and application tracking.

### 3. Resume Intelligence Agent
Analyzes resume content, identifies weaknesses, suggests
evidence-based improvements, and supports user-approved revisions.

### 4. Interview Preparation Agent
Conducts adaptive mock interviews, evaluates responses against
defined rubrics, identifies weaknesses, and recommends targeted
practice.

### 5. Career Roadmap Agent
Creates and adapts learning plans based on career goals, skill gaps,
available time, and verified progress.

### Architecture

The agents share backend services and do not independently implement
duplicate business logic. Each agent has a defined scope, controlled
tools, validated outputs, and access limited by the user's
permissions.

Agent actions that modify user data must follow explicit
authorization and confirmation rules.

Implementation status: Planned. Individual agents will be marked
implemented only after their workflows, integrations, and tests
are complete.

## Agent Architecture Rules

- Agents use shared backend services rather than duplicating business logic.
- Each agent has a clearly defined scope and a limited set of tools.
- Tool inputs and outputs are validated using defined schemas.
- Agents must not fabricate jobs, candidate qualifications, or achievements.
- Semantic similarity must not be presented as the probability of being hired.
- Agents must distinguish verified information from estimates and uncertainty.
- User-owned data must be protected by authentication, authorization,
  and database access policies.
- Agents must not modify resumes, profiles, roadmaps, or application
  records without following the applicable approval rules.
- LLM responses must be validated before they are used by the application.
- Agent failures must produce clear errors or controlled fallbacks.
- Sensitive resume content must not be written to application logs.
- Agent workflows must be tested for correctness, permissions,
  failure handling, and unintended actions.

## Module Build Order

Build in this exact sequence, test each before moving on:
1. Project setup.
2. CV Parser.
3. Canonical Candidate Profile.
4. Document Analyzer.
5. Structured extraction and schema validation.
6. Normalization Engine and Skill Ontology.
7. Semantic Embeddings.
8. CV pipeline integration and upload API.
9. Connect Resume Center to the backend.
10. Job Collection Engine.
11. Semantic Matching Engine.
12. Connect Find Jobs and matching explanations.
13. Resume Optimizer and Resume Builder.
14. Resume Readiness.
15. Career Roadmap and Roadmap Agent.
16. Interview Coach and Interview Preparation Agent.
17. Job Tracker, Profile, and related integrations.
18. Career Copilot and its controlled tools.
19. Job Search Agent enhancements and cross-module   coordination.
20. Dashboard, notifications, security hardening, testing, and deployment.
## License

MIT
