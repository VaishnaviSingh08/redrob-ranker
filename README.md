# 🏆 Intelligent Candidate Ranker
**Team Alcatraz** — Redrob Data & AI Challenge Submission(Track 1)

Traditional candidate filtering relies on keyword matching. If a job says "NLP" and a candidate writes "natural language processing", they get missed. We built a two stage AI ranking engine that understands meaning, not just words, and integrates deep behavioral signals to surface the best candidates from a pool of 100,000 profiles.

## System Architecture

Job Description (.docx / .txt)
        
    load_jd() + extract_jd_fields()
        
{
  raw text          → for embeddings & cross-encoder
  exp_range         → for experience multiplier
  notice_limit      → for notice period scoring
  locations         → for location scoring
  skills_required   → for skill scoring
}
        
candidates.jsonl → 100,000 candidate profiles
        

STAGE 1 — Score all 100,000 candidates
For each candidate, 5 independent scores:
  Semantic Score      (40%) — sentence embedding cosine similarity
  Skills Score        (20%) — JD-aware skill matching
  Career Relevance    (15%) — actual work history vs JD concepts
  Education Score     (15%) — institution tier + field relevance
  Behavioral Score    (10%) — activity, notice, github, location

weighted combine:
 title penalty multiplier       (dynamic, JD-aware)
 experience range multiplier    (pulled from JD)
 response rate multiplier
 notice period multiplier
 honeypot detection → score = 0.0 if flagged

Sort all 100,000 → Top 500

STAGE 2 — Cross-Encoder Reranking
Cross-encoder reads JD + candidate together
 deep pairwise relevance score
 new_final = 60% cross-encoder + 40% stage1_score
 Re-sort Top 500
OUTPUT: Top 100 → submission.csv

## Models Used

all-MiniLM-L6-v2: Sentence embeddings for semantic similarity
cross-encoder/ms-marco-MiniLM-L-6-v2: Precision re-ranking of top 500
Both are loaded via sentence-transformers and cached in memory after first load.

## Project Structure
redrob-ranker/
 app.py                  # Streamlit web UI
 rank.py                 # Core ranking logic
 data/
  job_description.docx   # Sample JD
  candidates.jsonl       # Candidate profiles
 .streamlit/
  config.toml            # Upload size config (500MB)
 venv/                   # Python virtual environment
 requirements.txt

## Technical Choices & Methodology
### Why Sentence Embeddings over TF-IDF
TF-IDF counts word overlap. It would score "NLP experience" vs "built text classification models" as zero similarity because no words match.

`all-MiniLM-L6-v2` maps both sentences into a 384-dimensional semantic space trained on millions of sentence pairs. Conceptually related phrases produce similar vectors regardless of exact wording which is exactly what the challenge asked for when it said "move beyond keywords."

### Why a Two-Stage Pipeline
The embedding model encodes the JD and candidate separately then compares vectors, fast but loses inter-text context. The cross-encoder reads both texts together, much more accurate but slow.
Running the cross-encoder on 100,000 candidates would take hours. Running it on just the top 500 from Stage 1 takes under 15 seconds. We get the speed of embeddings and the accuracy of the cross-encoder combined.

### Why JD-Aware Scoring
Every scoring parameter is extracted dynamically from the JD at runtime:
Experience range: parsed via regex
Notice period limit: parsed via regex
Preferred locations: keyword scan
Required skills: keyword scan

### Honeypot Detection
We flag and eliminate suspicious profiles with 2+ of these signals:
More than 8 "expert" level skills (unrealistically inflated)
Perfect scores across response rate, interview completion, and offer acceptance
High applications but near-zero recruiter response rate
GitHub score of 100 with zero career history
Profile completeness 100% but missing headline or summary

### Career Relevance Score
A fifth scoring dimension that checks if a candidate's actual work history supports their claimed skills. Rewards candidates who have genuinely done the work, deployed models, built pipelines, worked with NLP/embeddings/ranking systems rather than just listing keywords on their profile.

### Behavioral Signal Integration

GitHub activity score: Do they actually code?
Recruiter response rate: Will they engage?
Notice period: Can they join quickly?
`open_to_work` flag: Are they actively looking?
Last active date: Are they still in the market?
Offer acceptance rate: Will they actually join if selected?
Interview completion rate: Are they serious candidates?
Saved by recruiters (30d): Social proof from other recruiters
Applications submitted (30d): Active job search signal
Profile completeness: Are they presenting themselves seriously?

GitHub activity only contributes for technical JDs, detected automatically from JD content.

### Skill Scoring Depth
Each skill is scored across four dimensions:
 Proficiency level — beginner / intermediate / advanced / expert
 Duration — how long they used the skill (months)
 Endorsements — peer validated
 Assessment score — platform tested score

### Education Scoring
Scans all education entries and returns the best score, prevents penalizing candidates whose high school appears before their IIT/IIM degree in the data.

### Title Penalty

Fully dynamic no hardcoded blacklist. Checks if a candidate's title has meaningful overlap with JD terminology, excluding generic words like "engineer", "senior", "lead". Unrelated titles receive a 0.5 multiplier. A soft nudge, not elimination, since a career-switcher with strong ML skills should still rank if their skills and career history support it.


## Scoring Formula

Stage 1 Score:
final = (0.40 × semantic)
      + (0.20 × skills)
      + (0.15 × career_relevance)
      + (0.15 × education)
      + (0.10 × behavioral)
      × title_penalty
      × experience_multiplier
      × response_rate_multiplier
      × notice_period_multiplier

Stage 2 Score:
final = (0.60 × cross_encoder_score) + (0.40 × stage1_score)

The original score retains 40% weight in Stage 2 because it carries behavioral signals the cross-encoder never sees, those live in `redrob_signals`, not in profile text.


## Tech Stack

Language: Python 3.11 
Semantic Embeddings: `sentence-transformers` — `all-MiniLM-L6-v2` 
Cross-Encoder Reranking: `sentence-transformers` — `cross-encoder/ms-marco-MiniLM-L-6-v2`
JD Parsing: `python-docx`, `re`
Data Processing: `pandas`, `json`
Frontend: `streamlit`

## Setup & Usage
1. Create and activate virtual environment and install dependencies
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt

2. Configure upload size (run once)
mkdir -p .streamlit
echo "[server]
maxUploadSize = 500" > .streamlit/config.toml

4. Run the app
streamlit run app.py --server.port 8502

### Prepare data
data/
 candidates.jsonl (candidate profiles)
 job_description.docx (job description)
 validate_submission.py (provided by organizers)

### Run ranking script
python rank.py
Output: `submission.csv` with top 100 ranked candidates.

### Validate output
python data/validate_submission.py submission.csv
Expected output:Submission is valid.

## Output Format
candidate_id, rank, score, reasoning
CAND_0002025, 1, 0.9049, "Senior AI Engineer with 5.9 yrs experience; top skills: FAISS, OpenSearch, Weaviate; notice period: 30 days; response rate: 0.8; github score: 96.9."

## Using the App
1. Upload your Job Description (.docx or .txt) in the sidebar
2. Upload your Candidates file (.json or .jsonl) in the sidebar
3. Adjust Top N slider to control how many results to show
4. Click Run Ranking
5. View ranked candidates with score breakdowns
6. Download submission.csv


## Repository Structure
redrob-ranker/
rank.py              (main ranking engine (CLI))
app.py               (streamlit web interface)
requirements.txt     (dependencies)
README.md            (this file)
submission.csv       (ranked output (top 100))
data/
  candidates.jsonl
  job_description.docx
 validate_submission.py

Team Alcatraz — Built for the Redrob Data & AI Challenge*
