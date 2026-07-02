import json
import csv
import io
import re
from datetime import date, datetime
import numpy as np
import streamlit as st
from sentence_transformers import util

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="Candidate Ranking System",
                   page_icon="🏆", layout="wide")

# ── Lazy-load heavy models ────────────────────────────────


@st.cache_resource
def load_models():
    from sentence_transformers import CrossEncoder, SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return sbert, ce


# ── Constants ─────────────────────────────────────────────
TOP_RERANK = 500
TOP_OUTPUT = 100

# ── JD Parsing ────────────────────────────────────────────


def extract_jd_fields(text):
    text_lower = text.lower()

    exp_range = {'min': 0, 'max': 99}
    match = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)\s*years?', text_lower)
    if match:
        exp_range = {'min': int(match.group(1)), 'max': int(match.group(2))}
    else:
        match = re.search(
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience', text_lower)
        if match:
            exp_range = {'min': int(match.group(1)), 'max': 99}

    notice_limit = 90
    match = re.search(
        r'notice\s*period\s*(?:under|below|of|:)?\s*(\d+)\s*days?', text_lower)
    if match:
        notice_limit = int(match.group(1))

    KNOWN_CITIES = [
        'pune', 'noida', 'delhi', 'mumbai', 'hyderabad',
        'bangalore', 'bengaluru', 'chennai', 'gurgaon',
        'gurugram', 'kolkata', 'ahmedabad', 'remote'
    ]
    locations = [city for city in KNOWN_CITIES if city in text_lower]

    SKILL_KEYWORDS = [
        'python', 'pytorch', 'tensorflow', 'huggingface', 'transformers',
        'nlp', 'llm', 'rag', 'fine-tuning', 'embeddings', 'vector search',
        'faiss', 'pinecone', 'weaviate', 'elasticsearch', 'opensearch',
        'mlops', 'recommendation', 'ranking', 'retrieval', 'a/b testing',
        'kafka', 'spark', 'airflow', 'bert', 'gpt', 'milvus', 'qdrant',
        'sentence transformers', 'peft', 'lora', 'deep learning',
        'machine learning', 'model deployment', 'inference optimization'
    ]
    skills_required = [
        skill for skill in SKILL_KEYWORDS if skill in text_lower]

    return {
        'exp_range': exp_range,
        'notice_limit': notice_limit,
        'locations': locations,
        'skills_required': skills_required
    }


def load_jd_from_upload(uploaded_file):
    name = uploaded_file.name
    if name.endswith('.docx'):
        from docx import Document
        doc = Document(uploaded_file)
        raw_text = '\n'.join([p.text for p in doc.paragraphs])
    else:
        raw_text = uploaded_file.read().decode('utf-8')
    return {'raw': raw_text, 'fields': extract_jd_fields(raw_text)}


def load_candidates_from_upload(uploaded_file):
    name = uploaded_file.name
    content = uploaded_file.read().decode('utf-8')
    if name.endswith('.jsonl'):
        candidates = [json.loads(line)
                      for line in content.splitlines() if line.strip()]
    else:
        candidates = json.loads(content)
    return candidates


# ── Scoring functions (identical to rank.py) ─────────────
def candidate_to_text(candidate):
    parts = []
    profile = candidate.get('profile', {})
    parts.append(profile.get('headline', ''))
    parts.append(profile.get('summary', ''))
    parts.append(profile.get('current_title', ''))
    parts.append(profile.get('current_industry', ''))
    for job in candidate.get('career_history', []):
        parts.append(job.get('title', ''))
        parts.append(job.get('description', ''))
        parts.append(job.get('industry', ''))
    for skill in candidate.get('skills', []):
        parts.append(skill.get('name', ''))
    for edu in candidate.get('education', []):
        parts.append(edu.get('degree', ''))
        parts.append(edu.get('field_of_study', ''))
    for cert in candidate.get('certifications', []):
        parts.append(cert.get('name', ''))
    return " ".join(parts)


def truncate_for_cross_encoder(text, max_words=300):
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])


def title_penalty(candidate, JOB_DESCRIPTION, JD_FIELDS):
    current_title = candidate.get('profile', {}).get(
        'current_title', '').lower()
    skills_required = JD_FIELDS['skills_required']
    if any(skill in current_title for skill in skills_required):
        return 1.0
    IGNORE_WORDS = {'engineer', 'senior', 'lead', 'staff', 'manager',
                    'the', 'and', 'with', 'for', 'at', 'in', 'of'}
    jd_words = set(JOB_DESCRIPTION.lower().split()) - IGNORE_WORDS
    title_words = set(current_title.split()) - IGNORE_WORDS
    if jd_words & title_words:
        return 1.0
    return 0.5


def score_behavioral(candidate, JOB_DESCRIPTION, JD_FIELDS):
    score = 0.0
    signals = candidate.get('redrob_signals', {})
    profile = candidate.get('profile', {})

    if signals.get('open_to_work_flag'):
        score += 0.10

    last_active_str = signals.get('last_active_date', '2000-01-01')
    try:
        last_active = datetime.strptime(last_active_str, '%Y-%m-%d').date()
        days_inactive = (date.today() - last_active).days
    except:
        days_inactive = 999
    if days_inactive <= 7:
        score += 0.15
    elif days_inactive <= 30:
        score += 0.10
    elif days_inactive <= 90:
        score += 0.07
    elif days_inactive <= 180:
        score += 0.03

    response_rate = signals.get('recruiter_response_rate', 0)
    if response_rate >= 0.7:
        score += 0.10
    elif response_rate >= 0.4:
        score += 0.07
    elif response_rate >= 0.2:
        score += 0.03
    elif response_rate < 0.2:
        score -= 0.10

    avg_response_time = signals.get('avg_response_time_hours', 999)
    if avg_response_time <= 2:
        score += 0.05
    elif avg_response_time <= 12:
        score += 0.03
    elif avg_response_time <= 24:
        score += 0.01

    notice_limit = JD_FIELDS['notice_limit']
    notice = signals.get('notice_period_days', 90)
    if notice <= notice_limit:
        score += 0.10
    elif notice <= notice_limit + 30:
        score += 0.06
    elif notice <= notice_limit + 60:
        score += 0.02
    else:
        score -= 0.25

    preferred_cities = JD_FIELDS['locations']
    location = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
    willing_relocate = signals.get('willing_to_relocate', False)
    if country == 'india':
        if any(city in location for city in preferred_cities):
            score += 0.05
        elif willing_relocate:
            score += 0.03
    elif willing_relocate:
        score += 0.01

    TECHNICAL_SIGNALS = ['python', 'pytorch', 'tensorflow', 'nlp', 'llm',
                         'java', 'javascript', 'backend', 'frontend', 'devops']
    is_technical_jd = any(t in JOB_DESCRIPTION.lower()
                          for t in TECHNICAL_SIGNALS)
    github = signals.get('github_activity_score', -1)
    if is_technical_jd:
        if github == -1:
            score -= 0.05
        elif github >= 70:
            score += 0.10
        elif github >= 40:
            score += 0.07
        elif github >= 10:
            score += 0.03

    interview_rate = signals.get('interview_completion_rate', 0)
    if interview_rate >= 0.8:
        score += 0.08
    elif interview_rate >= 0.6:
        score += 0.04

    offer_rate = signals.get('offer_acceptance_rate', -1)
    if offer_rate >= 0.8:
        score += 0.05
    elif offer_rate >= 0.5:
        score += 0.03

    saved = signals.get('saved_by_recruiters_30d', 0)
    if saved >= 10:
        score += 0.05
    elif saved >= 5:
        score += 0.03
    elif saved >= 1:
        score += 0.01

    applications = signals.get('applications_submitted_30d', 0)
    if applications >= 5:
        score += 0.05
    elif applications >= 2:
        score += 0.03
    elif applications >= 1:
        score += 0.01

    completeness = signals.get('profile_completeness_score', 0)
    if completeness >= 90:
        score += 0.05
    elif completeness >= 70:
        score += 0.03

    if signals.get('verified_email'):
        score += 0.02
    if signals.get('verified_phone'):
        score += 0.02
    if signals.get('linkedin_connected'):
        score += 0.02

    work_mode = signals.get('preferred_work_mode', '')
    jd_lower = JOB_DESCRIPTION.lower()
    if 'remote' in jd_lower:
        preferred_mode = 'remote'
    elif 'onsite' in jd_lower or 'on-site' in jd_lower:
        preferred_mode = 'onsite'
    elif 'hybrid' in jd_lower:
        preferred_mode = 'hybrid'
    else:
        preferred_mode = None
    if preferred_mode:
        if work_mode == preferred_mode:
            score += 0.05
        elif work_mode == 'flexible':
            score += 0.02

    return max(0.0, min(1.0, score))


def is_honeypot(candidate):
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    flags = 0
    expert_skills = [s for s in skills if s.get('proficiency') == 'expert']
    if len(expert_skills) > 8:
        flags += 1
    if (signals.get('recruiter_response_rate', 0) == 1.0 and
        signals.get('interview_completion_rate', 0) == 1.0 and
            signals.get('offer_acceptance_rate', 0) == 1.0):
        flags += 1
    if (signals.get('applications_submitted_30d', 0) >= 10 and
        signals.get('open_to_work_flag') and
            signals.get('recruiter_response_rate', 0) < 0.1):
        flags += 1
    if signals.get('github_activity_score', 0) == 100 and len(career) == 0:
        flags += 1
    if signals.get('profile_completeness_score', 0) == 100:
        if not profile.get('headline') or not profile.get('summary'):
            flags += 1
    return flags >= 2


def score_skills(candidate, JD_FIELDS):
    signals = candidate.get('redrob_signals', {})
    assessment_scores = signals.get('skill_assessment_scores', {})
    proficiency_weights = {'expert': 1.0, 'advanced': 0.8,
                           'intermediate': 0.5, 'beginner': 0.2}
    HIGH_VALUE_SKILLS = JD_FIELDS['skills_required']
    total_score = 0.0
    matched_skills = 0
    for skill in candidate.get('skills', []):
        skill_name = skill['name'].lower()
        is_relevant = any(
            hv in skill_name or skill_name in hv for hv in HIGH_VALUE_SKILLS)
        if not is_relevant:
            continue
        matched_skills += 1
        prof = skill.get('proficiency', 'beginner')
        base = proficiency_weights.get(prof, 0.2)
        duration = skill.get('duration_months', 0)
        if duration >= 24:
            duration_boost = 0.3
        elif duration >= 12:
            duration_boost = 0.2
        elif duration >= 6:
            duration_boost = 0.1
        else:
            duration_boost = 0.0
        endorsements = skill.get('endorsements', 0)
        if endorsements >= 20:
            endorsement_boost = 0.2
        elif endorsements >= 10:
            endorsement_boost = 0.1
        elif endorsements >= 5:
            endorsement_boost = 0.05
        else:
            endorsement_boost = 0.0
        assessment_boost = 0.0
        for assessed_skill, test_score in assessment_scores.items():
            if assessed_skill.lower() in skill_name or skill_name in assessed_skill.lower():
                assessment_boost = (test_score / 100) * 0.3
                break
        skill_score = base + duration_boost + endorsement_boost + assessment_boost
        total_score += min(1.0, skill_score)
    if matched_skills == 0:
        return 0.0
    avg = total_score / matched_skills
    breadth_bonus = min(0.2, matched_skills * 0.02)
    return min(1.0, avg + breadth_bonus)


def score_education(candidate):
    education = candidate.get('education', [])
    if not education:
        return 0.0
    tier_scores = {'tier_1': 0.50, 'tier_2': 0.35,
                   'tier_3': 0.20, 'tier_4': 0.10, 'unknown': 0.10}
    RELEVANT_FIELDS = [
        'computer science', 'engineering', 'mathematics', 'statistics',
        'data science', 'artificial intelligence', 'machine learning',
        'information technology', 'physics', 'electronics', 'electrical'
    ]
    best_score = 0.0
    for edu in education:
        score = 0.0
        tier = edu.get('tier', 'unknown')
        score += tier_scores.get(tier, 0.10)
        field = edu.get('field_of_study', '').lower()
        if any(rf in field for rf in RELEVANT_FIELDS):
            score += 0.50
        score = min(1.0, score)
        if score > best_score:
            best_score = score
    return best_score


def score_career_relevance(candidate):
    career = candidate.get('career_history', [])
    skills = candidate.get('skills', [])
    if not career:
        if len(skills) > 5:
            return 0.1
        return 0.0
    career_text = ' '.join([
        job.get('description', '') + ' ' + job.get('title', '')
        for job in career
    ]).lower()
    WORK_CONCEPTS = [
        'recommendation', 'search', 'retrieval', 'ranking', 'pipeline',
        'model', 'training', 'deployed', 'built', 'developed', 'designed',
        'nlp', 'embedding', 'inference', 'fine-tun', 'vector', 'llm',
        'classification', 'generation', 'similarity', 'index'
    ]
    matched = sum(1 for concept in WORK_CONCEPTS if concept in career_text)
    if matched >= 6:
        return 1.0
    elif matched >= 4:
        return 0.75
    elif matched >= 2:
        return 0.50
    elif matched >= 1:
        return 0.25
    else:
        return 0.1


def score_semantic(candidate, embedder, jd_embedding):
    from sentence_transformers import util
    candidate_text = candidate_to_text(candidate)
    candidate_text = truncate_for_cross_encoder(candidate_text, max_words=300)
    candidate_embedding = embedder.encode(
        candidate_text, convert_to_tensor=True)
    similarity = util.cos_sim(jd_embedding, candidate_embedding).item()
    return round(float(similarity), 4)


def generate_reasoning(candidate, scores, JD_FIELDS):
    profile = candidate['profile']
    signals = candidate.get('redrob_signals', {})
    title = profile.get('current_title', 'N/A')
    years = profile.get('years_of_experience', 0)
    notice = signals.get('notice_period_days', 'N/A')
    response = signals.get('recruiter_response_rate', 0)
    github = signals.get('github_activity_score', 'N/A')
    if github == -1:
        github = 'N/A'
    skills_required = JD_FIELDS['skills_required']
    top_skills = [
        s['name'] for s in candidate.get('skills', [])
        if any(hv in s['name'].lower() or s['name'].lower() in hv for hv in skills_required)
    ][:3]
    if not top_skills:
        top_skills = [s['name'] for s in candidate.get('skills', [])][:2]
    skills_text = ', '.join(top_skills)
    reasoning = (
        f"{title} with {years} yrs experience; "
        f"top skills: {skills_text}; "
        f"notice period: {notice} days; "
        f"response rate: {round(response, 2)}; "
        f"github score: {github}."
    )
    return reasoning[:300]


# ── UI ────────────────────────────────────────────────────
st.title("🏆 Candidate Ranking System")

st.sidebar.header("📁 Upload Files")
jd_file = st.sidebar.file_uploader(
    "Upload Job Description (.docx or .txt)", type=["docx", "txt"])
candidates_file = st.sidebar.file_uploader(
    "Upload Candidates JSON / JSONL", type=["json", "jsonl"])
st.sidebar.markdown("---")
top_n = st.sidebar.slider("Show Top N Candidates",
                          min_value=1, max_value=100, value=10)

run_btn = st.sidebar.button("🚀 Run Ranking", use_container_width=True)

if not jd_file or not candidates_file:
    st.info("Upload a Job Description and Candidates file in the sidebar, then click **Run Ranking**.")
    st.stop()

if run_btn:
    # Load JD
    with st.spinner("Parsing job description..."):
        JD = load_jd_from_upload(jd_file)
        JOB_DESCRIPTION = JD['raw']
        JD_FIELDS = JD['fields']

    st.success(
        f"JD loaded. Skills detected: {', '.join(JD_FIELDS['skills_required']) or 'none'}")

    # Load candidates
    with st.spinner("Loading candidates..."):
        candidates = load_candidates_from_upload(candidates_file)
    st.success(f"Loaded {len(candidates)} candidates.")

    # Load models
    with st.spinner("Loading embedding model (first run may take a minute)..."):
        embedder, cross_encoder = load_models()
        jd_embedding = embedder.encode(JOB_DESCRIPTION, convert_to_tensor=True)

# Batch encode all candidates at once
    with st.spinner("Encoding all candidates..."):
        all_texts = []
        for candidate in candidates:
            text = candidate_to_text(candidate)
            text = truncate_for_cross_encoder(text, max_words=300)
            all_texts.append(text)

    all_embeddings = embedder.encode(
        all_texts,
        batch_size=64,
        convert_to_tensor=True,
        show_progress_bar=False
    )

    all_scores = []
    progress = st.progress(0, text="Stage 1: Scoring candidates...")
    for i, candidate in enumerate(candidates):
        semantic = round(
            float(util.cos_sim(jd_embedding, all_embeddings[i]).item()), 4)
        behavioral = score_behavioral(candidate, JOB_DESCRIPTION, JD_FIELDS)
        skills = score_skills(candidate, JD_FIELDS)
        education = score_education(candidate)
        penalty = title_penalty(candidate, JOB_DESCRIPTION, JD_FIELDS)
        career = score_career_relevance(candidate)

        final = round((
            (0.40 * semantic) +
            (0.20 * skills) +
            (0.15 * education) +
            (0.10 * behavioral) +
            (0.15 * career)
        ) * penalty, 4)

        if skills < 0.1:
            final = round(final * 0.75, 4)

        honeypot = is_honeypot(candidate)
        if honeypot:
            final = 0.0

        all_scores.append({
            'candidate_id': candidate['candidate_id'],
            'title': candidate.get('profile', {}).get('current_title', ''),
            'years': candidate.get('profile', {}).get('years_of_experience', 0),
            'semantic': semantic,
            'skills': skills,
            'education': education,
            'behavioral': behavioral,
            'career': career,
            'honeypot': honeypot,
            'final': final
        })
        progress.progress((i + 1) / len(candidates),
                          text=f"Stage 1: {i+1}/{len(candidates)}")

    all_scores.sort(key=lambda x: x['final'], reverse=True)

    # Stage 2: Cross-encoder re-ranking
    top_500 = all_scores[:TOP_RERANK]
    candidate_lookup = {c['candidate_id']: c for c in candidates}

    with st.spinner(f"Stage 2: Cross-encoder re-ranking top {len(top_500)} candidates..."):
        pairs = []
        for result in top_500:
            candidate = candidate_lookup[result['candidate_id']]
            candidate_text = truncate_for_cross_encoder(
                candidate_to_text(candidate))
            pairs.append((JOB_DESCRIPTION, candidate_text))

        ce_scores = cross_encoder.predict(pairs, show_progress_bar=False)
        ce_min = ce_scores.min()
        ce_max = ce_scores.max()
        ce_normalized = (ce_scores - ce_min) / (ce_max - ce_min + 1e-9)

        for i, result in enumerate(top_500):
            old_final = result['final']
            ce_score = float(ce_normalized[i])
            result['final'] = round((0.60 * ce_score) + (0.40 * old_final), 4)
            result['ce_score'] = round(ce_score, 4)

    top_500.sort(key=lambda x: (-x['final'], x['candidate_id']))
    st.success("✅ Ranking complete!")

    # ── Results ───────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"🏅 Top {top_n} Candidates")

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}

    for rank, result in enumerate(top_500[:top_n], 1):
        candidate = candidate_lookup[result['candidate_id']]
        profile = candidate.get('profile', {})
        reasoning = generate_reasoning(candidate, result, JD_FIELDS)
        honeypot = result.get('honeypot', False)

        medal = medals.get(rank, f"#{rank}")
        label = f"{medal} **{profile.get('current_title', 'N/A')}** — Score: `{result['final']}`"
        if honeypot:
            label += " ⚠️ HONEYPOT"

        with st.expander(label):
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Score", result['final'])
            col2.metric("Years Exp", result['years'])
            col3.metric("Semantic", result['semantic'])

            col4, col5, col6 = st.columns(3)
            col4.metric("Skills", result['skills'])
            col5.metric("Education", result['education'])
            col6.metric("Behavioral", result['behavioral'])

            st.markdown(f"**Candidate ID:** `{result['candidate_id']}`")
            st.markdown(f"**Reasoning:** {reasoning}")

            signals = candidate.get('redrob_signals', {})
            st.markdown(f"**Notice Period:** {signals.get('notice_period_days', 'N/A')} days &nbsp;|&nbsp; "
                        f"**Response Rate:** {signals.get('recruiter_response_rate', 'N/A')} &nbsp;|&nbsp; "
                        f"**GitHub Score:** {signals.get('github_activity_score', 'N/A')}")

    # ── CSV Export ────────────────────────────────────────
    st.markdown("---")
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
    for rank, result in enumerate(top_500[:TOP_OUTPUT], 1):
        candidate = candidate_lookup[result['candidate_id']]
        reasoning = generate_reasoning(candidate, result, JD_FIELDS)
        if result.get('honeypot'):
            reasoning = '[HONEYPOT DETECTED] ' + reasoning
        writer.writerow([result['candidate_id'], rank,
                        result['final'], reasoning])

    st.download_button(
        label="⬇️ Download submission.csv",
        data=csv_buffer.getvalue(),
        file_name="submission.csv",
        mime="text/csv"
    )
