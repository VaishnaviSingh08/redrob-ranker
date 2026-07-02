import json
import csv
from datetime import date, datetime
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util
import re
TOP_RERANK = 500
TOP_OUTPUT = 100


def JobDescription(source):
    if isinstance(source, str) and source.endswith(('.txt', '.json', '.docx')):
        if source.endswith('.json'):
            with open(source, 'r') as f:
                data = json.load(f)
                RawText = data.get('description', '')
        elif source.endswith('.docx'):
            from docx import Document
            doc = Document(source)
            RawText = '\n'.join([para.text for para in doc.paragraphs])
        else:
            with open(source, 'r') as f:
                RawText = f.read()
    else:
        RawText = source

    return {
        'raw': RawText,
        'fields': extract_jd_fields(RawText)}


def extract_jd_fields(text):
    TextLower = text.lower()
    ExpRange = {'min': 0, 'max': 99}
    match = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)\s*years?', TextLower)
    if match:
        ExpRange = {'min': int(match.group(1)), 'max': int(match.group(2))}
    else:
        match = re.search(
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience', TextLower)
        if match:
            ExpRange = {'min': int(match.group(1)), 'max': 99}
    NoticeLimit = 90
    match = re.search(
        r'notice\s*period\s*(?:under|below|of|:)?\s*(\d+)\s*days?', TextLower)
    if match:
        NoticeLimit = int(match.group(1))
    KnownCities = ['pune', 'noida', 'delhi', 'mumbai', 'hyderabad', 'bangalore',
                   'bengaluru', 'chennai', 'gurgaon', 'gurugram', 'kolkata', 'ahmedabad', 'remote']
    locations = [city for city in KnownCities if city in TextLower]
    SkillKeywords = ['python', 'pytorch', 'tensorflow', 'huggingface', 'transformers', 'nlp', 'llm', 'rag', 'fine-tuning', 'embeddings', 'vector search', 'faiss', 'pinecone', 'weaviate', 'elasticsearch', 'opensearch', 'mlops', 'recommendation',
                     'ranking', 'retrieval', 'a/b testing', 'kafka', 'spark', 'airflow', 'bert', 'gpt', 'milvus', 'qdrant', 'sentence transformers', 'peft', 'lora', 'deep learning', 'machine learning', 'model deployment', 'inference optimization']
    SkillsRequired = [
        skill for skill in SkillKeywords if skill in TextLower]
    return {'ExpRange': ExpRange, 'NoticeLimit': NoticeLimit, 'locations': locations, 'SkillsRequired': SkillsRequired}


JD = JobDescription('data/job_description.docx')
JOB_DESCRIPTION = JD['raw']
JD_FIELDS = JD['fields']


def LoadCandidates(filepath):
    candidates = []
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            candidates = json.load(f)
    elif filepath.endswith('.jsonl'):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

    print(f"Loaded {len(candidates)} candidates")
    return candidates


def CandidateToText(candidate):
    parts = []
    profile = candidate.get('profile', {})
    parts.append(profile.get('headline', ''))
    parts.append(profile.get('summary', ''))
    parts.append(profile.get('CurrentTitle', ''))
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


def TitlePenalty(candidate):
    CurrentTitle = candidate.get('profile', {}).get(
        'CurrentTitle', '').lower()
    SkillsRequired = JD_FIELDS['SkillsRequired']
    if any(skill in CurrentTitle for skill in SkillsRequired):
        return 1.0
    IgnoreWords = {'engineer', 'senior', 'lead', 'staff', 'manager',
                   'the', 'and', 'with', 'for', 'at', 'in', 'of'}
    JdWords = set(JOB_DESCRIPTION.lower().split()) - IgnoreWords
    TitleWords = set(CurrentTitle.split()) - IgnoreWords
    if JdWords & TitleWords:
        return 1.0
    return 0.5


def ScoreBehaviour(candidate):
    score = 0.0
    signals = candidate.get('redrob_signals', {})
    profile = candidate.get('profile', {})
    if signals.get('open_to_work_flag'):
        score += 0.10
    LastActiveStr = signals.get('LastActive_date', '2000-01-01')
    LastActive = datetime.strptime(LastActiveStr, '%Y-%m-%d').date()
    DaysInactive = (date.today() - LastActive).days
    if DaysInactive <= 7:
        score += 0.15
    elif DaysInactive <= 30:
        score += 0.10
    elif DaysInactive <= 90:
        score += 0.07
    elif DaysInactive <= 180:
        score += 0.03
    ResponseRate = signals.get('recruiter_ResponseRate', 0)
    if ResponseRate >= 0.7:
        score += 0.10
    elif ResponseRate >= 0.4:
        score += 0.07
    elif ResponseRate >= 0.2:
        score += 0.03
    elif ResponseRate < 0.2:
        score -= 0.10
    AvgResponseTime = signals.get('avg_response_time_hours', 999)
    if AvgResponseTime <= 2:
        score += 0.05
    elif AvgResponseTime <= 12:
        score += 0.03
    elif AvgResponseTime <= 24:
        score += 0.01
    NoticeLimit = JD_FIELDS['NoticeLimit']
    notice = signals.get('notice_period_days', 90)
    if notice <= NoticeLimit:
        score += 0.10
    elif notice <= NoticeLimit + 30:
        score += 0.06
    elif notice <= NoticeLimit + 60:
        score += 0.02
    else:
        score -= 0.25
    PreferredCities = JD_FIELDS['locations']
    location = profile.get('location', '').lower()
    country = profile.get('country', '').lower()
    WillingRelocate = signals.get('willing_to_relocate', False)
    if country == 'india':
        if any(city in location for city in PreferredCities):
            score += 0.05
        elif WillingRelocate:
            score += 0.03
    elif WillingRelocate:
        score += 0.01
    TECHNICALSIGNALS = ['python', 'pytorch', 'tensorflow', 'nlp', 'llm',
                        'java', 'javascript', 'backend', 'frontend', 'devops']
    IsTechnicalJd = any(t in JOB_DESCRIPTION.lower()
                        for t in TECHNICALSIGNALS)
    github = signals.get('github_activity_score', -1)
    if IsTechnicalJd:
        if github == -1:
            score -= 0.05
        elif github >= 70:
            score += 0.10
        elif github >= 40:
            score += 0.07
        elif github >= 10:
            score += 0.03
    InterviewRate = signals.get('interview_completion_rate', 0)
    if InterviewRate >= 0.8:
        score += 0.08
    elif InterviewRate >= 0.6:
        score += 0.04
    OfferRate = signals.get('offer_acceptance_rate', -1)
    if OfferRate >= 0.8:
        score += 0.05
    elif OfferRate >= 0.5:
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
    WorkMode = signals.get('preferred_work_mode', '')
    jd_lower = JOB_DESCRIPTION.lower()
    if 'remote' in jd_lower:
        PreferredMode = 'remote'
    elif 'onsite' in jd_lower or 'on-site' in jd_lower:
        PreferredMode = 'onsite'
    elif 'hybrid' in jd_lower:
        PreferredMode = 'hybrid'
    else:
        PreferredMode = None
    if PreferredMode:
        if WorkMode == PreferredMode:
            score += 0.05
        elif WorkMode == 'flexible':
            score += 0.02
    return max(0.0, min(1.0, score))


def IsHoneypot(candidate):
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    flags = 0
    ExpertSkills = [s for s in skills if s.get('proficiency') == 'expert']
    if len(ExpertSkills) > 8:
        flags += 1
    if (signals.get('recruiter_ResponseRate', 0) == 1.0 and
        signals.get('interview_completion_rate', 0) == 1.0 and
            signals.get('offer_acceptance_rate', 0) == 1.0):
        flags += 1
    if (signals.get('applications_submitted_30d', 0) >= 10 and
        signals.get('open_to_work_flag') and
            signals.get('recruiter_ResponseRate', 0) < 0.1):
        flags += 1
    if signals.get('github_activity_score', 0) == 100 and len(career) == 0:
        flags += 1
    if signals.get('profile_completeness_score', 0) == 100:
        if not profile.get('headline') or not profile.get('summary'):
            flags += 1
    return flags >= 2


def ScoreSkill(candidate):
    signals = candidate.get('redrob_signals', {})
    AssessmentScores = signals.get('skill_AssessmentScores', {})
    proficiency_weights = {
        'expert': 1.0,
        'advanced': 0.8,
        'intermediate': 0.5,
        'beginner': 0.2
    }
    HIGHVALUESKILLS = JD_FIELDS['SkillsRequired']
    TotalScore = 0.0
    MatchedSkills = 0
    for skill in candidate.get('skills', []):
        skill_name = skill['name'].lower()
        is_relevant = any(
            hv in skill_name or skill_name in hv
            for hv in HIGHVALUESKILLS
        )
        if not is_relevant:
            continue
        MatchedSkills += 1
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
            EndorsementBoost = 0.2
        elif endorsements >= 10:
            EndorsementBoost = 0.1
        elif endorsements >= 5:
            EndorsementBoost = 0.05
        else:
            EndorsementBoost = 0.0
        AssessmentBoost = 0.0
        for AssessedSkill, test_score in AssessmentScores.items():
            if AssessedSkill.lower() in skill_name or skill_name in AssessedSkill.lower():
                AssessmentBoost = (test_score / 100) * 0.3
                break
        SkillScore = base + duration_boost + EndorsementBoost + AssessmentBoost
        TotalScore += min(1.0, SkillScore)
    if MatchedSkills == 0:
        return 0.0
    avg = TotalScore / MatchedSkills
    breadth_bonus = min(0.2, MatchedSkills * 0.02)
    return min(1.0, avg + breadth_bonus)


def ScoreEducation(candidate):
    education = candidate.get('education', [])
    if not education:
        return 0.0
    TierScores = {
        'tier_1': 0.50,
        'tier_2': 0.35,
        'tier_3': 0.20,
        'tier_4': 0.10,
        'unknown': 0.10
    }
    RELEVANTFIELDS = [
        'computer science', 'engineering', 'mathematics',
        'statistics', 'data science', 'artificial intelligence',
        'machine learning', 'information technology',
        'physics', 'electronics', 'electrical'
    ]
    best_score = 0.0
    for edu in education:
        score = 0.0
        tier = edu.get('tier', 'unknown')
        score += TierScores.get(tier, 0.10)
        field = edu.get('field_of_study', '').lower()
        if any(rf in field for rf in RELEVANTFIELDS):
            score += 0.50
        score = min(1.0, score)
        if score > best_score:
            best_score = score
    return best_score


def truncate(text, max_words=300):
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])


def careerrel(candidate):
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
    WORKCONCEPTS = [
        'recommendation', 'search', 'retrieval', 'ranking', 'pipeline',
        'model', 'training', 'deployed', 'built', 'developed', 'designed',
        'nlp', 'embedding', 'inference', 'fine-tun', 'vector', 'llm',
        'classification', 'generation', 'similarity', 'index'
    ]
    matched = sum(1 for concept in WORKCONCEPTS if concept in career_text)
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


all_scores = []
candidates = LoadCandidates('data/candidates.jsonl')
print("embedding model")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
JdEmbedding = embedder.encode(JOB_DESCRIPTION, convert_to_tensor=True)
print("Encoding all candidates")
AllTexts = []
for candidate in candidates:
    text = CandidateToText(candidate)
    text = truncate(text, max_words=300)
    AllTexts.append(text)
AllEmbeddings = embedder.encode(
    AllTexts,
    batch_size=64,
    convert_to_tensor=True,
    show_progress_bar=True
)
for i, candidate in enumerate(candidates):
    semantic = round(
        float(util.cos_sim(JdEmbedding, AllEmbeddings[i]).item()), 4)
    behavioral = ScoreBehaviour(candidate)
    skills = ScoreSkill(candidate)
    education = ScoreEducation(candidate)
    penalty = TitlePenalty(candidate)
    career = careerrel(candidate)
    final = round((
        (0.40 * semantic) +
        (0.20 * skills) +
        (0.15 * education) +
        (0.10 * behavioral) +
        (0.15 * career)
    ) * penalty, 4)
    years = candidate.get('profile', {}).get('years_of_experience', 0)
    min_exp = JD_FIELDS['ExpRange']['min']
    max_exp = JD_FIELDS['ExpRange']['max']
    skills_score = skills
    if skills_score < 0.1:
        final = round(final * 0.75, 4)
    if IsHoneypot(candidate):
        final = 0.0
    all_scores.append({
        'candidate_id': candidate['candidate_id'],
        'title': candidate.get('profile', {}).get('CurrentTitle', ''),
        'years': candidate.get('profile', {}).get('years_of_experience', 0),
        'semantic':     semantic,
        'skills':       skills,
        'education':    education,
        'behavioral':   behavioral,
        'honeypot':     IsHoneypot(candidate),
        'final':        final
    })
all_scores.sort(key=lambda x: x['final'], reverse=True)


print("cross-encoder model")
CrossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
top_500 = all_scores[:TOP_RERANK]
candidate_lookup = {c['candidate_id']: c for c in candidates}
print("Running cross-encoder on top 500 candidates")
pairs = []
for result in top_500:
    candidate = candidate_lookup[result['candidate_id']]
    CandidateText = CandidateToText(candidate)
    CandidateText = truncate(CandidateText)
    pairs.append((JOB_DESCRIPTION, CandidateText))
CeScores = CrossEncoder.predict(pairs, show_progress_bar=True)
CeMin = CeScores.min()
CeMAX = CeScores.max()
CeNormalized = (CeScores - CeMin) / (CeMAX - CeMin + 1e-9)
for i, result in enumerate(top_500):
    old_final = result['final']
    ce_score = float(CeNormalized[i])
    new_final = round((0.60 * ce_score) + (0.40 * old_final), 4)
    result['final'] = new_final
    result['ce_score'] = round(ce_score, 4)

top_500.sort(key=lambda x: (-x['final'], x['candidate_id']))
print("Cross-encoder re-ranking complete.")


def generate_reasoning(candidate, scores):
    profile = candidate['profile']
    signals = candidate['redrob_signals']
    title = profile.get('CurrentTitle', 'N/A')
    years = profile.get('years_of_experience', 0)
    notice = signals.get('notice_period_days', 'N/A')
    response = signals.get('recruiter_ResponseRate', 0)
    github = signals.get('github_activity_score', 'N/A')
    if github == -1:
        github = 'N/A'
    SkillsRequired = JD_FIELDS['SkillsRequired']
    TopSkills = [
        s['name'] for s in candidate['skills']
        if any(hv in s['name'].lower() or s['name'].lower() in hv
               for hv in SkillsRequired)
    ][:3]
    if not TopSkills:
        TopSkills = [s['name'] for s in candidate['skills']][:2]
    skills_text = ', '.join(TopSkills)
    reasoning = (
        f"{title} with {years} yrs experience; "
        f"top skills: {skills_text}; "
        f"notice period: {notice} days; "
        f"response rate: {round(response, 2)}; "
        f"github score: {github}."
    )
    return reasoning[:300]


with open('submission.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
    for rank, result in enumerate(top_500[:TOP_OUTPUT], 1):
        candidate = candidate_lookup[result['candidate_id']]
        reasoning = generate_reasoning(candidate, result)
        if result.get('honeypot'):
            reasoning = '[HONEYPOT DETECTED] ' + reasoning
        writer.writerow([
            result['candidate_id'],
            rank,
            result['final'],
            reasoning
        ])
print("submission.csv written successfully")
