from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CopyRequest(BaseModel):
    industry: str
    product: str
    target_audience: str
    objective: str
    offer: str = ""
    business_name: str = ""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate-copy")
def generate_copy(req: CopyRequest):
    prompt = f"""
You are an expert advertising copywriter specializing in small and medium businesses.

Generate 3 different Facebook/Instagram ad copy variations for the following brief:

Business: {req.business_name}
Industry: {req.industry}
Product/Service: {req.product}
Target Audience: {req.target_audience}
Campaign Objective: {req.objective}
Special Offer: {req.offer if req.offer else "None"}

Each variation must have:
- A hook (first line that stops the scroll, max 10 words)
- Body (2-3 sentences explaining the value)
- CTA (call to action, max 5 words)
- Angle (the emotional approach used)

Return ONLY a JSON array with exactly 3 objects. No extra text. Format:
[
  {{
    "angle": "angle name",
    "hook": "hook text",
    "body": "body text",
    "cta": "cta text"
  }}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert ad copywriter. Always return valid JSON only, no markdown, no extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.8,
        max_tokens=1000
    )

    content = response.choices[0].message.content.strip()
    
    # Clean any markdown if present
    content = content.replace("```json", "").replace("```", "").strip()
    
    variants = json.loads(content)
    
    return {"variants": variants}

class Campaign(BaseModel):
    name: str
    status: str
    spend: float = 0
    clicks: int = 0
    ctr: float = 0
    cpc: float = 0
    daily_budget: float = 0
    impressions: int = 0

class RecommendationRequest(BaseModel):
    campaigns: list[Campaign]
    industry: str
    target_cpl: float = 50

@app.post("/generate-recommendations")
def generate_recommendations(req: RecommendationRequest):
    campaigns_text = ""
    for i, c in enumerate(req.campaigns):
        campaigns_text += f"""
Campaign {i+1}: {c.name}
- Status: {c.status}
- Daily Budget: ${c.daily_budget}
- Total Spend: ${c.spend}
- Impressions: {c.impressions}
- Clicks: {c.clicks}
- CTR: {c.ctr}%
- CPC: ${c.cpc}
"""

    prompt = f"""
You are an expert media buyer analyzing Facebook ad campaigns for a {req.industry} business.
The business target CPL is ${req.target_cpl}.

Here are the current campaigns:
{campaigns_text}

Analyze these campaigns and generate 3 specific, actionable recommendations.
Each recommendation must be based on the actual data provided.

Return ONLY a JSON array with exactly 3 objects. No extra text. Format:
[
  {{
    "type": "warning|success|info",
    "title": "short action title",
    "reasoning": "one sentence explanation based on the data",
    "confidence": 85
  }}
]

Types:
- warning: something is wrong and needs fixing
- success: something is working well and should be scaled
- info: something to monitor or improve
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert media buyer. Always return valid JSON only, no markdown, no extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=800
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    recommendations = json.loads(content)

    return {"recommendations": recommendations}



class ChatMessage(BaseModel):
    message: str
    campaigns: list[Campaign]
    industry: str = "business"

@app.post("/chat")
def chat(req: ChatMessage):
    campaigns_context = ""
    for c in req.campaigns:
        campaigns_context += f"""
Campaign: {c.name}
- Status: {c.status}
- Daily Budget: ${c.daily_budget}
- Spend: ${c.spend}
- Impressions: {c.impressions}
- Clicks: {c.clicks}
- CTR: {c.ctr}%
- CPC: ${c.cpc}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""You are AdPilot, an expert AI media buyer helping a {req.industry} business owner manage their Facebook and Google ad campaigns.

You have access to their real campaign data. Answer their questions clearly and specifically based on this data. Be direct, use numbers, and always give actionable advice.

Current campaign data:
{campaigns_context}

Keep responses concise — 2-4 sentences max unless a detailed explanation is needed. Always end with one specific action they should take."""
            },
            {
                "role": "user",
                "content": req.message
            }
        ],
        temperature=0.7,
        max_tokens=500
    )

    return {"response": response.choices[0].message.content}


class AuditRequest(BaseModel):
    business_name: str
    industry: str = "business"
    facebook_page_url: str = ""
    instagram_url: str = ""
    tiktok_url: str = ""
    website_url: str = ""
    currently_running_ads: str = "unknown"
    monthly_budget: str = "unknown"
    pixel_installed: str = "unknown"
    current_cpl: str = "unknown"
    ads_experience: str = "unknown"
    main_goal: str = "unknown"

@app.post("/audit")
def generate_audit(req: AuditRequest):
    # Build presence summary
    presence = []
    if req.facebook_page_url:
        presence.append(f"Facebook: {req.facebook_page_url}")
    if req.instagram_url:
        presence.append(f"Instagram: {req.instagram_url}")
    if req.tiktok_url:
        presence.append(f"TikTok: {req.tiktok_url}")
    if req.website_url:
        presence.append(f"Website: {req.website_url}")

    presence_text = "\n".join(presence) if presence else "No links provided"

    prompt = f"""
You are an expert digital advertising auditor for businesses in Egypt and the Middle East.

Business Information:
- Business Name: {req.business_name}
- Industry: {req.industry}
- Currently running ads: {req.currently_running_ads}
- Monthly ad budget: {req.monthly_budget}
- Meta Pixel installed: {req.pixel_installed}
- Current CPL (cost per lead): {req.current_cpl}
- Advertising experience: {req.ads_experience}
- Main goal: {req.main_goal}

Online Presence:
{presence_text}

Based on this REAL data provided by the business, generate an accurate and specific advertising audit.
Be direct about what they are doing wrong based on their actual situation.
If they are not running ads, explain what they are missing.
If their CPL is high, explain why and what to fix.
If pixel is not installed, mark that as a critical issue.

Evaluate these 7 areas with scores based on the real data:
1. Platform Presence
2. Ad Account Setup
3. Campaign Structure
4. Audience Targeting
5. Ad Creative Quality
6. Budget Efficiency
7. Overall Strategy

Return ONLY a JSON object. No extra text. Format:
{{
  "overall_score": 65,
  "estimated_monthly_waste": "EGP 1,200",
  "missing_platforms": ["TikTok", "Google"],
  "grades": {{
    "platform_presence": {{ "score": 60, "label": "Needs work", "finding": "specific finding based on real data" }},
    "ad_setup": {{ "score": 60, "label": "Needs work", "finding": "specific finding based on real data" }},
    "campaign_structure": {{ "score": 70, "label": "Average", "finding": "specific finding based on real data" }},
    "audience_targeting": {{ "score": 65, "label": "Average", "finding": "specific finding based on real data" }},
    "ad_creative": {{ "score": 75, "label": "Good", "finding": "specific finding based on real data" }},
    "budget_efficiency": {{ "score": 55, "label": "Needs work", "finding": "specific finding based on real data" }},
    "overall_strategy": {{ "score": 65, "label": "Average", "finding": "specific finding based on real data" }}
  }},
  "top_issues": [
    "specific issue based on real data 1",
    "specific issue based on real data 2",
    "specific issue based on real data 3"
  ],
  "quick_wins": [
    "specific quick win based on real data 1",
    "specific quick win based on real data 2",
    "specific quick win based on real data 3"
  ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert digital advertising auditor for businesses in Egypt and the Middle East. Always return valid JSON only, no markdown, no extra text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=1000
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    audit = json.loads(content)

    return audit