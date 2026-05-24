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