from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import httpx
from bs4 import BeautifulSoup
import re

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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



class ScanRequest(BaseModel):
    website_url: str = ""
    facebook_url: str = ""
    instagram_url: str = ""

@app.post("/scan-business")
async def scan_business(req: ScanRequest):
    scraped_text = ""

    # Scrape website
    if req.website_url:
        try:
            url = req.website_url
            if not url.startswith("http"):
                url = "https://" + url
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http_client:
                res = await http_client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(res.text, "html.parser")

                # Extract title
                title = soup.find("title")
                if title:
                    scraped_text += f"Website title: {title.text.strip()}\n"

                # Extract meta description
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc:
                    scraped_text += f"Meta description: {meta_desc.get('content', '')}\n"

                # Extract all text
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                body_text = soup.get_text(separator=" ", strip=True)
                body_text = re.sub(r'\s+', ' ', body_text)[:3000]
                scraped_text += f"Website content: {body_text}\n"

                print(f"Website scraped: {len(scraped_text)} chars")
                print(f"First 500 chars: {scraped_text[:500]}")

                # Find social links
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if "facebook.com" in href and not req.facebook_url:
                        scraped_text += f"Found Facebook: {href}\n"
                    if "instagram.com" in href and not req.instagram_url:
                        scraped_text += f"Found Instagram: {href}\n"
                    if "tiktok.com" in href:
                        scraped_text += f"Found TikTok: {href}\n"

        except Exception as e:
            scraped_text += f"Website scan failed: {str(e)}\n"
            print(f"Scraping error: {str(e)}")

    # Scrape Facebook page
    if req.facebook_url:
        try:
            url = req.facebook_url
            if not url.startswith("http"):
                url = "https://" + url
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http_client:
                res = await http_client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                soup = BeautifulSoup(res.text, "html.parser")

                title = soup.find("title")
                if title:
                    scraped_text += f"Facebook page title: {title.text.strip()}\n"

                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc:
                    scraped_text += f"Facebook description: {meta_desc.get('content', '')}\n"

                og_desc = soup.find("meta", attrs={"property": "og:description"})
                if og_desc:
                    scraped_text += f"Facebook og description: {og_desc.get('content', '')}\n"

                for tag in soup(["script", "style"]):
                    tag.decompose()
                fb_text = soup.get_text(separator=" ", strip=True)
                fb_text = re.sub(r'\s+', ' ', fb_text)[:1000]
                if fb_text:
                    scraped_text += f"Facebook page content: {fb_text}\n"

                print(f"Facebook scraped: {len(scraped_text)} chars")
        except Exception as e:
            scraped_text += f"Facebook page: {req.facebook_url}\n"
            print(f"Facebook scraping failed: {str(e)}")

    # Scrape Instagram profile
    if req.instagram_url:
        try:
            url = req.instagram_url
            if not url.startswith("http"):
                url = "https://" + url
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http_client:
                res = await http_client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                })
                soup = BeautifulSoup(res.text, "html.parser")

                og_title = soup.find("meta", attrs={"property": "og:title"})
                if og_title:
                    scraped_text += f"Instagram name: {og_title.get('content', '')}\n"

                og_desc = soup.find("meta", attrs={"property": "og:description"})
                if og_desc:
                    scraped_text += f"Instagram bio: {og_desc.get('content', '')}\n"

                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc:
                    scraped_text += f"Instagram description: {meta_desc.get('content', '')}\n"

                print(f"Instagram scraped: {len(scraped_text)} chars")
        except Exception as e:
            scraped_text += f"Instagram profile: {req.instagram_url}\n"
            print(f"Instagram scraping failed: {str(e)}")

    if not scraped_text or len(scraped_text) < 100:
        return {
            "business_name": "",
            "industry": "business",
            "city": "",
            "description": "",
            "services": [],
            "unique_selling_point": "",
            "price_range": "mid-range",
            "brand_tone": "professional",
            "target_audience": "",
            "min_age": 25,
            "max_age": 45,
            "gender": "all",
            "facebook_url": req.facebook_url,
            "instagram_url": req.instagram_url,
            "tiktok_url": "",
            "phone_number": "",
            "scan_failed": True,
            "scan_message": "We couldn't scan this website automatically. Please fill in your business details manually."
        }

    # Send to GPT-4o for analysis
    prompt = f"""
You are an expert business analyst. Analyze the following website and social media content and extract structured business information.

{scraped_text}

Return ONLY a JSON object with no extra text:
{{
  "business_name": "extracted business name",
  "industry": "one of: gym, clinic, restaurant, salon, ecommerce, real_estate, automotive, business",
  "city": "city name in Egypt or Middle East if found, otherwise empty",
  "description": "2-3 sentence description of the business",
  "services": ["service 1", "service 2", "service 3"],
  "unique_selling_point": "what makes this business different",
  "price_range": "one of: budget, mid-range, premium, luxury",
  "brand_tone": "one of: professional, friendly, urgent, luxury, energetic",
  "target_audience": "who their customers are",
  "min_age": 25,
  "max_age": 45,
  "gender": "one of: male, female, all",
  "facebook_url": "facebook url if found",
  "instagram_url": "instagram url if found",
  "tiktok_url": "tiktok url if found",
  "phone_number": "phone number if found"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a business analyst. Extract structured data from website and social media content. Return valid JSON only. Never make up data that is not in the content provided."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=800
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    result = json.loads(content)
    return result