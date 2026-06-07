from fastapi import FastAPI , HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import httpx
from bs4 import BeautifulSoup
import re
import chromadb
from chromadb.utils import embedding_functions

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

# ChromaDB setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Strategy RAG collection (shared across all businesses)
strategy_collection = chroma_client.get_or_create_collection(
    name="media_buying_strategies",
    embedding_function=openai_ef
)

def initialize_strategy_rag():
    existing = strategy_collection.count()
    if existing > 0:
        print(f"Strategy RAG already loaded: {existing} documents")
        return

    strategies = [
        {"id": "meta_001", "text": "Meta's Andromeda system (2025) fundamentally changed how ads are delivered. One ad now generates hundreds of variations automatically. The algorithm now does targeting — advertisers should focus on creative quality, not audience micro-segmentation. Broad targeting + strong creative outperforms narrow targeting + average creative.", "category": "platform", "industry": "all"},
        {"id": "meta_002", "text": "Advantage+ campaigns are now Meta's default in 2026. They use AI to automatically optimize targeting, placements, and creative. Best practice: use Advantage+ for scaling proven campaigns, use manual campaigns for testing new audiences and creative. Never use Advantage+ before you have a proven offer.", "category": "platform", "industry": "all"},
        {"id": "meta_003", "text": "Meta's learning phase in 2026: Some accounts now require only 10 conversions over 3 days to exit learning (down from 50 over 7 days). Never make significant edits during learning phase. Bundle all edits and make them at once to minimize learning resets.", "category": "platform", "industry": "all"},
        {"id": "meta_004", "text": "Advantage+ Creative (2026): All new Sales, Leads, and App Promotion campaigns launch with AI creative enhancements by default. Meta uses up to 5% of impressions to test enhancement combinations and scales winners automatically. Always provide multiple creative variations — minimum 5 different assets per campaign.", "category": "platform", "industry": "all"},
        {"id": "meta_005", "text": "AI-generated content disclosure (March 2026): Meta now requires disclosure on ads containing AI-generated or AI-modified content. Skipping this is now the most common reason for ad rejection. Always label AI-generated creatives properly.", "category": "platform", "industry": "all"},
        {"id": "str_001", "text": "Modern Meta campaign structure (2026): Consolidate ad sets. Instead of many small ad sets with low budgets, use fewer large ad sets with bigger budgets. Meta's algorithm needs volume to optimize. One campaign, 2-3 ad sets, 5-10 ads per ad set outperforms 10 campaigns with small budgets.", "category": "structure", "industry": "all"},
        {"id": "str_002", "text": "The Power of Five creative strategy for Advantage+ (2026): Every campaign needs 5 distinct ad types — (1) Problem/Solution Video, (2) High-quality photo, (3) User-Generated Content (UGC), (4) Educational carousel, (5) Testimonial or social proof.", "category": "structure", "industry": "all"},
        {"id": "str_003", "text": "Full funnel approach: Don't just target purchases or leads. Build campaigns for every stage — cold audience awareness, warm audience consideration, hot audience conversion, and customer retention. Each stage needs different creative and messaging.", "category": "structure", "industry": "all"},
        {"id": "str_004", "text": "Campaign Budget Optimization (CBO) vs Ad Set Budget (ABO): Use CBO when scaling above EGP 500/day — Meta allocates budget to best performing ad sets automatically. Use ABO when testing new audiences or creatives to ensure each gets enough budget for a fair test.", "category": "structure", "industry": "all"},
        {"id": "obj_001", "text": "For gyms and fitness centers in Egypt, Lead Generation campaigns consistently outperform Traffic and Awareness. Use OUTCOME_LEADS with instant forms. Expected CPL: EGP 30-80 Cairo, EGP 20-60 Alexandria. Always ask for name + phone in the form — not email.", "category": "objectives", "industry": "gym"},
        {"id": "obj_002", "text": "For clinics and healthcare in Egypt, Lead Generation with phone call optimization works best. Patients prefer calling before booking. Use call ads combined with instant forms. Ask 1-2 qualifying questions in the form. Expected CPL: EGP 40-100.", "category": "objectives", "industry": "clinic"},
        {"id": "obj_003", "text": "For restaurants in Egypt, Reach and Traffic objectives work better than leads. Focus on awareness and foot traffic. Video ads showing food preparation outperform photos. Boost posts with 1000+ organic reach first, then run paid traffic to those posts.", "category": "objectives", "industry": "restaurant"},
        {"id": "obj_004", "text": "For e-commerce in Egypt, Catalog Sales (Dynamic Ads) and Conversions outperform Traffic. Install Meta Pixel and set up purchase events before running conversion campaigns. Use Dynamic Product Ads for retargeting abandoned carts. Expected ROAS: 2x-5x.", "category": "objectives", "industry": "ecommerce"},
        {"id": "obj_005", "text": "For real estate in Egypt, Lead Generation with detailed instant forms works best. Ask for budget range and timeline to qualify leads. Use virtual tour videos instead of static photos. Expected CPL: EGP 100-300 for qualified leads.", "category": "objectives", "industry": "real_estate"},
        {"id": "obj_006", "text": "For beauty salons in Egypt, Lead Generation for new client acquisition and Reach for retention. Offer a first-visit discount. Before/after photos (with consent) outperform all other creative types. Expected CPL: EGP 25-70.", "category": "objectives", "industry": "salon"},
        {"id": "aud_001", "text": "2026 targeting philosophy: Broad targeting + great creative outperforms narrow targeting. Meta's AI is better at finding your customers than you are. Start broad, let the algorithm learn, then use the data to understand who's actually converting.", "category": "targeting", "industry": "all"},
        {"id": "aud_002", "text": "For Egyptian businesses, interest-based targeting still works for cold audiences. Stack 3-5 relevant interests. For gyms: fitness + health + weight loss + nutrition. Avoid broad interests like 'sports' — too wide. Narrow interests reduce waste.", "category": "targeting", "industry": "all"},
        {"id": "aud_003", "text": "Lookalike audiences in Egypt: Build from customer lists of 500+ or website visitors of 1000+ monthly. Use 1-2% lookalike for prospecting, 3-5% for scaling. Lookalikes from buyers outperform lookalikes from page fans by 3x.", "category": "targeting", "industry": "all"},
        {"id": "aud_004", "text": "Cairo audience segmentation: New Cairo and Maadi = higher purchasing power. Zamalek and Heliopolis = older and premium. 6th October and Giza = younger and price-sensitive. Segment by district for different ad messages and offers.", "category": "targeting", "industry": "all"},
        {"id": "aud_005", "text": "Retargeting sequence for Egypt: Day 1-3 website visitors see social proof ad. Day 4-7 see offer ad. Day 8-14 see urgency/scarcity ad. This 3-touch sequence improves conversion by 40%. Exclude converters from all retargeting audiences.", "category": "targeting", "industry": "all"},
        {"id": "aud_006", "text": "Sequential exclusions for advanced funnels: Exclude people who watched 50% of your video from seeing it again — show them the next video instead. Exclude page engagers from cold audience campaigns to avoid overlap and wasted spend.", "category": "targeting", "industry": "all"},
        {"id": "aud_007", "text": "Facebook penetration in Egypt is 89% as of 2026. Facebook remains the primary platform for family-oriented services and broad demographics. Instagram and TikTok dominate Gen Z and Millennials. WhatsApp is used for closing sales and customer support — not advertising.", "category": "targeting", "industry": "all"},
        {"id": "bud_001", "text": "Minimum viable budget for testing in Egypt: EGP 50/day per ad set. Below this, Meta's algorithm cannot exit learning phase. For campaigns with multiple ad sets, allocate at least EGP 150/day total.", "category": "budget", "industry": "all"},
        {"id": "bud_002", "text": "The 20% scaling rule: Never increase daily budget by more than 20% at a time or the campaign re-enters learning phase. Wait 3-4 days between increases. Scaling ladder example: EGP 100 → 120 → 145 → 175 → 210 → 250.", "category": "budget", "industry": "all"},
        {"id": "bud_003", "text": "Horizontal vs vertical scaling: For aggressive scaling, duplicate winning campaigns (horizontal scaling) rather than increasing budget on one campaign (vertical scaling). This avoids learning phase resets and maintains performance.", "category": "budget", "industry": "all"},
        {"id": "bud_004", "text": "Budget allocation framework: 70% to proven winning campaigns, 20% to testing new creatives and audiences, 10% to retargeting. Review and rebalance weekly based on CPL performance.", "category": "budget", "industry": "all"},
        {"id": "bud_005", "text": "During Ramadan in Egypt, CPL drops 20-30% due to increased social media time. Increase budgets 2 weeks before Ramadan. Best ad times: after iftar 8-11pm and suhour 2-4am. Ads with Ramadan-specific messaging outperform generic ads by 50%.", "category": "budget", "industry": "all"},
        {"id": "cre_001", "text": "Creative quality is the #1 performance variable in 2026, outweighing audience targeting, bidding, and placement. In Meta's Advantage+ system, the creative itself does the targeting. Focus 80% of effort on creative.", "category": "creative", "industry": "all"},
        {"id": "cre_002", "text": "Video ads outperform images in Egypt by 30-50% CTR. 15-30 second videos work best. Use captions — 80% of videos are watched without sound. Hook within first 3 seconds is critical.", "category": "creative", "industry": "all"},
        {"id": "cre_003", "text": "UGC (User Generated Content) is the highest performing ad format in 2026. Real customer reviews, testimonials, and before/after videos outperform polished professional ads by 2-3x.", "category": "creative", "industry": "all"},
        {"id": "cre_004", "text": "Arabic ads get 40% more engagement than English for Egyptian mass-market businesses. Exception: premium and luxury brands — bilingual ads maintain premium perception. Use Egyptian Arabic dialect not formal Arabic.", "category": "creative", "industry": "all"},
        {"id": "cre_005", "text": "Social proof is the strongest creative element for Egyptian businesses. Include: number of customers served, years in business, certifications, testimonials. Numbers build credibility instantly.", "category": "creative", "industry": "all"},
        {"id": "cre_006", "text": "For gym ads in Egypt, transformation photos (before/after) generate the lowest CPL. Real member results outperform stock photos by 3x. Show the process, not just the result.", "category": "creative", "industry": "gym"},
        {"id": "cre_007", "text": "Urgency and scarcity work well in Egypt: 'Limited spots available', 'Offer ends Friday', 'Only 5 places left this week'. Countdown-style language increases CTR by 20-30%.", "category": "creative", "industry": "all"},
        {"id": "cre_008", "text": "For clinic ads in Egypt, doctor credibility is the #1 conversion factor. Show doctor name, photo, credentials, years of experience. Patient testimonials are the second strongest element.", "category": "creative", "industry": "clinic"},
        {"id": "cre_009", "text": "Ad creative refresh cadence: Refresh creatives when frequency exceeds 2.5 in 7 days OR when CTR drops more than 30% from its peak. Top brands produce 10-20 new creative variations per month.", "category": "creative", "industry": "all"},
        {"id": "cre_010", "text": "Hook writing for Egyptian audience: Use questions, numbers, and local references. First 3 words must stop the scroll. Avoid starting with the business name — start with the customer's problem.", "category": "creative", "industry": "all"},
        {"id": "sea_001", "text": "Ramadan strategy for Egypt: Start 2 weeks before with awareness. First week: launch lead gen with Ramadan offer. Last 10 days: urgency campaign. After Eid Al-Fitr: reactivation campaign. CPL is 20-30% lower during Ramadan.", "category": "seasonal", "industry": "all"},
        {"id": "sea_002", "text": "Back to school Egypt (August-September): High demand for tutoring, fitness for students, uniforms, and supplies. Target parents of school-age children. Budget should increase 30% in August.", "category": "seasonal", "industry": "all"},
        {"id": "sea_003", "text": "Summer in Egypt (June-August): Gym memberships spike in June before summer. Clinics see increase in skin and aesthetic procedures. 'Summer body' and 'beach ready' messaging for fitness businesses peaks in May-June.", "category": "seasonal", "industry": "all"},
        {"id": "sea_004", "text": "Eid Al-Fitr Egypt: Launch gift and celebration campaigns 1 week before. Run gym renewal campaigns right after Eid — people feel guilty about eating and are motivated to restart.", "category": "seasonal", "industry": "all"},
        {"id": "sea_005", "text": "New Year (January) Egypt: Fitness, self-improvement, and education businesses see highest lead volume of the year. January is the best month to acquire gym members.", "category": "seasonal", "industry": "gym"},
        {"id": "opt_001", "text": "CPL diagnosis framework for Egypt: EGP 0-50 = excellent, scale immediately. EGP 50-100 = average, test new creative. EGP 100-150 = poor, test new audience and creative urgently. EGP 150+ = stop and rebuild.", "category": "optimization", "industry": "all"},
        {"id": "opt_002", "text": "CTR benchmarks for Egypt: Below 0.5% = change creative immediately. 0.5-1% = average, test new hooks. 1-2% = good. Above 2% = excellent, scale budget aggressively.", "category": "optimization", "industry": "all"},
        {"id": "opt_003", "text": "Frequency management: When frequency exceeds 3 in 7 days, CPL rises and CTR drops. Refresh creative or expand audience when frequency hits 2.5.", "category": "optimization", "industry": "all"},
        {"id": "opt_004", "text": "A/B testing priority order for Egyptian campaigns: 1) Offer, 2) Creative hook, 3) Audience, 4) Ad format. Test one element at a time. Wait 7 days minimum before declaring a winner.", "category": "optimization", "industry": "all"},
        {"id": "opt_005", "text": "Conversion rate optimization: If CTR is good (>1%) but CPL is high, the problem is the landing page or instant form — not the ad. If CTR is low (<0.5%), the problem is the creative or audience.", "category": "optimization", "industry": "all"},
        {"id": "opt_006", "text": "Lead quality scoring for Egyptian businesses: Track which campaigns bring leads that actually convert to paying customers. Optimize for cost per customer, not cost per lead.", "category": "optimization", "industry": "all"},
        {"id": "mis_001", "text": "Most common mistake by Egyptian SMEs: using Traffic objective when they want leads. Traffic optimizes for clicks, not conversions. Always match campaign objective to actual business goal.", "category": "mistakes", "industry": "all"},
        {"id": "mis_002", "text": "Second most common mistake: targeting all of Egypt with no interest filters. Always use interest targeting, location radius, or custom audiences to narrow focus.", "category": "mistakes", "industry": "all"},
        {"id": "mis_003", "text": "Third most common mistake: editing campaigns too early. Campaigns need 7-14 days minimum to exit learning phase. Premature editing resets learning and wastes budget.", "category": "mistakes", "industry": "all"},
        {"id": "mis_004", "text": "Not installing Meta Pixel is critical mistake #1. Without Pixel, you cannot run conversion campaigns, build website custom audiences, or measure true ROI.", "category": "mistakes", "industry": "all"},
        {"id": "mis_005", "text": "Running too many campaigns with small budgets. Five campaigns with EGP 20/day each perform far worse than one campaign with EGP 100/day. Consolidate campaigns and concentrate budget on winners.", "category": "mistakes", "industry": "all"},
        {"id": "mis_006", "text": "Using stock photos in Egyptian market. Real photos of the actual business, team, products, and customers outperform stock photos by 2-3x.", "category": "mistakes", "industry": "all"},
        {"id": "wa_001", "text": "WhatsApp is the #1 sales closing tool in Egypt. After getting a lead, the fastest path to conversion is a WhatsApp message within 5 minutes of form submission.", "category": "conversion", "industry": "all"},
        {"id": "wa_002", "text": "Click-to-WhatsApp ads in Egypt: For businesses that close sales via phone or chat, Click-to-WhatsApp ads often outperform standard lead forms. Best for: clinics, real estate, high-ticket services, restaurants.", "category": "conversion", "industry": "all"},
        {"id": "wa_003", "text": "Lead follow-up sequence Egypt: Message 1 (within 5 min): Welcome + confirm interest. Message 2 (same day): Share social proof or offer details. Message 3 (day 2): Address common objection. Message 4 (day 3): Create urgency.", "category": "conversion", "industry": "all"},
        {"id": "bench_001", "text": "Egypt gym/fitness benchmarks 2026: Good CPL EGP 30-60, Average EGP 60-100, Poor above EGP 100. CTR benchmark: 1.5-2.5%. Best performing audience: Women 25-40 Cairo and Giza.", "category": "benchmarks", "industry": "gym"},
        {"id": "bench_002", "text": "Egypt clinic/healthcare benchmarks 2026: Good CPL EGP 50-90, Average EGP 90-150, Poor above EGP 150. Phone call leads convert 3x better than form leads.", "category": "benchmarks", "industry": "clinic"},
        {"id": "bench_003", "text": "Egypt e-commerce benchmarks 2026: Good ROAS 3x+, Average 2-3x, Poor below 2x. Average CTR for product ads: 1-2%. Cart abandonment retargeting typically achieves 4-6x ROAS.", "category": "benchmarks", "industry": "ecommerce"},
        {"id": "bench_004", "text": "Egypt real estate benchmarks 2026: Good CPL EGP 150-250 for qualified leads, Average EGP 250-400, Poor above EGP 400. Video tours and 360 photos significantly reduce CPL.", "category": "benchmarks", "industry": "real_estate"},
        {"id": "bench_005", "text": "Egypt salon/beauty benchmarks 2026: Good CPL EGP 25-50, Average EGP 50-80, Poor above EGP 80. Before/after content has 3x lower CPL than promotional content.", "category": "benchmarks", "industry": "salon"},
    ]

    documents = [s["text"] for s in strategies]
    ids = [s["id"] for s in strategies]
    metadatas = [{"category": s["category"], "industry": s["industry"]} for s in strategies]

    strategy_collection.add(documents=documents, ids=ids, metadatas=metadatas)
    print(f"Strategy RAG initialized with {len(strategies)} documents")


initialize_strategy_rag()

def get_or_create_business_collection(business_id: str):
    collection_name = f"business_{business_id.replace('-', '_')}"
    return chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)


def build_business_knowledge(business_id: str, business_data: dict):
    collection = get_or_create_business_collection(business_id)
    documents = []
    ids = []
    metadatas = []

    if business_data.get("businessName") and business_data.get("description"):
        documents.append(f"Business: {business_data.get('businessName')}. Industry: {business_data.get('industry')}. City: {business_data.get('city')}. Description: {business_data.get('description')}")
        ids.append("identity")
        metadatas.append({"type": "identity"})

    if business_data.get("services"):
        documents.append(f"Services and products offered: {business_data.get('services')}")
        ids.append("services")
        metadatas.append({"type": "services"})

    if business_data.get("uniqueSellingPoint"):
        documents.append(f"What makes this business unique: {business_data.get('uniqueSellingPoint')}. Price range: {business_data.get('priceRange')}. Brand tone: {business_data.get('brandTone')}")
        ids.append("usp")
        metadatas.append({"type": "usp"})

    if business_data.get("targetAudience"):
        documents.append(f"Target audience: {business_data.get('targetAudience')}. Age range: {business_data.get('minAge')} to {business_data.get('maxAge')}. Gender: {business_data.get('gender')}. Customer source: {business_data.get('customerSource')}. Buying cycle: {business_data.get('buyingCycle')}. Average customer value: EGP {business_data.get('averageCustomerValue')}")
        ids.append("audience")
        metadatas.append({"type": "audience"})

    if business_data.get("mainGoal"):
        documents.append(f"Main advertising goal: {business_data.get('mainGoal')}. Monthly budget: {business_data.get('monthlyBudget')}. Target CPL: EGP {business_data.get('targetCpl')}. Biggest challenge: {business_data.get('biggestChallenge')}")
        ids.append("goals")
        metadatas.append({"type": "goals"})

    if business_data.get("competitors"):
        documents.append(f"Competitors: {business_data.get('competitors')}. What competitors do better: {business_data.get('competitorAdvantage')}. What this business does better: {business_data.get('ourAdvantage')}")
        ids.append("competition")
        metadatas.append({"type": "competition"})

    urls = []
    if business_data.get("websiteUrl"): urls.append(f"Website: {business_data.get('websiteUrl')}")
    if business_data.get("facebookPageUrl"): urls.append(f"Facebook: {business_data.get('facebookPageUrl')}")
    if business_data.get("instagramUrl"): urls.append(f"Instagram: {business_data.get('instagramUrl')}")
    if urls:
        documents.append("Online presence: " + ", ".join(urls))
        ids.append("online_presence")
        metadatas.append({"type": "online_presence"})

    if documents:
        try:
            existing_ids = collection.get()["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)
        except:
            pass
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
        print(f"Business RAG built for {business_id}: {len(documents)} documents")

    return len(documents)


def query_business_rag(business_id: str, query: str, n_results: int = 3) -> str:
    try:
        collection = get_or_create_business_collection(business_id)
        if collection.count() == 0:
            return ""
        results = collection.query(query_texts=[query], n_results=min(n_results, collection.count()))
        return "\n".join(results["documents"][0])
    except Exception as e:
        print(f"Business RAG query error: {e}")
        return ""


def query_strategy_rag(query: str, industry: str = "all", n_results: int = 5) -> str:
    try:
        where_filter = {"$or": [{"industry": industry}, {"industry": "all"}]} if industry != "all" else None
        results = strategy_collection.query(query_texts=[query], n_results=n_results, where=where_filter if where_filter else None)
        return "\n".join(results["documents"][0])
    except Exception as e:
        print(f"Strategy RAG query error: {e}")
        return ""


class BuildRagRequest(BaseModel):
    business_id: str
    business_data: dict

@app.post("/build-business-rag")
def build_rag(req: BuildRagRequest):
    count = build_business_knowledge(req.business_id, req.business_data)
    return {"status": "success", "documents": count}

@app.get("/query-rag")
def query_rag(business_id: str, query: str, industry: str = "all"):
    business_context = query_business_rag(business_id, query)
    strategy_context = query_strategy_rag(query, industry)
    return {"business_context": business_context, "strategy_context": strategy_context}


class CopyRequest(BaseModel):
    industry: str
    product: str
    target_audience: str
    objective: str
    offer: str = ""
    business_name: str = ""
    business_id: str = ""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate-copy")
def generate_copy(req: CopyRequest):
    business_context = ""
    strategy_context = ""
    if req.business_id:
        business_context = query_business_rag(req.business_id, f"ad copy for {req.product}")
        strategy_context = query_strategy_rag(f"ad copy creative strategy for {req.industry}", req.industry)

    prompt = f"""
You are an expert advertising copywriter specializing in Egyptian and Middle Eastern businesses.

Generate 3 different Facebook/Instagram ad copy variations for:
Business: {req.business_name}
Industry: {req.industry}
Product/Service: {req.product}
Target Audience: {req.target_audience}
Campaign Objective: {req.objective}
Special Offer: {req.offer if req.offer else "None"}

{f"Business context (use this to personalize): {business_context}" if business_context else ""}
{f"Creative strategy knowledge: {strategy_context}" if strategy_context else ""}

Each variation must have:
- hook (first line that stops the scroll, max 10 words)
- body (2-3 sentences explaining the value)
- cta (call to action, max 5 words)
- angle (the emotional approach used)

Return ONLY a JSON array with exactly 3 objects:
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
            {"role": "system", "content": "You are an expert ad copywriter for Egypt and MENA. Always return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=1000
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    variants = json.loads(content)
    return {"variants": variants}


# ── UPDATED MODELS WITH PLATFORM IDs ──

class Campaign(BaseModel):
    name: str
    status: str
    spend: float = 0
    clicks: int = 0
    ctr: float = 0
    cpc: float = 0
    daily_budget: float = 0
    impressions: int = 0
    platform_campaign_id: str = ""

class AdSetData(BaseModel):
    id: str = ""
    name: str
    status: str
    spend: float = 0
    clicks: int = 0
    ctr: float = 0
    cpc: float = 0
    impressions: int = 0
    optimization_goal: str = ""
    min_age: int = 0
    max_age: int = 0
    targeting: str = ""
    platform_ad_set_id: str = ""

class AdData(BaseModel):
    id: str = ""
    name: str
    status: str
    spend: float = 0
    clicks: int = 0
    ctr: float = 0
    cpc: float = 0
    impressions: int = 0
    creative_format: str = ""
    headline: str = ""
    body: str = ""
    platform_ad_id: str = ""

class CampaignWithDetails(BaseModel):
    name: str
    status: str
    spend: float = 0
    clicks: int = 0
    ctr: float = 0
    cpc: float = 0
    daily_budget: float = 0
    impressions: int = 0
    platform_campaign_id: str = ""
    ad_sets: list[AdSetData] = []

class RecommendationRequest(BaseModel):
    campaigns: list[CampaignWithDetails]
    industry: str
    target_cpl: float = 50
    business_id: str = ""

@app.post("/generate-recommendations")
def generate_recommendations(req: RecommendationRequest):
    full_context = ""
    for i, c in enumerate(req.campaigns):
        full_context += f"""
CAMPAIGN {i+1}: {c.name}
- Platform ID: {c.platform_campaign_id}
- Status: {c.status}
- Daily Budget: EGP {c.daily_budget}
- Total Spend: EGP {c.spend}
- Impressions: {c.impressions}
- Clicks: {c.clicks}
- CTR: {c.ctr}%
- CPC: EGP {c.cpc}
"""
        for j, adset in enumerate(c.ad_sets):
            full_context += f"""
  AD SET {j+1}: {adset.name}
  - Platform ID: {adset.platform_ad_set_id}
  - Status: {adset.status}
  - Spend: EGP {adset.spend}
  - Impressions: {adset.impressions}
  - Clicks: {adset.clicks}
  - CTR: {adset.ctr}%
  - CPC: EGP {adset.cpc}
  - Optimization: {adset.optimization_goal}
  - Age range: {adset.min_age}-{adset.max_age}
  - Targeting: {adset.targeting[:200] if adset.targeting else 'N/A'}
"""

    business_context = ""
    strategy_context = ""
    if req.business_id:
        business_context = query_business_rag(req.business_id, "campaign optimization recommendations")
        strategy_context = query_strategy_rag(f"campaign optimization for {req.industry}", req.industry)

    prompt = f"""
You are an expert media buyer analyzing Facebook ad campaigns for a {req.industry} business in Egypt and the Middle East.
All monetary values are in Egyptian Pounds (EGP). Never use $ or USD — always use EGP.
The business target CPL is EGP {req.target_cpl}.

Egypt benchmarks:
- Good CTR: above 1.5% — Average: 0.5-1.5% — Poor: below 0.5%
- Good CPC: below EGP 10 — Average: EGP 10-30 — Poor: above EGP 30
- Good CPL: below EGP 50 — Average: EGP 50-100 — Poor: above EGP 100
- Minimum daily budget: EGP 50/day to exit learning phase

{f"Business context: {business_context}" if business_context else ""}
{f"Expert Egypt/MENA strategy: {strategy_context}" if strategy_context else ""}

Full campaign data including platform IDs:
{full_context}

Generate 5 specific actionable recommendations covering ALL levels:
- At least 1 about overall campaign strategy
- At least 1 about ad set targeting or budget
- At least 1 about specific ad performance
- Use actual names, EGP numbers, and percentages from the data
- Compare against Egypt benchmarks
- For each recommendation, specify the exact action using the platform IDs

Action types available:
- PAUSE_CAMPAIGN — pause underperforming campaign
- RESUME_CAMPAIGN — resume paused campaign
- PAUSE_ADSET — pause underperforming ad set
- RESUME_ADSET — resume paused ad set
- PAUSE_AD — pause underperforming ad
- RESUME_AD — resume paused ad
- INCREASE_BUDGET — increase daily budget (newValue = suggested EGP amount)
- DECREASE_BUDGET — decrease daily budget (newValue = suggested EGP amount)
- NONE — no direct action possible

Return ONLY a JSON array with exactly 5 objects:
[
  {{
    "type": "warning|success|info",
    "level": "campaign|adset|ad",
    "title": "short action title",
    "reasoning": "specific explanation using actual names and EGP values",
    "confidence": 85,
    "action": {{
      "type": "PAUSE_CAMPAIGN|RESUME_CAMPAIGN|PAUSE_ADSET|RESUME_ADSET|PAUSE_AD|RESUME_AD|INCREASE_BUDGET|DECREASE_BUDGET|NONE",
      "entityId": "exact platform ID from the data above",
      "level": "campaign|adset|ad",
      "newValue": null
    }}
  }}
]

Action rules:
- ALWAYS use the exact platform ID from the data
- Pause recommendations → use PAUSE_CAMPAIGN/PAUSE_ADSET/PAUSE_AD with the correct ID
- Scale recommendations → use INCREASE_BUDGET with newValue as suggested daily budget in EGP
- No clear action → use NONE with entityId as ""
- newValue is a number for budget actions, null for status actions
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert media buyer for Egypt and MENA. All monetary values must be in EGP. Always return valid JSON only with exact platform IDs from the provided data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1500
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    recommendations = json.loads(content)
    return {"recommendations": recommendations}


class ChatMessage(BaseModel):
    message: str
    campaigns: list[Campaign]
    industry: str = "business"
    business_id: str = ""

@app.post("/chat")
def chat(req: ChatMessage):
    campaigns_context = ""
    for c in req.campaigns:
        campaigns_context += f"""
Campaign: {c.name}
- Status: {c.status}
- Daily Budget: EGP {c.daily_budget}
- Spend: EGP {c.spend}
- Impressions: {c.impressions}
- Clicks: {c.clicks}
- CTR: {c.ctr}%
- CPC: EGP {c.cpc}
"""

    business_context = ""
    strategy_context = ""
    if req.business_id:
        business_context = query_business_rag(req.business_id, req.message)
        strategy_context = query_strategy_rag(req.message, req.industry)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""You are AdPilot, an expert AI media buyer helping a {req.industry} business owner in Egypt manage their ad campaigns.

You have access to their real campaign data AND deep knowledge of their business AND expert media buying strategies for Egypt and MENA.

{f"Business knowledge: {business_context}" if business_context else ""}
{f"Expert strategy knowledge: {strategy_context}" if strategy_context else ""}

Current campaign data:
{campaigns_context}

Be direct, specific, and actionable. Use their actual business context when answering.
Keep responses concise — 2-4 sentences max unless detailed explanation is needed.
Always end with one specific action they should take.
All monetary values in EGP."""
            },
            {"role": "user", "content": req.message}
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
    scanned_description: str = ""
    scanned_services: str = ""
    scanned_target_audience: str = ""
    unique_selling_point: str = ""
    average_price: str = ""
    conversion_rate: str = ""
    monthly_customers_from_ads: str = ""
    customer_retention: str = ""
    monthly_revenue: str = ""
    revenue_from_ads_pct: str = ""


def normalize_budget(budget_str):
    return budget_str.replace('–', '-').replace('—', '-').strip()


@app.post("/audit")
def generate_audit(req: AuditRequest):
    presence = []
    if req.facebook_page_url: presence.append(f"Facebook: {req.facebook_page_url}")
    if req.instagram_url: presence.append(f"Instagram: {req.instagram_url}")
    if req.tiktok_url: presence.append(f"TikTok: {req.tiktok_url}")
    if req.website_url: presence.append(f"Website: {req.website_url}")
    presence_text = "\n".join(presence) if presence else "No links provided"

    scanned_info = ""
    if req.scanned_description: scanned_info += f"\nScanned business description: {req.scanned_description}"
    if req.scanned_services: scanned_info += f"\nScanned services/products: {req.scanned_services}"
    if req.scanned_target_audience: scanned_info += f"\nScanned target audience: {req.scanned_target_audience}"
    if req.unique_selling_point: scanned_info += f"\nUnique selling point: {req.unique_selling_point}"

    financial_analysis = ""
    if req.average_price: financial_analysis += f"\n- Average product/service price: EGP {req.average_price}"
    if req.monthly_revenue: financial_analysis += f"\n- Monthly total revenue: EGP {req.monthly_revenue}"
    if req.conversion_rate: financial_analysis += f"\n- Lead to customer conversion rate: {req.conversion_rate}"
    if req.monthly_customers_from_ads: financial_analysis += f"\n- Monthly new customers from ads: {req.monthly_customers_from_ads}"
    if req.customer_retention: financial_analysis += f"\n- Customer retention/frequency: {req.customer_retention}"
    if req.revenue_from_ads_pct: financial_analysis += f"\n- Revenue from ads: {req.revenue_from_ads_pct}"

    budget_midpoints = {
        "Under EGP 3,000": 1500, "EGP 3,000-10,000": 6500, "EGP 10,000-30,000": 20000,
        "EGP 30,000-100,000": 65000, "EGP 100,000-300,000": 200000, "Over EGP 300,000": 350000, "Not running ads": 0,
    }

    roas_analysis = ""
    try:
        if req.monthly_revenue and req.monthly_budget and req.revenue_from_ads_pct and req.revenue_from_ads_pct != "I don't know":
            revenue = float(req.monthly_revenue)
            ad_spend = budget_midpoints.get(normalize_budget(req.monthly_budget), 0)
            pct_map = {"Less than 20%": 0.1, "20-40%": 0.3, "40-60%": 0.5, "60-80%": 0.7, "Over 80%": 0.85}
            normalized_pct = normalize_budget(req.revenue_from_ads_pct)
            pct = pct_map.get(normalized_pct, 0.5)
            revenue_from_ads = revenue * pct
            if ad_spend > 0:
                roas = revenue_from_ads / ad_spend
                roas_label = "POOR — ads are losing money" if roas < 2 else "AVERAGE — needs improvement" if roas < 4 else "GOOD — consider scaling"
                roas_analysis = f"\nCalculated ROAS: {roas:.1f}x (Revenue from ads: EGP {revenue_from_ads:.0f} / Ad spend: EGP {ad_spend:.0f}) — {roas_label}"

        if req.average_price and req.monthly_budget and req.monthly_customers_from_ads:
            avg_price = float(req.average_price)
            ad_spend = budget_midpoints.get(normalize_budget(req.monthly_budget), 0)
            customers_map = {"0": 0, "1-5": 3, "5-20": 12, "20-50": 35, "50-100": 75, "Over 100": 120}
            customers = customers_map.get(normalize_budget(req.monthly_customers_from_ads), 0)
            if customers > 0 and ad_spend > 0:
                cost_per_customer = ad_spend / customers
                roi = ((avg_price - cost_per_customer) / cost_per_customer) * 100
                roas_analysis += f"\nCost per customer: EGP {cost_per_customer:.0f} vs average price EGP {avg_price:.0f} — ROI: {roi:.0f}%"
                if cost_per_customer > avg_price:
                    roas_analysis += " — LOSING MONEY on every customer"
                elif cost_per_customer > avg_price * 0.3:
                    roas_analysis += " — High acquisition cost, needs optimization"
                else:
                    roas_analysis += " — Healthy acquisition cost"
    except Exception as e:
        print(f"ROAS calculation error: {e}")

    budget = budget_midpoints.get(normalize_budget(req.monthly_budget), 0)
    waste_pct = 0
    if req.pixel_installed == "No": waste_pct += 35
    elif req.pixel_installed == "I don't know": waste_pct += 20
    if req.current_cpl == "Over EGP 1,000": waste_pct += 40
    elif req.current_cpl in ["EGP 300-1,000", "EGP 300–1,000"]: waste_pct += 25
    elif req.current_cpl in ["EGP 100-300", "EGP 100–300"]: waste_pct += 10
    elif req.current_cpl == "I don't track this": waste_pct += 30
    if req.ads_experience in ["Never", "Less than 3 months"]: waste_pct += 20
    if normalize_budget(req.monthly_budget) == "Under EGP 3,000": waste_pct += 15
    if req.conversion_rate == "Less than 5%": waste_pct += 25
    elif req.conversion_rate == "I don't know": waste_pct += 15
    has_facebook = bool(req.facebook_page_url)
    has_instagram = bool(req.instagram_url)
    if not has_facebook and not has_instagram: waste_pct += 20
    elif not has_facebook or not has_instagram: waste_pct += 10
    waste_pct = min(waste_pct, 80)
    estimated_waste = round((budget * waste_pct / 100) / 100) * 100
    waste_str = f"EGP {estimated_waste:,}" if estimated_waste > 0 else "EGP 0 (not running ads)"

    prompt = f"""
You are an expert digital advertising auditor for businesses in Egypt and the Middle East.

Business Information:
- Business Name: {req.business_name}
- Industry: {req.industry}
- Currently running ads: {req.currently_running_ads}
- Monthly ad budget: {req.monthly_budget}
- Meta Pixel installed: {req.pixel_installed}
- Current CPL: {req.current_cpl}
- Advertising experience: {req.ads_experience}
- Main goal: {req.main_goal}

Online Presence:
{presence_text}
{scanned_info if scanned_info else ""}

Financial Data:
{financial_analysis if financial_analysis else "Not provided"}
{roas_analysis if roas_analysis else ""}

Generate an accurate and specific advertising audit. Return ONLY a JSON object:
{{
  "overall_score": 65,
  "estimated_monthly_waste": "PLACEHOLDER",
  "missing_platforms": ["TikTok", "Google"],
  "grades": {{
    "platform_presence": {{ "score": 60, "label": "Needs work", "finding": "specific finding" }},
    "ad_setup": {{ "score": 60, "label": "Needs work", "finding": "specific finding" }},
    "campaign_structure": {{ "score": 70, "label": "Average", "finding": "specific finding" }},
    "audience_targeting": {{ "score": 65, "label": "Average", "finding": "specific finding" }},
    "ad_creative": {{ "score": 75, "label": "Good", "finding": "specific finding" }},
    "budget_efficiency": {{ "score": 55, "label": "Needs work", "finding": "specific finding with ROAS data" }},
    "overall_strategy": {{ "score": 65, "label": "Average", "finding": "specific finding" }}
  }},
  "top_issues": ["specific issue with EGP numbers", "specific issue", "specific issue"],
  "quick_wins": ["specific quick win with expected EGP impact", "specific quick win", "specific quick win"]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert digital advertising auditor for businesses in Egypt and the Middle East. Always return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1200
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    audit = json.loads(content)
    audit["estimated_monthly_waste"] = waste_str
    return audit


class ScanRequest(BaseModel):
    website_url: str = ""
    facebook_url: str = ""
    instagram_url: str = ""

@app.post("/scan-business")
async def scan_business(req: ScanRequest):
    scraped_text = ""

    if req.website_url:
        try:
            url = req.website_url
            if not url.startswith("http"): url = "https://" + url
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http_client:
                res = await http_client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(res.text, "html.parser")
                title = soup.find("title")
                if title: scraped_text += f"Website title: {title.text.strip()}\n"
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc: scraped_text += f"Meta description: {meta_desc.get('content', '')}\n"
                for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
                body_text = soup.get_text(separator=" ", strip=True)
                body_text = re.sub(r'\s+', ' ', body_text)[:3000]
                scraped_text += f"Website content: {body_text}\n"
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if "facebook.com" in href and not req.facebook_url: scraped_text += f"Found Facebook: {href}\n"
                    if "instagram.com" in href and not req.instagram_url: scraped_text += f"Found Instagram: {href}\n"
                    if "tiktok.com" in href: scraped_text += f"Found TikTok: {href}\n"
        except Exception as e:
            scraped_text += f"Website scan failed: {str(e)}\n"

    if req.facebook_url:
        try:
            url = req.facebook_url
            if not url.startswith("http"): url = "https://" + url
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http_client:
                res = await http_client.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
                soup = BeautifulSoup(res.text, "html.parser")
                title = soup.find("title")
                if title: scraped_text += f"Facebook page title: {title.text.strip()}\n"
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc: scraped_text += f"Facebook description: {meta_desc.get('content', '')}\n"
                og_desc = soup.find("meta", attrs={"property": "og:description"})
                if og_desc: scraped_text += f"Facebook og description: {og_desc.get('content', '')}\n"
        except Exception as e:
            scraped_text += f"Facebook page: {req.facebook_url}\n"

    if req.instagram_url:
        try:
            url = req.instagram_url
            if not url.startswith("http"): url = "https://" + url
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as http_client:
                res = await http_client.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
                soup = BeautifulSoup(res.text, "html.parser")
                og_title = soup.find("meta", attrs={"property": "og:title"})
                if og_title: scraped_text += f"Instagram name: {og_title.get('content', '')}\n"
                og_desc = soup.find("meta", attrs={"property": "og:description"})
                if og_desc: scraped_text += f"Instagram bio: {og_desc.get('content', '')}\n"
        except Exception as e:
            scraped_text += f"Instagram profile: {req.instagram_url}\n"

    if not scraped_text or len(scraped_text) < 100:
        return {
            "business_name": "", "industry": "business", "city": "", "description": "",
            "services": [], "unique_selling_point": "", "price_range": "mid-range",
            "brand_tone": "professional", "target_audience": "", "min_age": 25, "max_age": 45,
            "gender": "all", "facebook_url": req.facebook_url, "instagram_url": req.instagram_url,
            "tiktok_url": "", "phone_number": "", "scan_failed": True,
            "scan_message": "We couldn't scan this website automatically. Please fill in your business details manually."
        }

    prompt = f"""
You are an expert business analyst. Analyze the following website and social media content and extract structured business information.

{scraped_text}

Return ONLY a JSON object:
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
            {"role": "system", "content": "You are a business analyst. Extract structured data from website content. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=800
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


class UpdateRagRequest(BaseModel):
    business_id: str
    learning: str
    learning_id: str
    learning_type: str = "campaign_performance"

@app.post("/update-business-rag")
def update_business_rag(req: UpdateRagRequest):
    try:
        collection = get_or_create_business_collection(req.business_id)
        try:
            collection.delete(ids=[req.learning_id])
        except:
            pass
        collection.add(documents=[req.learning], ids=[req.learning_id], metadatas=[{"type": req.learning_type}])
        return {"status": "success", "learning_id": req.learning_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


class AnalyzeCampaignsRequest(BaseModel):
    business_id: str
    campaigns: list[Campaign]
    industry: str = "business"

@app.post("/analyze-campaign-learnings")
def analyze_campaign_learnings(req: AnalyzeCampaignsRequest):
    if not req.campaigns:
        return {"status": "no_campaigns"}

    campaigns_text = ""
    for c in req.campaigns:
        campaigns_text += f"""
Campaign: {c.name}
- Status: {c.status}
- Daily Budget: EGP {c.daily_budget}
- Spend: EGP {c.spend}
- Impressions: {c.impressions}
- Clicks: {c.clicks}
- CTR: {c.ctr}%
- CPC: EGP {c.cpc}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert media buyer. Extract concise campaign performance learnings. Return JSON only."},
            {"role": "user", "content": f"""Analyze these campaigns for a {req.industry} business in Egypt and extract key learnings.

{campaigns_text}

Return ONLY a JSON array of 2-4 learning strings. Each learning should be 1-2 sentences.
Example: ["Campaign X has CTR of 2.1% which is above benchmark — this audience and creative combination is working well."]
No extra text."""}
        ],
        temperature=0.3,
        max_tokens=500
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    learnings = json.loads(content)

    collection = get_or_create_business_collection(req.business_id)
    import time
    timestamp = str(int(time.time()))
    for i, learning in enumerate(learnings):
        try:
            collection.add(documents=[learning], ids=[f"campaign_learning_{timestamp}_{i}"], metadatas=[{"type": "campaign_performance"}])
        except:
            pass

    return {"status": "success", "learnings": learnings, "count": len(learnings)}


class CreateCampaignRequest(BaseModel):
    business_id: str
    industry: str = "business"
    goal: str
    target_audience: str
    daily_budget: float
    duration_days: int = 30
    offer: str = ""
    city: str = ""

@app.post("/generate-campaign-strategy")
def generate_campaign_strategy(req: CreateCampaignRequest):
    business_context = query_business_rag(req.business_id, f"campaign strategy for {req.goal}")
    strategy_context = query_strategy_rag(f"campaign creation {req.goal} {req.industry} Egypt", req.industry)

    prompt = f"""
You are an expert Meta media buyer creating a campaign for a {req.industry} business in Egypt.

Business context: {business_context if business_context else "No business context available"}
Expert strategy knowledge: {strategy_context if strategy_context else "No strategy context available"}

Campaign brief:
- Goal: {req.goal}
- Target audience: {req.target_audience}
- Daily budget: EGP {req.daily_budget}
- Duration: {req.duration_days} days
- Offer: {req.offer if req.offer else "No specific offer"}
- City: {req.city if req.city else "Egypt"}

Return ONLY a JSON object:
{{
  "campaign_name": "descriptive campaign name",
  "objective": "one of: OUTCOME_LEADS, OUTCOME_SALES, OUTCOME_TRAFFIC, OUTCOME_AWARENESS, OUTCOME_ENGAGEMENT",
  "ad_set_name": "descriptive ad set name",
  "optimization_goal": "one of: LEAD_GENERATION, OFFSITE_CONVERSIONS, LINK_CLICKS, REACH, IMPRESSIONS",
  "targeting": {{
    "age_min": 25,
    "age_max": 45,
    "genders": [1, 2],
    "geo_locations": {{
      "cities": [
        {{"country": "EG", "name": "Cairo", "region": "Cairo Governorate", "radius": 25, "distance_unit": "mile"}}
      ]
    }},
    "flexible_spec": [
      {{
        "interests": [
          {{"id": "6003139266461", "name": "relevant interest 1"}},
          {{"id": "6003397425735", "name": "relevant interest 2"}}
        ]
      }}
    ]
  }},
  "daily_budget": {req.daily_budget},
  "ad_copy": {{
    "headline": "compelling headline max 40 chars",
    "body": "engaging ad body 2-3 sentences",
    "cta": "one of: LEARN_MORE, SIGN_UP, GET_QUOTE, CONTACT_US, BOOK_NOW"
  }},
  "strategy_reasoning": "2-3 sentences explaining why this strategy will work",
  "estimated_cpl": "estimated cost per lead in EGP",
  "estimated_reach": "estimated weekly reach"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert Meta media buyer for Egypt and MENA. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=1500
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


class GenerateImageRequest(BaseModel):
    prompt: str
    business_id: str = ""

@app.post("/generate-image")
async def generate_image(req: GenerateImageRequest):
    try:
        business_context = ""
        if req.business_id:
            business_context = query_business_rag(req.business_id, "ad creative visual style brand")

        enhanced_prompt = f"""Professional Facebook/Instagram advertisement image.
{req.prompt}
Style: Clean, modern, high quality, suitable for social media advertising.
No text overlays. Photorealistic. Well-lit. Professional composition."""
        if business_context:
            enhanced_prompt += f"\nBrand context: {business_context[:200]}"

        response = client.images.generate(model="dall-e-3", prompt=enhanced_prompt, size="1024x1024", quality="standard", n=1)
        return {"image_url": response.data[0].url, "revised_prompt": response.data[0].revised_prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


class CopyVariantsRequest(BaseModel):
    business_id: str = ""
    industry: str = "business"
    goal: str = ""
    target_audience: str = ""
    offer: str = ""

@app.post("/generate-copy-variants")
def generate_copy_variants(req: CopyVariantsRequest):
    business_context = ""
    strategy_context = ""
    if req.business_id:
        business_context = query_business_rag(req.business_id, "ad copy headlines body text")
        strategy_context = query_strategy_rag(f"ad copy for {req.industry} Egypt", req.industry)

    prompt = f"""
You are an expert Facebook ads copywriter for Egypt and MENA market.
Industry: {req.industry}
Goal: {req.goal}
Target audience: {req.target_audience}
Offer: {req.offer if req.offer else "None"}

{f"Business context: {business_context}" if business_context else ""}
{f"Expert strategy: {strategy_context}" if strategy_context else ""}

Generate 3 headline variants and 3 body text variants.
- Headlines: max 40 characters, punchy and attention-grabbing
- Bodies: 2-3 sentences, benefit-focused, Egypt market context
- All in English, use EGP for prices
- Each variant distinctly different in angle

Return ONLY a JSON object:
{{
  "headlines": ["Headline 1", "Headline 2", "Headline 3"],
  "bodies": ["Body 1 text.", "Body 2 text.", "Body 3 text."]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert Facebook ads copywriter for Egypt and MENA. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=800
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


class CampaignSuggestionsRequest(BaseModel):
    business_id: str = ""
    industry: str = "business"
    existing_campaigns: list = []
    total_spend: float = 0
    avg_ctr: float = 0

@app.post("/generate-campaign-suggestions")
def generate_campaign_suggestions(req: CampaignSuggestionsRequest):
    business_context = ""
    strategy_context = ""
    if req.business_id:
        business_context = query_business_rag(req.business_id, "campaign ideas growth opportunities")
        strategy_context = query_strategy_rag(f"campaign ideas for {req.industry} Egypt", req.industry)

    existing = ", ".join([c.get("name", "") for c in req.existing_campaigns]) if req.existing_campaigns else "None"

    from datetime import datetime
    current_date = datetime.now().strftime("%B %Y")

    prompt = f"""
You are an expert Meta media buyer for Egypt and MENA.
Today's date: {current_date}
Business industry: {req.industry}
Existing campaigns: {existing}
Total spend so far: EGP {req.total_spend}
Average CTR: {req.avg_ctr}%

{f"Business context: {business_context}" if business_context else ""}
{f"Expert strategy knowledge: {strategy_context}" if strategy_context else ""}

Generate 4 specific campaign suggestions this business should run next.
Consider:
- Gaps in their current campaigns (retargeting, awareness, conversion)
- UPCOMING Egypt seasonal opportunities based on today's date ({current_date}) — only suggest seasons and events that have NOT happened yet
- Industry best practices for {req.industry} in Egypt
- Budget efficiency — suggest campaigns that complement existing ones
- Current month context: what are Egyptians focused on right now in {current_date}?

Today is {current_date} — only suggest what makes sense NOW or in the coming weeks.

Return ONLY a JSON array with exactly 4 objects:
[
  {{
    "title": "Campaign title",
    "type": "retargeting|awareness|conversion|seasonal|engagement",
    "description": "One sentence explaining why this campaign will work",
    "goal": "Get more leads|Increase sales|Build brand awareness|Get more website traffic",
    "suggested_budget": 100,
    "estimated_cpl": "EGP 30-50",
    "urgency": "high|medium|low",
    "reason": "Why now — specific reason based on today being {current_date}"
  }}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are an expert media buyer for Egypt and MENA. Today is {current_date}. Only suggest seasonal campaigns for events that have NOT yet occurred. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1200
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    return {"suggestions": json.loads(content)}


class CompetitorSpyRequest(BaseModel):
    competitor_name: str
    industry: str = "business"
    business_id: str = ""
    country: str = "EG"

@app.post("/competitor-spy")
async def competitor_spy(req: CompetitorSpyRequest):
    app_access_token = os.getenv("META_APP_ACCESS_TOKEN")
    ads = []

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                "https://graph.facebook.com/v19.0/ads_archive",
                params={
                    "access_token": app_access_token,
                    "ad_reached_countries": req.country,
                    "search_terms": req.competitor_name,
                    "ad_active_status": "ACTIVE",
                    "fields": "id,ad_creation_time,ad_creative_body,ad_creative_link_title,ad_snapshot_url,page_name,page_id,impressions,spend",
                    "limit": 20
                },
                timeout=15.0
            )
            data = response.json()
            print(f"Ad Library response: {data}")
            ads = data.get("data", [])
    except Exception as e:
        print(f"Meta Ad Library error: {e}")

    if not ads:
        analysis = {
            "strategy_summary": f"No active ads found for {req.competitor_name} in Egypt. They may not be running Facebook ads currently.",
            "ad_frequency": "0 active ads",
            "main_message": "Unknown",
            "target_audience": "Unknown",
            "insights": [{"type": "opportunity", "title": "Gap in competitor advertising", "detail": f"{req.competitor_name} is not running active Facebook ads — this is your opportunity to dominate the space."}],
            "recommended_response": "This is your chance to capture market share while your competitor is not advertising."
        }
        return {"competitor": req.competitor_name, "ads": [], "analysis": analysis}

    ads_summary = ""
    for i, ad in enumerate(ads[:10]):
        ads_summary += f"""
Ad {i+1}:
- Page: {ad.get('page_name', 'Unknown')}
- Created: {ad.get('ad_creation_time', 'Unknown')}
- Headline: {ad.get('ad_creative_link_title', 'N/A')}
- Body: {str(ad.get('ad_creative_body', 'N/A'))[:200]}
- Snapshot: {ad.get('ad_snapshot_url', 'N/A')}
"""

    strategy_context = query_strategy_rag(f"competitor analysis {req.industry} Egypt", req.industry)

    prompt = f"""
You are an expert media buyer analyzing competitor Facebook ads for a {req.industry} business in Egypt.
Competitor: {req.competitor_name}
Number of active ads: {len(ads)}
Their ads: {ads_summary}
Strategy context: {strategy_context if strategy_context else ""}

Return ONLY a JSON object:
{{
    "strategy_summary": "2-3 sentence summary of their overall ad strategy",
    "ad_frequency": "description of how many ads and how often",
    "main_message": "their main value proposition",
    "target_audience": "who they appear to be targeting",
    "insights": [
        {{"type": "opportunity|threat|observation", "title": "insight title", "detail": "actionable detail in EGP context"}}
    ],
    "recommended_response": "What you should do differently to compete"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert media buyer for Egypt and MENA. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=1000
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    return {"competitor": req.competitor_name, "total_ads": len(ads), "ads": ads[:10], "analysis": json.loads(content)}


@app.post("/seed-egypt-calendar")
def seed_egypt_calendar():
    egypt_calendar = """
EGYPT MARKETING SEASONAL CALENDAR 2026-2027
============================================

ISLAMIC HOLIDAYS (approximate dates, shift ~11 days earlier each year):
- Ramadan 2026: ~February 18 - March 19, 2026
- Eid Al-Fitr 2026: ~March 20-23, 2026 (3 days)
- Eid Al-Adha 2026: ~May 27 - June 1, 2026 (4 days)
- Islamic New Year 2026: ~June 26, 2026
- Prophet's Birthday (Mawlid) 2026: ~September 4, 2026
- Ramadan 2027: ~February 7 - March 8, 2027
- Eid Al-Fitr 2027: ~March 9-12, 2027
- Eid Al-Adha 2027: ~May 17-20, 2027

EGYPTIAN NATIONAL HOLIDAYS:
- New Year's Day: January 1
- Coptic Christmas: January 7
- Revolution Day (Jan 25): January 25
- Sinai Liberation Day: April 25
- Labour Day: May 1
- Revolution Day (June 30): June 30
- Armed Forces Day: October 6
- Suez Day: October 24
- National Day: July 23

MONTHLY MARKETING CONTEXT FOR EGYPT:

JANUARY: New Year campaigns, Coptic Christmas (Jan 7), gyms peak, post-holiday detox. Best for: Gyms, health, beauty, home.
FEBRUARY: Valentine's Day (Feb 14), Pre-Ramadan preparation. Best for: Restaurants, jewellery, gifts, beauty.
MARCH: Ramadan 2026 (Feb 18-Mar 19), iftar/suhoor campaigns, night shopping peaks 8pm-2am. Best for: Food, fashion, electronics.
APRIL: Eid Al-Fitr (~Mar 20), post-Eid slowdown, Sham El-Nessim. Best for: Fashion, gifts, travel.
MAY: Pre-summer, Eid Al-Adha approaching (~May 27). Best for: Travel, fashion, home, beauty.
JUNE: Eid Al-Adha (~May 27-Jun 1), schools finish, summer begins, Islamic New Year (~Jun 26). Best for: Travel, fashion, food.
JULY: Peak summer, North Coast/Ain Sokhna travel, back to school prep starts. Best for: Travel, beach, fashion, electronics.
AUGUST: Back to school peaks, uniforms/supplies/electronics. Best for: Electronics, fashion, school supplies.
SEPTEMBER: Schools start, Mawlid (~Sep 4). Best for: Education, tutoring, fashion, health.
OCTOBER: Armed Forces Day (Oct 6), fall/winter fashion, pre-holiday season. Best for: Fashion, electronics, home.
NOVEMBER: Black Friday (last Friday), winter campaigns. Best for: Electronics, fashion, home appliances, e-commerce.
DECEMBER: Coptic Christmas prep, New Year, end of year sales. Best for: Gifts, fashion, restaurants, travel.

INDUSTRY-SPECIFIC PEAKS:
GYMS: January (New Year), Post-Ramadan (Apr/May), Pre-summer (May/Jun). Low: Ramadan.
CLINICS: Back to school (Aug/Sep), Post-Ramadan, Winter flu season (Oct-Dec).
RESTAURANTS: Ramadan (huge), Eid periods, Summer delivery.
REAL ESTATE: Post-Eid, Summer (before school year).
E-COMMERCE: Black Friday, Ramadan nights, Back to school.
SALONS: Pre-Eid, Pre-wedding season (Apr-Jun), Pre-summer (May).

KEY CONSUMER BEHAVIOR:
- Price-sensitive — discounts work extremely well
- Family-oriented purchasing
- Mobile-first — 95%+ use mobile
- Evening peak — best 8pm-11pm (Ramadan: 10pm-2am)
- Arabic content 40% more engagement for mass market
- WhatsApp primary sales closing channel
- Installment payments (taqseet) very popular
"""

    try:
        strategy_collection_local = chroma_client.get_or_create_collection(name="adpilot_strategy_rag", metadata={"description": "Media buying strategies for Egypt and MENA"})
        strategy_collection_local.upsert(documents=[egypt_calendar], ids=["egypt_seasonal_calendar_2026_2027"], metadatas=[{"type": "seasonal_calendar", "region": "Egypt", "language": "English", "year": "2026-2027"}])
        return {"status": "success", "message": "Egypt seasonal calendar added to Strategy RAG"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to seed calendar: {str(e)}")