import os
import re
import json
import sqlite3
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
CONFIG_PATH = project_root / "config.yaml"
DB_PATH = project_root / "data" / "change_regions.db"
RESULTS_DIR = project_root / "outputs" / "results"

TN_DISTRICTS = [
    "ariyalur", "chengalpattu", "chennai", "coimbatore", "cuddalore",
    "dharmapuri", "dindigul", "erode", "kallakurichi", "kanchipuram",
    "kanyakumari", "karur", "krishnagiri", "madurai", "mayiladuthurai",
    "nagapattinam", "namakkal", "nilgiris", "perambalur", "pudukkottai",
    "ramanathapuram", "ranipet", "salem", "sivaganga", "tenkasi",
    "thanjavur", "theni", "thoothukudi", "tiruchirappalli", "trichy", "tirunelveli",
    "tirupathur", "tiruppur", "tiruvallur", "tiruvannamalai", "tiruvarur",
    "vellore", "viluppuram", "virudhunagar"
]


def normalize_confidence(score_val: float, area_m2: float) -> float:
    """Clamp and normalize confidence percentage cleanly between 5.0% and 99.5%."""
    try:
        score_val = float(score_val)
        area_m2 = float(area_m2)
    except (ValueError, TypeError):
        return 75.0

    area_px = (area_m2 / 100.0) if area_m2 > 0 else 0.0
    if score_val > 1.0 and area_px > 0:
        derived_p = score_val / (area_px / 100.0)
        conf_pct = float(min(99.5, max(15.0, derived_p * 100.0)))
    elif 0.0 < score_val <= 1.0 and area_px > 0 and score_val <= (area_px / 100.0) * 1.05 and score_val <= 0.6 and area_px < 100:
        derived_p = score_val / (area_px / 100.0)
        if 0.5 <= derived_p <= 1.0:
            conf_pct = derived_p * 100.0
        else:
            conf_pct = score_val * 100.0
    elif 0.0 <= score_val <= 1.0:
        conf_pct = score_val * 100.0
    else:
        conf_pct = min(99.5, max(15.0, score_val))
    return round(float(min(99.5, max(5.0, conf_pct))), 1)


def is_conversational_query(text: str) -> Tuple[bool, str]:
    """Detect if the user query is a greeting, capability query, or pleasantry."""
    q = text.strip().lower()
    clean_q = re.sub(r'[!?,.]+', '', q).strip()

    greetings = {
        "hi", "hii", "hiii", "hello", "hey", "heyy", "hola", "vanakkam", "namaste",
        "good morning", "good afternoon", "good evening", "wassup", "sup", "yo", "howdy"
    }
    if clean_q in greetings or re.match(r'^(hi+|hey+|hello+|vanakkam|namaste)\b', clean_q):
        return True, "greeting"

    help_phrases = [
        "who are you", "what can you do", "help", "how do you work", "what is this",
        "what are you", "how to use", "what features", "commands", "guide me"
    ]
    if clean_q in help_phrases or clean_q.startswith("what can you do") or clean_q.startswith("who are you"):
        return True, "help"

    pleasantries = {
        "thanks", "thank you", "thank u", "thx", "ok", "okay", "got it", "cool",
        "nice", "great", "awesome", "bye", "goodbye", "see ya"
    }
    if clean_q in pleasantries:
        return True, "pleasantry"

    return False, "telemetry"


TELEMETRY_KEYWORDS = [
    "change", "difference", "stat", "area", "encroach", "hotspot", "loss",
    "hectare", "parcel", "acre", "sq km", "building", "structure", "road",
    "highway", "water", "lake", "reservoir", "river", "vegetation", "forest",
    "tree", "canopy", "bare", "soil", "convert", "transition", "shift",
    "sprawl", "expansion", "shrink", "flood", "inundat", "ndvi", "ndbi",
    "statewide", "overall", "total", "ranking"
]

def has_telemetry_intent(user_query: str, parsed: Dict[str, Any]) -> bool:
    """Check if query is asking for concrete GIS satellite change metrics."""
    if parsed.get("district") or parsed.get("t1_class") or parsed.get("t2_class"):
        return True
    q = user_query.lower()
    return any(k in q for k in TELEMETRY_KEYWORDS)


class GroqLLMClient:
    """
    Client for Groq LPU inference.
    Executes fast entity parsing, conversational greetings, and executive intelligence synthesis.
    """
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = "qwen/qwen3.8-27b"
        self.fallback_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        self.temperature = 0.2
        self.client = None

        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                    gcfg = cfg.get("groq", {})
                    if not self.api_key:
                        self.api_key = gcfg.get("api_key")
                    self.model = gcfg.get("model", self.model)
                    self.fallback_models = gcfg.get("fallback_models", self.fallback_models)
                    self.temperature = gcfg.get("temperature", self.temperature)
            except Exception as e:
                print(f"[GroqLLMClient] Error loading config: {e}")

        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"[GroqLLMClient] Failed to initialize Groq SDK: {e}")

    def is_available(self) -> bool:
        return self.client is not None

    def chat_conversational(self, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Friendly, conversational greeting or capability response without querying database."""
        system_prompt = (
            "You are the friendly, intelligent AI Assistant for the Tamil Nadu Satellite Change Detection & Remote Sensing System.\n"
            "You monitor bi-temporal Sentinel-2 satellite data (2020 vs 2025) across all 38 districts of Tamil Nadu.\n"
            "GUIDELINES:\n"
            "1. If the user greets you (e.g. 'hi', 'hello', 'hii'), greet them warmly (e.g., 'Vanakkam! 👋') and introduce yourself in 1-2 friendly sentences.\n"
            "2. Explain concisely that you analyze satellite transitions (Vegetation, Water, Buildings, Roads, Bare Land) and identify encroachment hotspots.\n"
            "3. If they ask who you are or what you can do, provide a clear, helpful overview.\n"
            "4. Provide 3 specific bullet query examples the user can ask.\n"
            "5. Do NOT invent database numbers; invite the user to pick a district or land-cover change.\n"
            "6. Keep the response natural, concise, and formatted in clean markdown."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for msg in chat_history[-2:]:
                if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"][:250]})
        messages.append({"role": "user", "content": user_query})

        if self.is_available():
            for target_model in [self.model] + self.fallback_models:
                try:
                    resp = self.client.chat.completions.create(
                        model=target_model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=300
                    )
                    content = resp.choices[0].message.content.strip()
                    if content:
                        return content
                except Exception as e:
                    print(f"[GroqLLMClient] Conversational chat failed with {target_model}: {e}")
                    continue

        return (
            "Vanakkam! 👋 Welcome to the **Tamil Nadu Satellite Difference Intelligence Bot**.\n\n"
            "I analyze bi-temporal Sentinel-2 satellite telemetry (2020 vs 2025) across all 38 districts of Tamil Nadu. "
            "I can calculate exact surface area transitions, encroachment hotspots, and route actionable directives to state authorities.\n\n"
            "**What would you like to investigate?**\n"
            "- *'the karur vegetation converted to building area and statistics'*\n"
            "- *'what are the water body loss and changes in chennai'*\n"
            "- *'tell me the area which is new building is come in the VEGETATION area'*"
        )

    def parse_query(self, user_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
        """Extract user intent, target district, and land-cover classes via Groq JSON mode."""
        if not self.is_available():
            return None

        system_prompt = (
            "You are an environmental GIS query parser for Tamil Nadu satellite change monitoring.\n"
            "Extract the user intent and entities into a JSON object with keys:\n"
            "- intent: one of 'greeting', 'help', 'pleasantry', 'telemetry'\n"
            "- district: Tamil Nadu district name in lowercase (e.g., 'karur', 'chennai', 'coimbatore', 'salem', 'madurai', 'tiruchirappalli'), or null if statewide/unspecified.\n"
            "- t1_class: original/source land-cover class ('vegetation', 'water', 'bare_land', 'road', 'building', or null).\n"
            "- t2_class: target/new land-cover class ('vegetation', 'water', 'bare_land', 'road', 'building', or null).\n"
            "- change_type: 'new', 'removed', 'expanded', or null.\n"
            "Respond ONLY with valid JSON."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for msg in chat_history[-3:]:
                if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                    content = msg["content"][:300]
                    messages.append({"role": msg["role"], "content": content})
        messages.append({"role": "user", "content": user_query})

        for target_model in [self.model] + self.fallback_models:
            try:
                resp = self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=250
                )
                raw_json = resp.choices[0].message.content
                parsed = json.loads(raw_json)

                # Validate district
                d = parsed.get("district")
                if d:
                    d = str(d).lower().strip()
                    if d == "trichy":
                        d = "tiruchirappalli"
                    if d not in TN_DISTRICTS:
                        d = None
                    parsed["district"] = d

                # Validate classes
                valid_classes = {"vegetation", "water", "bare_land", "road", "building"}
                t1 = parsed.get("t1_class")
                t2 = parsed.get("t2_class")
                parsed["t1_class"] = t1 if t1 in valid_classes else None
                parsed["t2_class"] = t2 if t2 in valid_classes else None
                parsed["used_model"] = target_model
                return parsed
            except Exception as e:
                print(f"[GroqLLMClient] Parse failed with model {target_model}: {e}")
                continue
        return None


    def synthesize_response(
        self,
        user_query: str,
        metrics: Dict[str, Any],
        district_breakdown: List[Dict[str, Any]],
        top_hotspots: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Optional[str]:
        """Synthesize a clear, concise, and normal satellite difference briefing."""
        if not self.is_available():
            return None

        system_prompt = (
            "You are a helpful and clear satellite difference intelligence assistant.\n"
            "Your job is to provide direct, clean, and normal information based strictly on the Sentinel-2 satellite telemetry facts provided.\n\n"
            "CRITICAL RULES:\n"
            "1. Give NORMAL, concise information answering the user's inquiry directly.\n"
            "2. In 2 to 4 clean sentences or simple bullet points, summarize:\n"
            "   - What changed (e.g. vegetation transitioned to buildings or bare land).\n"
            "   - The district/location, total surface area in hectares and sq km, and parcel count.\n"
            "   - The main hotspot location with coordinates and confidence.\n"
            "3. DO NOT generate bureaucratic threat assessments, legal policy orders, or government action directives.\n"
            "4. NEVER invent or alter any numbers or coordinates. Use ONLY the provided telemetry facts.\n"
            "5. Keep the tone natural, conversational, and easy to read."
        )

        telemetry_payload = {
            "scope": metrics.get("scope", "Tamil Nadu Statewide"),
            "total_parcels": metrics.get("total_parcels", 0),
            "total_surface_m2": metrics.get("total_area_m2", 0),
            "total_hectares": metrics.get("total_hectares", 0),
            "total_sq_km": metrics.get("total_sq_km", 0),
            "average_confidence": metrics.get("average_confidence", "N/A"),
            "top_districts": district_breakdown[:4],
            "top_georeferenced_hotspots": top_hotspots[:4]
        }

        user_content = (
            f"User Query: \"{user_query}\"\n\n"
            f"Grounded Telemetry Facts from Database:\n"
            f"{json.dumps(telemetry_payload, indent=2)}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for msg in chat_history[-2:]:
                if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"][:300]})
        messages.append({"role": "user", "content": user_content})

        for target_model in [self.model] + self.fallback_models:
            try:
                resp = self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=800
                )
                briefing = resp.choices[0].message.content.strip()
                if briefing:
                    return briefing
            except Exception as e:
                print(f"[GroqLLMClient] Synthesis failed with model {target_model}: {e}")
                continue
        return None


# Global singleton client
groq_llm = GroqLLMClient()


class InfoChatbotEngine:
    """
    Text-only intelligence bot for land-cover change and difference analysis.
    Powered by Groq LPU™ LLM with grounded SQLite telemetry and rule-based fallback.
    """

    @staticmethod
    def parse_query_regex(user_query: str) -> Dict[str, Any]:
        """Regex-based intent & class parser fallback."""
        q = user_query.strip().lower()

        # 1. Detect District
        detected_district = None
        for d in TN_DISTRICTS:
            if re.search(r'\b' + re.escape(d) + r'\b', q):
                detected_district = "tiruchirappalli" if d == "trichy" else d
                break

        # 2. Detect Transition Source (t1) and Target (t2)
        t1 = None
        t2 = None

        p1 = re.search(r'(?:new|built|constructed|fresh|added)?\s*(building|structures?|road|highway|water|lake|reservoir|vegetation|trees?|forest|bare\s*land|soil)\s*(?:is\s*come\s*in|come\s*in|appeared\s*in|in|into|on|over|replacing|encroaching\s*on)\s*(?:the\s*)?(vegetation|water|bare\s*land|forest|greenery|agriculture|farmland|soil|road|building)', q)
        if p1:
            target_raw = p1.group(1).replace(" ", "_")
            source_raw = p1.group(2).replace(" ", "_")

            target_map = {"structures": "building", "structure": "building", "highway": "road", "lake": "water", "reservoir": "water", "soil": "bare_land", "trees": "vegetation", "tree": "vegetation"}
            source_map = {"forest": "vegetation", "greenery": "vegetation", "agriculture": "vegetation", "farmland": "vegetation", "soil": "bare_land", "trees": "vegetation"}

            t2 = target_map.get(target_raw, target_raw)
            t1 = source_map.get(source_raw, source_raw)
        else:
            p2 = re.search(r'(vegetation|forest|greenery|water|lake|bare\s*land|soil|road|building)\s*(?:converted\s*to|transition\s*to|shifted\s*to|turned\s*into|to|into)\s*(?:new\s*)?(building|structures?|road|highway|water|vegetation|bare\s*land|soil)', q)
            if p2:
                source_raw = p2.group(1).replace(" ", "_")
                target_raw = p2.group(2).replace(" ", "_")
                source_map = {"forest": "vegetation", "greenery": "vegetation", "soil": "bare_land"}
                target_map = {"structures": "building", "highway": "road", "soil": "bare_land"}
                t1 = source_map.get(source_raw, source_raw)
                t2 = target_map.get(target_raw, target_raw)

        # Single class fallbacks
        if not t1 and not t2:
            if any(w in q for w in ["building", "construction", "built-up", "house", "urban"]):
                t2 = "building"
            elif any(w in q for w in ["road", "highway", "expressway"]):
                t2 = "road"
            elif any(w in q for w in ["water", "lake", "reservoir", "river"]):
                t2 = "water"
            elif any(w in q for w in ["vegetation", "forest", "tree", "canopy", "agriculture", "green"]):
                t2 = "vegetation"
            elif any(w in q for w in ["bare", "soil", "quarry", "cleared"]):
                t2 = "bare_land"

        change_type = None
        if "new" in q or "appeared" in q or "built" in q:
            change_type = "new"
        elif "lost" in q or "loss" in q or "removed" in q or "cleared" in q:
            change_type = "removed"
        elif "expand" in q or "widened" in q or "growth" in q:
            change_type = "expanded"

        return {
            "query": user_query,
            "district": detected_district,
            "t1_class": t1,
            "t2_class": t2,
            "change_type": change_type
        }

    @staticmethod
    def parse_query(user_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Hybrid parser: attempts Groq JSON extraction first, falls back to regex."""
        llm_parsed = groq_llm.parse_query(user_query, chat_history=chat_history)
        regex_parsed = InfoChatbotEngine.parse_query_regex(user_query)

        if llm_parsed:
            district = llm_parsed.get("district") or regex_parsed.get("district")
            t1 = llm_parsed.get("t1_class") or regex_parsed.get("t1_class")
            t2 = llm_parsed.get("t2_class") or regex_parsed.get("t2_class")
            ctype = llm_parsed.get("change_type") or regex_parsed.get("change_type")
            return {
                "query": user_query,
                "district": district,
                "t1_class": t1,
                "t2_class": t2,
                "change_type": ctype,
                "intent": llm_parsed.get("intent", "statistics"),
                "parser": "groq_llm"
            }

        regex_parsed["intent"] = "statistics"
        regex_parsed["parser"] = "regex_rule"
        return regex_parsed

    @staticmethod
    def answer_query(user_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process info query and return grounded multi-temporal intelligence or conversational responses."""
        is_conv, conv_intent = is_conversational_query(user_query)
        parsed = InfoChatbotEngine.parse_query(user_query, chat_history=chat_history)
        intent = parsed.get("intent", "telemetry")
        is_telemetry = has_telemetry_intent(user_query, parsed)

        # Handle natural conversational greetings, capabilities, or pleasantries without running heavy SQL dumps
        if is_conv or (not is_telemetry and intent in ["greeting", "help", "pleasantry"]):
            reply = groq_llm.chat_conversational(user_query, chat_history=chat_history)
            return {
                "status": "success",
                "query": user_query,
                "is_conversational": True,
                "intent": conv_intent if is_conv else intent,
                "llm_used": groq_llm.is_available(),
                "llm_model": groq_llm.model if groq_llm.is_available() else None,
                "ai_briefing": reply,
                "metrics": {
                    "total_parcels": 0,
                    "total_area_m2": 0.0,
                    "total_hectares": 0.0,
                    "total_sq_km": 0.0,
                    "average_confidence": "N/A",
                    "scope": "Conversational"
                },
                "summary_text": reply,
                "district_breakdown": [],
                "top_hotspots": [],
                "suggested_prompts": [
                    "the karur vegetation converted to building area and statistics",
                    "tell me the area which is new building is come in the VEGETATION area",
                    "what are the water body loss and changes in chennai",
                    "which areas in coimbatore had road network expansion"
                ]
            }

        t1 = parsed.get("t1_class")
        t2 = parsed.get("t2_class")
        dist = parsed.get("district")
        ctype = parsed.get("change_type")

        if not DB_PATH.exists():
            return {
                "status": "error",
                "message": "Database of detected change regions is not yet available."
            }

        conn = sqlite3.connect(str(DB_PATH))
        cur = conn.cursor()


        clauses = []
        params = []

        if t1:
            clauses.append("t1_class = ?")
            params.append(t1)
        if t2:
            clauses.append("(t2_class = ? OR semantic_class = ?)")
            params.extend([t2, t2])
        if dist:
            clauses.append("scene_id LIKE ?")
            params.append(f"%{dist.upper()}%")

        where_sql = " AND ".join(clauses) if clauses else "1=1"

        # 1. Aggregates
        cur.execute(f"""
            SELECT COUNT(*), COALESCE(SUM(area_m2), 0), COALESCE(AVG(change_score), 0)
            FROM change_regions WHERE {where_sql}
        """, params)
        total_count, total_area, avg_score = cur.fetchone()

        # If zero matches for exact t1->t2, fallback to t2 alone
        if total_count == 0 and t1 and t2:
            clauses_fallback = ["(t2_class = ? OR semantic_class = ?)"]
            params_fallback = [t2, t2]
            if dist:
                clauses_fallback.append("scene_id LIKE ?")
                params_fallback.append(f"%{dist.upper()}%")
            where_fallback = " AND ".join(clauses_fallback)
            cur.execute(f"""
                SELECT COUNT(*), COALESCE(SUM(area_m2), 0), COALESCE(AVG(change_score), 0)
                FROM change_regions WHERE {where_fallback}
            """, params_fallback)
            total_count, total_area, avg_score = cur.fetchone()
            if total_count > 0:
                where_sql = where_fallback
                params = params_fallback
                t1 = None

        # 2. District breakdown
        cur.execute(f"""
            SELECT 
                CASE 
                    WHEN INSTR(scene_id, '_') > 0 THEN SUBSTR(scene_id, 1, INSTR(scene_id, '_') - 1)
                    ELSE scene_id
                END as dname,
                COUNT(*),
                SUM(area_m2),
                AVG(centroid_lat),
                AVG(centroid_lon)
            FROM change_regions
            WHERE {where_sql}
            GROUP BY dname
            ORDER BY SUM(area_m2) DESC
            LIMIT 8
        """, params)
        dist_rows = cur.fetchall()

        # 3. Top hotspot parcels
        cur.execute(f"""
            SELECT scene_id, region_id, area_m2, centroid_lat, centroid_lon, change_score, t1_class, t2_class
            FROM change_regions
            WHERE {where_sql}
            ORDER BY area_m2 DESC
            LIMIT 5
        """, params)
        hotspot_rows = cur.fetchall()
        conn.close()

        ha = total_area / 10000.0
        km2 = total_area / 1000000.0

        dist_table_rows = []
        for d in dist_rows:
            d_name = d[0].capitalize()
            d_cnt = d[1]
            d_area = d[2]
            d_ha = d_area / 10000.0
            lat_str = f"{d[3]:.4f}° N" if d[3] else "N/A"
            lon_str = f"{d[4]:.4f}° E" if d[4] else "N/A"
            dist_table_rows.append({
                "district": d_name,
                "count": d_cnt,
                "area_m2": round(d_area, 1),
                "hectares": round(d_ha, 2),
                "center_coords": f"{lat_str}, {lon_str}"
            })

        hotspots_list = []
        for h in hotspot_rows:
            sc_id = h[0]
            r_id = h[1]
            h_area = h[2]
            h_lat = h[3]
            h_lon = h[4]
            h_score = h[5]
            h_t1 = h[6]
            h_t2 = h[7]
            norm_conf = normalize_confidence(h_score, h_area)
            hotspots_list.append({
                "scene": sc_id,
                "region_id": r_id,
                "area_m2": round(h_area, 1),
                "hectares": round(h_area / 10000.0, 2),
                "latitude": round(h_lat, 5) if h_lat else None,
                "longitude": round(h_lon, 5) if h_lon else None,
                "coordinates": f"{h_lat:.5f}° N, {h_lon:.5f}° E" if (h_lat and h_lon) else "N/A",
                "transition": f"{h_t1} → {h_t2}",
                "confidence": f"{norm_conf:.1f}%"
            })

        if hotspots_list:
            avg_conf_val = round(sum(float(h["confidence"].replace("%", "")) for h in hotspots_list) / len(hotspots_list), 1)
            avg_confidence_str = f"{avg_conf_val:.1f}%"
        else:
            avg_confidence_str = f"{normalize_confidence(avg_score, total_area / max(1, total_count)):.1f}%"

        # Baseline summary strings
        scope_str = f"in **{dist.capitalize()}** district" if dist else "across all monitored districts of **Tamil Nadu**"
        transition_str = ""
        if t1 and t2:
            transition_str = f"where prior **{t1.upper()}** land transitioned into **{t2.upper()}** (e.g., new construction/structures)"
        elif t2:
            transition_str = f"classified as **{t2.upper()}** change"
        else:
            transition_str = "across all physical surface changes"

        header_summary = (
            f"Based on bi-temporal Sentinel-2 satellite analysis (2020 vs 2025) {scope_str}, "
            f"we identified **{total_count:,} discrete change parcels** {transition_str}, "
            f"covering a combined surface area of **{total_area:,.1f} m²** ({ha:,.2f} hectares / {km2:,.3f} km²)."
        )

        metrics_dict = {
            "total_parcels": total_count,
            "total_area_m2": round(total_area, 1),
            "total_hectares": round(ha, 2),
            "total_sq_km": round(km2, 3),
            "average_confidence": avg_confidence_str,
            "scope": dist.capitalize() if dist else "Tamil Nadu Statewide"
        }

        # Attempt Groq LLM executive synthesis
        ai_briefing = groq_llm.synthesize_response(
            user_query=user_query,
            metrics=metrics_dict,
            district_breakdown=dist_table_rows,
            top_hotspots=hotspots_list,
            chat_history=chat_history
        )

        llm_used = ai_briefing is not None

        return {
            "status": "success",
            "query": user_query,
            "parsed": parsed,
            "llm_used": llm_used,
            "llm_model": groq_llm.model if llm_used else None,
            "ai_briefing": ai_briefing,
            "metrics": metrics_dict,
            "summary_text": header_summary,
            "district_breakdown": dist_table_rows,
            "top_hotspots": hotspots_list,
            "impact_analysis": ""
        }



class ImageChatbotEngine:
    """
    Visual satellite difference bot.
    Retrieves and displays exact Before (T1), After (T2), and Difference imagery for requested regions.
    """

    @staticmethod
    def answer_query(user_query: str, top_k: int = 6) -> Dict[str, Any]:
        """Find matching change areas and guarantee visual before/after/diff crop URLs."""
        q = user_query.strip().lower()

        # Detect district
        detected_district = None
        for d in TN_DISTRICTS:
            if re.search(r'\b' + re.escape(d) + r'\b', q):
                detected_district = "tiruchirappalli" if d == "trichy" else d
                break

        # Detect semantic class
        target_class = None
        if any(w in q for w in ["building", "construction", "built-up", "house", "urban", "structure"]):
            target_class = "building"
        elif any(w in q for w in ["road", "highway", "expressway", "street"]):
            target_class = "road"
        elif any(w in q for w in ["water", "lake", "reservoir", "river", "canal"]):
            target_class = "water"
        elif any(w in q for w in ["vegetation", "forest", "tree", "canopy", "agriculture", "crop"]):
            target_class = "vegetation"
        elif any(w in q for w in ["bare", "soil", "quarry", "cleared", "excavation"]):
            target_class = "bare_land"

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        clauses = []
        params = []
        if detected_district:
            clauses.append("scene_id LIKE ?")
            params.append(f"%{detected_district.upper()}%")
        if target_class:
            clauses.append("(semantic_class = ? OR t2_class = ?)")
            params.extend([target_class, target_class])

        where_sql = " AND ".join(clauses) if clauses else "1=1"

        cur.execute(f"""
            SELECT * FROM change_regions
            WHERE {where_sql}
            ORDER BY area_m2 DESC
            LIMIT {top_k * 3}
        """, params)
        candidates = cur.fetchall()

        # If empty and district was specified, relax class filter
        if len(candidates) == 0 and target_class and detected_district:
            cur.execute(f"""
                SELECT * FROM change_regions
                WHERE scene_id LIKE ?
                ORDER BY area_m2 DESC
                LIMIT {top_k * 3}
            """, [f"%{detected_district.upper()}%"])
            candidates = cur.fetchall()

        # If still empty, grab top statewide changes
        if len(candidates) == 0:
            cur.execute(f"""
                SELECT * FROM change_regions
                ORDER BY area_m2 DESC
                LIMIT {top_k * 3}
            """)
            candidates = cur.fetchall()

        results = []
        seen_keys = set()

        for row in candidates:
            r = dict(row)
            run_id = r.get("run_id", "")
            reg_id = r.get("region_id", 0)
            sc_id = r.get("scene_id", "")
            dist_name = sc_id.split("_")[0].capitalize() if "_" in sc_id else "Tamil Nadu"

            # Check / generate crops
            run_dir = RESULTS_DIR / run_id
            if not run_dir.exists():
                # Locate any existing run matching scene_id
                matching_runs = list(RESULTS_DIR.glob(f"*{dist_name.lower()}*"))
                if matching_runs:
                    run_dir = matching_runs[0]
                    run_id = run_dir.name

            crops_dir = run_dir / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)

            b_crop = crops_dir / f"region_{reg_id}_before.png"
            a_crop = crops_dir / f"region_{reg_id}_after.png"
            d_crop = crops_dir / f"region_{reg_id}_diff.png"

            # On-demand crop creation if missing
            if (not b_crop.exists() or not a_crop.exists()) and (run_dir / "before.png").exists() and (run_dir / "after.png").exists():
                try:
                    img_b = Image.open(run_dir / "before.png")
                    img_a = Image.open(run_dir / "after.png")
                    img_d = Image.open(run_dir / "change_map.png") if (run_dir / "change_map.png").exists() else None

                    x = int(r.get("bbox_x", 0))
                    y = int(r.get("bbox_y", 0))
                    w = int(r.get("bbox_width", 50))
                    h = int(r.get("bbox_height", 50))

                    pad = max(20, int(max(w, h) * 0.2))
                    box = (
                        max(0, x - pad),
                        max(0, y - pad),
                        min(img_b.width, x + w + pad),
                        min(img_b.height, y + h + pad)
                    )

                    img_b.crop(box).save(b_crop)
                    img_a.crop(box).save(a_crop)
                    if img_d:
                        img_d.crop(box).save(d_crop)
                except Exception:
                    pass

            # Fallback URLs if specific crop missing
            t1_url = f"/static/generated/{run_id}/crops/region_{reg_id}_before.png" if b_crop.exists() else f"/static/generated/{run_id}/before.png"
            t2_url = f"/static/generated/{run_id}/crops/region_{reg_id}_after.png" if a_crop.exists() else f"/static/generated/{run_id}/after.png"
            diff_url = f"/static/generated/{run_id}/crops/region_{reg_id}_diff.png" if d_crop.exists() else f"/static/generated/{run_id}/change_map.png"

            dedup = (sc_id, reg_id)
            if dedup in seen_keys:
                continue
            seen_keys.add(dedup)

            lat = r.get("centroid_lat")
            lon = r.get("centroid_lon")

            results.append({
                "region_id": reg_id,
                "scene_id": sc_id,
                "run_id": run_id,
                "district": dist_name,
                "semantic_class": r.get("semantic_class", "building"),
                "t1_class": r.get("t1_class", "unknown"),
                "t2_class": r.get("t2_class", "unknown"),
                "change_type": r.get("change_type", "transition"),
                "area_m2": round(float(r.get("area_m2", 0)), 1),
                "hectares": round(float(r.get("area_m2", 0)) / 10000.0, 2),
                "confidence_pct": round(float(r.get("change_score", 0.75)) * 100.0, 1),
                "latitude": round(lat, 5) if lat else None,
                "longitude": round(lon, 5) if lon else None,
                "coordinates_str": f"{lat:.5f}° N, {lon:.5f}° E" if (lat and lon) else "Georeferenced",
                "t1_url": t1_url,
                "t2_url": t2_url,
                "diff_url": diff_url,
                "google_maps_url": f"https://www.google.com/maps?q={lat},{lon}" if (lat and lon) else None
            })

            if len(results) >= top_k:
                break

        conn.close()

        dist_label = detected_district.capitalize() if detected_district else "Tamil Nadu"
        class_label = target_class.capitalize() if target_class else "Surface Change"
        intro_text = (
            f"Here are the satellite difference comparisons for **{dist_label}** ({class_label} areas). "
            f"Each card presents the **T₁ Before Image**, **T₂ After Image**, and the **Change Difference Mask** "
            f"with exact surface area metrics and GPS coordinates."
        )

        return {
            "status": "success",
            "query": user_query,
            "district": dist_label,
            "target_class": class_label,
            "message": intro_text,
            "total_found": len(results),
            "cards": results
        }
