import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

DB_PATH = Path("data/change_regions.db")

class EnvironmentalAuditGenerator:
    """
    Generates official government-grade environmental audit reports,
    encroachment threat triage, and field inspection checklists
    from georeferenced satellite change detection records.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)

    def generate_district_audit(self, district: str = "all") -> Dict[str, Any]:
        """
        Generate comprehensive environmental audit briefing for a given district.
        """
        district_clean = district.lower().strip()
        query = "SELECT * FROM change_regions WHERE centroid_lat IS NOT NULL"
        params = []

        if district_clean != "all" and district_clean != "":
            query += " AND LOWER(scene_id) LIKE ?"
            params.append(f"{district_clean}%")

        query += " ORDER BY area_m2 DESC"

        records = []
        if self.db_path.exists():
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                rows = cursor.execute(query, params).fetchall()
                records = [dict(r) for r in rows]

        # Aggregate metrics
        total_regions = len(records)
        total_changed_m2 = sum(float(r.get("area_m2") or 0.0) for r in records)
        total_changed_hectares = round(total_changed_m2 / 10000.0, 2)

        # Class breakdown
        class_breakdown = {
            "water": {"area_m2": 0.0, "count": 0},
            "vegetation": {"area_m2": 0.0, "count": 0},
            "building": {"area_m2": 0.0, "count": 0},
            "road": {"area_m2": 0.0, "count": 0},
            "bare_land": {"area_m2": 0.0, "count": 0},
            "unknown": {"area_m2": 0.0, "count": 0}
        }

        # Threat classification buckets
        critical_encroachments = []
        high_ecological_alerts = []
        moderate_alerts = []
        standard_changes = []

        for r in records:
            cls = str(r.get("semantic_class", "unknown")).lower()
            chg_type = str(r.get("change_type", "unknown")).lower()
            t1_cls = str(r.get("t1_class", "unknown")).lower()
            t2_cls = str(r.get("t2_class", "unknown")).lower()
            area = float(r.get("area_m2") or 0.0)
            score = float(r.get("change_score") or 0.0)
            lat = float(r.get("centroid_lat") or 0.0)
            lon = float(r.get("centroid_lon") or 0.0)
            run_id = r.get("run_id", "")
            reg_id = r.get("region_id", 1)

            if cls in class_breakdown:
                class_breakdown[cls]["area_m2"] += area
                class_breakdown[cls]["count"] += 1
            else:
                class_breakdown["unknown"]["area_m2"] += area
                class_breakdown["unknown"]["count"] += 1

            crop_url = f"/static/generated/{run_id}/crops/region_{reg_id}_crop.png"

            item = {
                "id": r.get("id"),
                "region_id": reg_id,
                "scene_id": r.get("scene_id"),
                "run_id": run_id,
                "district": r.get("scene_id", "").split("_")[0].capitalize(),
                "semantic_class": cls,
                "change_type": chg_type,
                "t1_class": t1_cls,
                "t2_class": t2_cls,
                "area_m2": round(area, 1),
                "area_hectares": round(area / 10000.0, 3),
                "change_score": score,
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "crop_url": crop_url,
                "deep_link": f"/region/{run_id}/{reg_id}"
            }

            # 1. CRITICAL THREATS
            # 1a. Lakebed / Water Encroachment (Water body converted to building, bare land, or road)
            if t1_cls == "water" and t2_cls in ["building", "bare_land", "road"]:
                item["threat_level"] = "CRITICAL"
                item["threat_badge"] = "🔴 Critical Encroachment"
                item["threat_reason"] = "Unauthorized encroachment or physical land reclamation over water body / lakebed."
                item["department"] = "Water Resources Department (TWAD) & Revenue Division"
                critical_encroachments.append(item)

            # 1b. Severe Urban Flooding / Built-up Inundation (Buildings/roads submerged by water)
            elif t1_cls in ["building", "road"] and t2_cls == "water":
                item["threat_level"] = "CRITICAL"
                item["threat_badge"] = "🔴 Inundation Alert"
                item["threat_reason"] = "Severe urban flood inundation or critical infrastructure submergence."
                item["department"] = "State Disaster Management Authority (TNSDMA) & Municipal Corp"
                critical_encroachments.append(item)

            # 1c. Water body lost to siltation / weed overgrowth or drying
            elif t1_cls == "water" and chg_type in ["removed", "transition"] and t2_cls != "water":
                item["threat_level"] = "CRITICAL"
                item["threat_badge"] = "🔴 Water Body Loss"
                item["threat_reason"] = "Substantial water body shrinkage, lakebed drying, or severe weed siltation."
                item["department"] = "Water Resources Department (TWAD) & Environment Department"
                critical_encroachments.append(item)

            # 2. HIGH THREATS
            # 2a. Forest / Tree Canopy / Agricultural Clearance to Built-Up or Roads
            elif t1_cls == "vegetation" and t2_cls in ["building", "road"]:
                item["threat_level"] = "HIGH"
                item["threat_badge"] = "🟠 Construction on Green Cover"
                item["threat_reason"] = "Conversion of vegetative cover to built-up development or unauthorized roads."
                item["department"] = "Tamil Nadu Forest Department & Directorate of Town and Country Planning"
                high_ecological_alerts.append(item)

            # 2b. High Deforestation / Vegetation Loss to Bare Land
            elif (t1_cls == "vegetation" and t2_cls == "bare_land") or (t1_cls == "vegetation" and chg_type == "removed"):
                if area >= 2000:
                    item["threat_level"] = "HIGH"
                    item["threat_badge"] = "🟠 High Deforestation"
                    item["threat_reason"] = "Significant tree canopy or agricultural cover clearance."
                    item["department"] = "Tamil Nadu Forest Department & Agriculture Division"
                    high_ecological_alerts.append(item)
                else:
                    item["threat_level"] = "MODERATE"
                    item["threat_badge"] = "🟡 Vegetative Clearing"
                    item["threat_reason"] = "Localized vegetative clearing or seasonal crop harvesting."
                    item["department"] = "Tamil Nadu Forest Department & Agriculture Division"
                    moderate_alerts.append(item)

            # 3. MODERATE THREATS & WATCH ITEMS
            # 3a. Water Inflow / Reservoir Replenishment / Inundation
            elif t2_cls == "water" and t1_cls in ["bare_land", "vegetation", "unknown"]:
                item["threat_level"] = "MODERATE"
                item["threat_badge"] = "🟡 Inundation / Water Inflow"
                item["threat_reason"] = "Surface water accumulation, reservoir replenishment, or seasonal flood inflow."
                item["department"] = "Water Resources Department (TWAD) & State Ground Water Authority"
                moderate_alerts.append(item)

            # 3b. Urban Sprawl / Construction on Bare Land
            elif t2_cls in ["building", "road"] and t1_cls in ["bare_land", "unknown"]:
                item["threat_level"] = "MODERATE"
                item["threat_badge"] = "🟡 Rapid Urban Sprawl"
                item["threat_reason"] = "New structural or road infrastructure expansion requiring zoning audit."
                item["department"] = "CMDA / Directorate of Town and Country Planning"
                moderate_alerts.append(item)

            # 3c. Large ground restructuring / earthwork
            elif area >= 5000:
                item["threat_level"] = "MODERATE"
                item["threat_badge"] = "🟡 Moderate Sprawl"
                item["threat_reason"] = "Substantial physical ground restructuring requiring zoning compliance audit."
                item["department"] = "CMDA / Directorate of Town and Country Planning"
                moderate_alerts.append(item)

            # 4. STANDARD: Normal changes (greening, minor bare land, etc.)
            else:
                item["threat_level"] = "NORMAL"
                if t2_cls == "vegetation" and t1_cls in ["bare_land", "unknown"]:
                    item["threat_badge"] = "🟢 Green Cover Gain"
                    item["threat_reason"] = "Vegetation expansion, agricultural regrowth, or afforestation."
                    item["department"] = "Tamil Nadu Forest Department & Agriculture Division"
                else:
                    item["threat_badge"] = "🟢 Standard Change"
                    item["threat_reason"] = "Routine land-surface modification."
                    item["department"] = "District Land Survey Department"
                standard_changes.append(item)

        # Ranked Priority Hotspots (top 10 by area and score)
        priority_hotspots = critical_encroachments[:4] + high_ecological_alerts[:3] + moderate_alerts[:3]
        if len(priority_hotspots) < 10:
            remaining = [r for r in (critical_encroachments + high_ecological_alerts + moderate_alerts + standard_changes) if r not in priority_hotspots]
            priority_hotspots.extend(remaining[:10 - len(priority_hotspots)])

        for idx, h in enumerate(priority_hotspots):
            h["priority_rank"] = idx + 1

        # Field Checklist for Ground Officers
        checklist = []
        for idx, h in enumerate(priority_hotspots):
            checklist.append({
                "item_no": idx + 1,
                "region_id": h["region_id"],
                "district": h["district"],
                "target_location": f"{h['lat']}° N, {h['lon']}° E",
                "area_m2": f"{h['area_m2']:,.1f} m²",
                "threat_badge": h["threat_badge"],
                "assigned_department": h["department"],
                "inspection_mandate": f"Verify on-ground legal permits for {h['change_type']} ({h['t1_class']} ➔ {h['t2_class']}).",
                "deep_link": h["deep_link"]
            })

        # Overall district threat score (0 to 100)
        threat_score = min(100, (len(critical_encroachments) * 15) + (len(high_ecological_alerts) * 8) + (len(moderate_alerts) * 2))
        risk_rating = "CRITICAL WATCH" if threat_score >= 60 else "MODERATE ATTENTION" if threat_score >= 30 else "LOW RISK"

        district_name = district.capitalize() if district != "all" else "Tamil Nadu State Overview"

        # Calculate percentages for hotspots
        for h in priority_hotspots:
            raw_share = (h["area_m2"] / max(1.0, total_changed_m2)) * 100.0
            if raw_share > 0 and raw_share < 0.01:
                h["percentage_of_district_change"] = 0.01
            else:
                h["percentage_of_district_change"] = round(raw_share, 2)

            score_val = float(h.get("change_score", 0.0))
            area_px = float(h.get("area_pixels", 0) or (h.get("area_m2", 0) / 100.0))
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
                conf_pct = score_val

            h["confidence_percentage"] = round(float(min(100.0, max(5.0, conf_pct))), 1)

        # Calculate percentages for class breakdown
        enhanced_class_breakdown = {}
        for k, v in class_breakdown.items():
            enhanced_class_breakdown[k] = {
                "area_m2": round(v["area_m2"], 1),
                "hectares": round(v["area_m2"] / 10000.0, 2),
                "count": v["count"],
                "percentage_of_total_change": round((v["area_m2"] / max(1.0, total_changed_m2)) * 100, 2),
                "percentage_of_regions": round((v["count"] / max(1, total_regions)) * 100, 1)
            }

        return {
            "status": "success",
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "reporting_authority": "Government of Tamil Nadu — Remote Sensing & Urban Intelligence Cell",
                "dataset_source": "Sentinel-2 Multi-Temporal Imagery & Regional Slippy Ortho-Tiles",
                "model_core": "XGBoost Supervised Change Classifier (Frozen Threshold = 0.72) + Spectral Engine"
            },
            "district": district_name,
            "summary": {
                "total_detected_regions": total_regions,
                "total_changed_m2": round(total_changed_m2, 1),
                "total_changed_hectares": total_changed_hectares,
                "threat_score": threat_score,
                "risk_rating": risk_rating,
                "critical_encroachments_count": len(critical_encroachments),
                "high_ecological_alerts_count": len(high_ecological_alerts),
                "moderate_alerts_count": len(moderate_alerts)
            },
            "class_breakdown": enhanced_class_breakdown,
            "priority_hotspots": priority_hotspots,
            "field_checklist": checklist
        }

    def generate_pdf_report(self, district: str = "all", output_path: Optional[Path] = None) -> Path:
        """
        Generate official Government of Tamil Nadu Environmental Audit & Threat Intelligence PDF.
        """
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        data = self.generate_district_audit(district)
        district_name = data.get("district", "Tamil Nadu")

        project_root = Path(__file__).resolve().parent.parent
        if output_path is None:
            out_dir = project_root / "outputs" / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            dist_slug = district.lower().replace(" ", "_")
            output_path = (out_dir / f"audit_report_{dist_slug}_{int(time.time())}.pdf").resolve()
        else:
            output_path = Path(output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        normal = styles["Normal"]

        title_style = ParagraphStyle(
            "GovTitle",
            parent=normal,
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#0f172a"),
            alignment=1
        )
        subtitle_style = ParagraphStyle(
            "GovSubtitle",
            parent=normal,
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
            alignment=1
        )
        h2_style = ParagraphStyle(
            "SectionH2",
            parent=normal,
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=8,
            spaceAfter=4
        )
        cell_style = ParagraphStyle(
            "CellNormal",
            parent=normal,
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#1e293b")
        )
        cell_bold = ParagraphStyle(
            "CellBold",
            parent=normal,
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#0f172a")
        )

        elements = []

        # Header
        elements.append(Paragraph("GOVERNMENT OF TAMIL NADU", title_style))
        elements.append(Paragraph("State Remote Sensing & Urban Land-Cover Intelligence Cell", subtitle_style))
        elements.append(Paragraph("OFFICIAL ENVIRONMENTAL AUDIT & THREAT INTELLIGENCE REPORT", ParagraphStyle(
            "DocType", parent=title_style, fontSize=11, leading=14, textColor=colors.HexColor("#2563eb"), spaceAfter=4
        )))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceBefore=3, spaceAfter=6))

        # Metadata bar
        meta = data.get("metadata", {})
        summary = data.get("summary", {})
        meta_text = f"<b>Survey District:</b> {district_name.upper()} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Report Ref:</b> TN-ENV-{district[:3].upper()}-2026 &nbsp;&nbsp;|&nbsp;&nbsp; <b>Generated:</b> {meta.get('generated_at', time.strftime('%Y-%m-%d %H:%M:%S'))}"
        elements.append(Paragraph(meta_text, subtitle_style))
        elements.append(Spacer(1, 8))

        # KPI Summary Table
        risk = summary.get("risk_rating", "LOW RISK")
        risk_color = colors.HexColor("#dc2626") if "CRITICAL" in risk else (colors.HexColor("#d97706") if "MODERATE" in risk else colors.HexColor("#16a34a"))

        kpi_data = [
            [
                Paragraph("<b>Total Change Regions</b>", cell_style),
                Paragraph(f"<b>{summary.get('total_detected_regions', 0):,}</b>", cell_bold),
                Paragraph("<b>Total Area Changed</b>", cell_style),
                Paragraph(f"<b>{summary.get('total_changed_hectares', 0):,.2f} ha</b> ({summary.get('total_changed_m2', 0):,.1f} m²)", cell_bold)
            ],
            [
                Paragraph("<b>Overall Threat Score</b>", cell_style),
                Paragraph(f"<b>{summary.get('threat_score', 0)} / 100</b>", cell_bold),
                Paragraph("<b>Risk Rating</b>", cell_style),
                Paragraph(f"<b>{risk}</b>", ParagraphStyle("Risk", parent=cell_bold, textColor=risk_color))
            ],
            [
                Paragraph("<b>Critical Encroachments</b>", cell_style),
                Paragraph(f"<b>{summary.get('critical_encroachments_count', 0)}</b>", cell_bold),
                Paragraph("<b>Ecological Alerts</b>", cell_style),
                Paragraph(f"<b>{summary.get('high_ecological_alerts_count', 0)} High, {summary.get('moderate_alerts_count', 0)} Mod</b>", cell_bold)
            ]
        ]
        kpi_table = Table(kpi_data, colWidths=[130, 130, 130, 133])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(kpi_table)
        elements.append(Spacer(1, 10))

        # Class Breakdown Table
        elements.append(Paragraph("1. Semantic Class Breakdown", h2_style))
        cb = data.get("class_breakdown", {})
        cb_rows = [["Semantic Class", "Changed Area (m²)", "Hectares", "Region Count", "% of Change"]]
        for cls_name, vals in cb.items():
            if vals.get("count", 0) > 0 or vals.get("area_m2", 0) > 0:
                cb_rows.append([
                    cls_name.capitalize(),
                    f"{vals.get('area_m2', 0):,.1f}",
                    f"{vals.get('hectares', 0):.2f}",
                    str(vals.get("count", 0)),
                    f"{vals.get('percentage_of_total_change', 0):.1f}%"
                ])
        if len(cb_rows) == 1:
            cb_rows.append(["No classified changes recorded", "-", "-", "-", "-"])

        cb_table = Table(cb_rows, colWidths=[120, 110, 95, 95, 103])
        cb_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        elements.append(cb_table)
        elements.append(Spacer(1, 10))

        # Top Priority Encroachment Hotspots
        elements.append(Paragraph("2. Top Priority Hotspots & Encroachment Alerts", h2_style))
        hotspots = data.get("priority_hotspots", [])
        hp_rows = [["Rank", "Region ID", "Target GPS", "Area (m²)", "Threat Level", "Assigned Department"]]
        for idx, h in enumerate(hotspots[:6], 1):
            hp_rows.append([
                str(idx),
                str(h.get("region_id", "-")),
                f"{h.get('lat', 0):.4f} N, {h.get('lon', 0):.4f} E",
                f"{h.get('area_m2', 0):,.1f}",
                h.get("threat_level", "NORMAL"),
                h.get("department", "Land Survey Dept")[:35]
            ])
        if len(hp_rows) == 1:
            hp_rows.append(["-", "No priority hotspots found", "-", "-", "-", "-"])

        hp_table = Table(hp_rows, colWidths=[35, 60, 120, 75, 75, 158])
        hp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('ALIGN', (0, 0), (0, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        elements.append(hp_table)
        elements.append(Spacer(1, 10))

        # Ground Inspection Checklist
        elements.append(Paragraph("3. Field Inspection Checklist for Ground Officers", h2_style))
        checklist = data.get("field_checklist", [])
        cl_rows = [["Item", "GPS Location", "Area", "Threat Badge", "Inspection Mandate", "Officer Sign-Off"]]
        for c in checklist[:5]:
            cl_rows.append([
                str(c.get("item_no", "-")),
                c.get("target_location", "-"),
                c.get("area_m2", "-"),
                c.get("threat_badge", "-"),
                Paragraph(c.get("inspection_mandate", "-"), cell_style),
                "[  ] Verified\nDate: ____"
            ])
        if len(cl_rows) == 1:
            cl_rows.append(["-", "No checklist items", "-", "-", "-", "-"])

        cl_table = Table(cl_rows, colWidths=[30, 95, 60, 90, 168, 80])
        cl_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        elements.append(cl_table)
        elements.append(Spacer(1, 14))

        # Verification Signatures
        sig_data = [
            [
                Paragraph("<b>Inspecting Remote Sensing Analyst</b><br/><br/>___________________________<br/>Digital Signature & Stamp", cell_style),
                Paragraph("<b>Revenue Divisional Officer (RDO)</b><br/><br/>___________________________<br/>Seal & Verification Date", cell_style),
                Paragraph("<b>District Collector & Magistrate</b><br/><br/>___________________________<br/>Executive Action Order", cell_style)
            ]
        ]
        sig_table = Table(sig_data, colWidths=[174, 175, 174])
        sig_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEABOVE', (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(sig_table)

        doc.build(elements)
        return output_path
