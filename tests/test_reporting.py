import pytest
from src.reporting import EnvironmentalAuditGenerator
from app.app import app

def test_environmental_audit_generator_chennai():
    """Verify EnvironmentalAuditGenerator logic and calculations."""
    gen = EnvironmentalAuditGenerator()
    report = gen.generate_district_audit("Chennai")

    assert report["status"] == "success"
    assert "metadata" in report
    assert "summary" in report
    assert "class_breakdown" in report
    assert "priority_hotspots" in report
    assert "field_checklist" in report

    summary = report["summary"]
    assert summary["total_detected_regions"] > 0
    assert summary["total_changed_m2"] > 0
    assert summary["total_changed_hectares"] > 0
    assert 0 <= summary["threat_score"] <= 100
    assert summary["risk_rating"] in ["CRITICAL WATCH", "MODERATE ATTENTION", "LOW RISK"]

    # Verify checklist structure
    if report["field_checklist"]:
        item0 = report["field_checklist"][0]
        assert "target_location" in item0
        assert "assigned_department" in item0
        assert "inspection_mandate" in item0
        assert "threat_badge" in item0

def test_environmental_audit_generator_all():
    """Verify state-wide overview audit generation."""
    gen = EnvironmentalAuditGenerator()
    report = gen.generate_district_audit("all")

    assert report["status"] == "success"
    assert report["summary"]["total_detected_regions"] >= 600
    assert len(report["priority_hotspots"]) <= 10

def test_flask_reports_endpoints():
    """Verify /reports UI and /api/reports/district REST API."""
    client = app.test_client()

    # 1. UI route
    res_page = client.get("/reports?district=chennai")
    assert res_page.status_code == 200
    assert b"Environmental Audit Report" in res_page.data
    assert b"GOVERNMENT OF TAMIL NADU" in res_page.data
    assert b"Field Verification Checklist" in res_page.data

    # 2. REST API route
    res_api = client.get("/api/reports/district?district=coimbatore")
    assert res_api.status_code == 200
    api_data = res_api.get_json()
    assert api_data["status"] == "success"
    assert "Coimbatore" in api_data["district"]
    assert api_data["summary"]["total_detected_regions"] > 0

def test_environmental_audit_pdf_generation(tmp_path):
    """Verify backend PDF report generation with ReportLab."""
    gen = EnvironmentalAuditGenerator()
    out_pdf = tmp_path / "test_audit.pdf"
    result_path = gen.generate_pdf_report(district="chennai", output_path=out_pdf)
    assert result_path.exists()
    assert result_path.stat().st_size > 1000

def test_flask_reports_pdf_export_endpoint():
    """Verify /reports/export-pdf returns valid PDF binary."""
    client = app.test_client()
    res = client.get("/reports/export-pdf?district=karur")
    assert res.status_code == 200
    assert "application/pdf" in res.headers.get("Content-Type")
    assert len(res.data) > 1000
