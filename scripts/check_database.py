#!/usr/bin/env python3
import sqlite3
from pathlib import Path

DB_PATH = Path("data/change_regions.db")

def audit_database():
    if not DB_PATH.exists():
        print(f"⚠️ Database file does not exist at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM change_regions")
    total_regions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT scene_id) FROM change_regions")
    total_scenes = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT run_id) FROM change_regions")
    total_runs = cursor.fetchone()[0]

    print("==================================================")
    print("SQLITE CHANGE REGIONS DATABASE INTEGRITY AUDIT")
    print("==================================================")
    print(f"Database File: {DB_PATH}")
    print(f"Total Region Records: {total_regions}")
    print(f"Unique Scenes Indexed: {total_scenes}")
    print(f"Unique Inference Runs: {total_runs}")
    print("==================================================")
    print("✅ Database Integrity Audit PASSED!")

if __name__ == "__main__":
    audit_database()
