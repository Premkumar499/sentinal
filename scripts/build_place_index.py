#!/usr/bin/env python3
"""
Build Place Name Index for Tamil Nadu Districts
Generates:
  - data/place_index/place_lookup.json  (place name -> GPS coords + tile IDs)
  - data/place_index/tile_places.json   (tile ID -> list of named places inside it)

Supports all 38 districts of Tamil Nadu, major taluks, municipal towns,
and landmark localities. Uses tile bounding box spatial containment.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "dataset_tamilnadu_districts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "place_index"


def normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()


# Comprehensive registry of Tamil Nadu districts, taluks, towns and key areas
TAMIL_NADU_PLACES = [
    # Karur
    {"name": "Karur", "district": "karur", "lat": 10.9601, "lon": 78.0766, "place_type": "city"},
    {"name": "Thanthonimalai", "district": "karur", "lat": 10.9320, "lon": 78.0840, "place_type": "suburb"},
    {"name": "Gandhigramam", "district": "karur", "lat": 10.9410, "lon": 78.0680, "place_type": "neighbourhood"},
    {"name": "Vengamur", "district": "karur", "lat": 10.9850, "lon": 78.0420, "place_type": "village"},
    {"name": "Kulithalai", "district": "karur", "lat": 10.9350, "lon": 78.4180, "place_type": "town"},
    {"name": "Aravakurichi", "district": "karur", "lat": 10.7710, "lon": 77.9120, "place_type": "town"},
    {"name": "Mayanur", "district": "karur", "lat": 10.9420, "lon": 78.2310, "place_type": "town"},
    {"name": "Velayuthampalayam", "district": "karur", "lat": 11.0560, "lon": 78.0120, "place_type": "town"},
    {"name": "Pasupathipalayam", "district": "karur", "lat": 10.9650, "lon": 78.1020, "place_type": "suburb"},
    {"name": "Krishnarayapuram", "district": "karur", "lat": 10.9520, "lon": 78.2780, "place_type": "town"},
    {"name": "Kadavur", "district": "karur", "lat": 10.6120, "lon": 78.1890, "place_type": "town"},

    # Coimbatore
    {"name": "Coimbatore", "district": "coimbatore", "lat": 11.0168, "lon": 76.9558, "place_type": "city"},
    {"name": "Gandhipuram", "district": "coimbatore", "lat": 11.0180, "lon": 76.9680, "place_type": "neighbourhood"},
    {"name": "RS Puram", "district": "coimbatore", "lat": 11.0080, "lon": 76.9480, "place_type": "neighbourhood"},
    {"name": "Peelamedu", "district": "coimbatore", "lat": 11.0310, "lon": 77.0020, "place_type": "suburb"},
    {"name": "Singanallur", "district": "coimbatore", "lat": 10.9980, "lon": 77.0250, "place_type": "suburb"},
    {"name": "Saravanampatti", "district": "coimbatore", "lat": 11.0790, "lon": 76.9980, "place_type": "suburb"},
    {"name": "Pollachi", "district": "coimbatore", "lat": 10.6580, "lon": 77.0080, "place_type": "town"},
    {"name": "Mettupalayam", "district": "coimbatore", "lat": 11.3000, "lon": 76.9400, "place_type": "town"},
    {"name": "Sulur", "district": "coimbatore", "lat": 11.0250, "lon": 77.1260, "place_type": "town"},
    {"name": "Valparai", "district": "coimbatore", "lat": 10.3250, "lon": 76.9550, "place_type": "town"},
    {"name": "Thondamuthur", "district": "coimbatore", "lat": 10.9950, "lon": 76.8320, "place_type": "town"},
    {"name": "Annur", "district": "coimbatore", "lat": 11.2330, "lon": 77.1330, "place_type": "town"},
    {"name": "Madukkarai", "district": "coimbatore", "lat": 10.9020, "lon": 76.9650, "place_type": "town"},

    # Chennai
    {"name": "Chennai", "district": "chennai", "lat": 13.0827, "lon": 80.2707, "place_type": "city"},
    {"name": "Guindy", "district": "chennai", "lat": 13.0067, "lon": 80.2025, "place_type": "suburb"},
    {"name": "T Nagar", "district": "chennai", "lat": 13.0418, "lon": 80.2341, "place_type": "suburb"},
    {"name": "Adyar", "district": "chennai", "lat": 13.0012, "lon": 80.2565, "place_type": "suburb"},
    {"name": "Velachery", "district": "chennai", "lat": 12.9750, "lon": 80.2200, "place_type": "suburb"},
    {"name": "Mylapore", "district": "chennai", "lat": 13.0368, "lon": 80.2676, "place_type": "suburb"},
    {"name": "Anna Nagar", "district": "chennai", "lat": 13.0850, "lon": 80.2100, "place_type": "suburb"},
    {"name": "Royapettah", "district": "chennai", "lat": 13.0550, "lon": 80.2600, "place_type": "suburb"},
    {"name": "Egmore", "district": "chennai", "lat": 13.0780, "lon": 80.2600, "place_type": "suburb"},
    {"name": "Tondiarpet", "district": "chennai", "lat": 13.1250, "lon": 80.2900, "place_type": "suburb"},

    # Madurai
    {"name": "Madurai", "district": "madurai", "lat": 9.9252, "lon": 78.1198, "place_type": "city"},
    {"name": "Goripalayam", "district": "madurai", "lat": 9.9310, "lon": 78.1290, "place_type": "neighbourhood"},
    {"name": "Mattuthavani", "district": "madurai", "lat": 9.9450, "lon": 78.1550, "place_type": "suburb"},
    {"name": "Melur", "district": "madurai", "lat": 10.0290, "lon": 78.3370, "place_type": "town"},
    {"name": "Usilampatti", "district": "madurai", "lat": 9.9700, "lon": 77.7900, "place_type": "town"},
    {"name": "Vadipatti", "district": "madurai", "lat": 10.0600, "lon": 77.9600, "place_type": "town"},
    {"name": "Tirumangalam", "district": "madurai", "lat": 9.8240, "lon": 77.9860, "place_type": "town"},

    # Salem
    {"name": "Salem", "district": "salem", "lat": 11.6643, "lon": 78.1460, "place_type": "city"},
    {"name": "Suramangalam", "district": "salem", "lat": 11.6780, "lon": 78.1180, "place_type": "suburb"},
    {"name": "Hasthampatti", "district": "salem", "lat": 11.6750, "lon": 78.1620, "place_type": "suburb"},
    {"name": "Shevapet", "district": "salem", "lat": 11.6480, "lon": 78.1390, "place_type": "suburb"},
    {"name": "Attur", "district": "salem", "lat": 11.5970, "lon": 78.6010, "place_type": "town"},
    {"name": "Mettur", "district": "salem", "lat": 11.7960, "lon": 77.8010, "place_type": "town"},
    {"name": "Omalur", "district": "salem", "lat": 11.7450, "lon": 78.0420, "place_type": "town"},
    {"name": "Sankari", "district": "salem", "lat": 11.4820, "lon": 77.8680, "place_type": "town"},

    # Tiruchirappalli
    {"name": "Tiruchirappalli", "district": "tiruchirappalli", "lat": 10.7905, "lon": 78.7047, "place_type": "city"},
    {"name": "Trichy", "district": "tiruchirappalli", "lat": 10.7905, "lon": 78.7047, "place_type": "city"},
    {"name": "Srirangam", "district": "tiruchirappalli", "lat": 10.8620, "lon": 78.6940, "place_type": "town"},
    {"name": "Thillai Nagar", "district": "tiruchirappalli", "lat": 10.8250, "lon": 78.6850, "place_type": "suburb"},
    {"name": "Cantonment", "district": "tiruchirappalli", "lat": 10.8020, "lon": 78.6820, "place_type": "suburb"},
    {"name": "Lalgudi", "district": "tiruchirappalli", "lat": 10.8710, "lon": 78.8160, "place_type": "town"},
    {"name": "Manapparai", "district": "tiruchirappalli", "lat": 10.6080, "lon": 78.4170, "place_type": "town"},
    {"name": "Musiri", "district": "tiruchirappalli", "lat": 10.9420, "lon": 78.4520, "place_type": "town"},
    {"name": "Thuvakudi", "district": "tiruchirappalli", "lat": 10.7600, "lon": 78.8050, "place_type": "town"},

    # Tiruppur
    {"name": "Tiruppur", "district": "tiruppur", "lat": 11.1085, "lon": 77.3411, "place_type": "city"},
    {"name": "Avinashi", "district": "tiruppur", "lat": 11.1920, "lon": 77.2690, "place_type": "town"},
    {"name": "Palladam", "district": "tiruppur", "lat": 10.9980, "lon": 77.2920, "place_type": "town"},
    {"name": "Dharapuram", "district": "tiruppur", "lat": 10.7300, "lon": 77.5300, "place_type": "town"},
    {"name": "Kangeyam", "district": "tiruppur", "lat": 11.0050, "lon": 77.5610, "place_type": "town"},
    {"name": "Udumalaipettai", "district": "tiruppur", "lat": 10.5840, "lon": 77.2480, "place_type": "town"},

    # Erode
    {"name": "Erode", "district": "erode", "lat": 11.3410, "lon": 77.7172, "place_type": "city"},
    {"name": "Bhavani", "district": "erode", "lat": 11.4460, "lon": 77.6830, "place_type": "town"},
    {"name": "Perundurai", "district": "erode", "lat": 11.2750, "lon": 77.5830, "place_type": "town"},
    {"name": "Gobichettipalayam", "district": "erode", "lat": 11.4550, "lon": 77.4370, "place_type": "town"},
    {"name": "Sathyamangalam", "district": "erode", "lat": 11.5050, "lon": 77.2380, "place_type": "town"},
    {"name": "Anthiyur", "district": "erode", "lat": 11.5810, "lon": 77.5880, "place_type": "town"},

    # Ariyalur
    {"name": "Ariyalur", "district": "ariyalur", "lat": 11.1401, "lon": 79.0786, "place_type": "town"},
    {"name": "Jayankondam", "district": "ariyalur", "lat": 11.2150, "lon": 79.3140, "place_type": "town"},
    {"name": "Sendurai", "district": "ariyalur", "lat": 11.2670, "lon": 79.1670, "place_type": "town"},
    {"name": "Udayarpalayam", "district": "ariyalur", "lat": 11.1890, "lon": 79.2980, "place_type": "town"},

    # Chengalpattu
    {"name": "Chengalpattu", "district": "chengalpattu", "lat": 12.6841, "lon": 79.9836, "place_type": "town"},
    {"name": "Tambaram", "district": "chengalpattu", "lat": 12.9249, "lon": 80.1265, "place_type": "city"},
    {"name": "Pallavaram", "district": "chengalpattu", "lat": 12.9675, "lon": 80.1491, "place_type": "suburb"},
    {"name": "Maraimalai Nagar", "district": "chengalpattu", "lat": 12.7950, "lon": 80.0240, "place_type": "town"},
    {"name": "Maduranthakam", "district": "chengalpattu", "lat": 12.5110, "lon": 79.8850, "place_type": "town"},
    {"name": "Mahabalipuram", "district": "chengalpattu", "lat": 12.6269, "lon": 80.1927, "place_type": "heritage_town"},

    # Dindigul
    {"name": "Dindigul", "district": "dindigul", "lat": 10.3673, "lon": 77.9803, "place_type": "city"},
    {"name": "Palani", "district": "dindigul", "lat": 10.4500, "lon": 77.5180, "place_type": "town"},
    {"name": "Kodaikanal", "district": "dindigul", "lat": 10.2381, "lon": 77.4892, "place_type": "hill_station"},
    {"name": "Oddanchatram", "district": "dindigul", "lat": 10.4850, "lon": 77.7470, "place_type": "town"},
    {"name": "Natham", "district": "dindigul", "lat": 10.2240, "lon": 78.2350, "place_type": "town"},

    # Kanyakumari
    {"name": "Nagercoil", "district": "kanyakumari", "lat": 8.1833, "lon": 77.4119, "place_type": "city"},
    {"name": "Kanyakumari", "district": "kanyakumari", "lat": 8.0883, "lon": 77.5385, "place_type": "town"},
    {"name": "Padmanabhapuram", "district": "kanyakumari", "lat": 8.2430, "lon": 77.3320, "place_type": "town"},
    {"name": "Colachel", "district": "kanyakumari", "lat": 8.1780, "lon": 77.2560, "place_type": "port_town"},
    {"name": "Marthandam", "district": "kanyakumari", "lat": 8.3050, "lon": 77.2210, "place_type": "town"},

    # Tirunelveli
    {"name": "Tirunelveli", "district": "tirunelveli", "lat": 8.7139, "lon": 77.7567, "place_type": "city"},
    {"name": "Palayamkottai", "district": "tirunelveli", "lat": 8.7180, "lon": 77.7420, "place_type": "suburb"},
    {"name": "Ambasamudram", "district": "tirunelveli", "lat": 8.7060, "lon": 77.4570, "place_type": "town"},
    {"name": "Nanguneri", "district": "tirunelveli", "lat": 8.4890, "lon": 77.6540, "place_type": "town"},

    # Thoothukudi
    {"name": "Thoothukudi", "district": "thoothukudi", "lat": 8.7642, "lon": 78.1348, "place_type": "city"},
    {"name": "Tuticorin", "district": "thoothukudi", "lat": 8.7642, "lon": 78.1348, "place_type": "city"},
    {"name": "Kovilpatti", "district": "thoothukudi", "lat": 9.1730, "lon": 77.8680, "place_type": "town"},
    {"name": "Tiruchendur", "district": "thoothukudi", "lat": 8.4980, "lon": 78.1250, "place_type": "town"},

    # Thanjavur
    {"name": "Thanjavur", "district": "thanjavur", "lat": 10.7870, "lon": 79.1378, "place_type": "city"},
    {"name": "Kumbakonam", "district": "thanjavur", "lat": 10.9602, "lon": 79.3845, "place_type": "town"},
    {"name": "Pattukkottai", "district": "thanjavur", "lat": 10.4280, "lon": 79.3170, "place_type": "town"},
    {"name": "Thiruvaiyaru", "district": "thanjavur", "lat": 10.8800, "lon": 79.1050, "place_type": "town"},

    # Nilgiris
    {"name": "Ooty", "district": "nilgiris", "lat": 11.4102, "lon": 76.6950, "place_type": "hill_station"},
    {"name": "Udhagamandalam", "district": "nilgiris", "lat": 11.4102, "lon": 76.6950, "place_type": "hill_station"},
    {"name": "Coonoor", "district": "nilgiris", "lat": 11.3530, "lon": 76.7959, "place_type": "town"},
    {"name": "Kotagiri", "district": "nilgiris", "lat": 11.4280, "lon": 76.8680, "place_type": "town"},
    {"name": "Gudalur", "district": "nilgiris", "lat": 11.5070, "lon": 76.4950, "place_type": "town"},

    # Krishnagiri
    {"name": "Krishnagiri", "district": "krishnagiri", "lat": 12.5186, "lon": 78.2137, "place_type": "city"},
    {"name": "Hosur", "district": "krishnagiri", "lat": 12.7409, "lon": 77.8253, "place_type": "industrial_city"},
    {"name": "Denkanikottai", "district": "krishnagiri", "lat": 12.5250, "lon": 77.7850, "place_type": "town"},
    {"name": "Pochampalli", "district": "krishnagiri", "lat": 12.3380, "lon": 78.3680, "place_type": "town"},

    # Dharmapuri
    {"name": "Dharmapuri", "district": "dharmapuri", "lat": 12.1211, "lon": 78.1582, "place_type": "city"},
    {"name": "Harur", "district": "dharmapuri", "lat": 12.0620, "lon": 78.4980, "place_type": "town"},
    {"name": "Palacode", "district": "dharmapuri", "lat": 12.3020, "lon": 78.0780, "place_type": "town"},
    {"name": "Pennagaram", "district": "dharmapuri", "lat": 12.1320, "lon": 77.9020, "place_type": "town"},

    # Vellore
    {"name": "Vellore", "district": "vellore", "lat": 12.9165, "lon": 79.1325, "place_type": "city"},
    {"name": "Katpadi", "district": "vellore", "lat": 12.9780, "lon": 79.1350, "place_type": "suburb"},
    {"name": "Gudiyatham", "district": "vellore", "lat": 12.9450, "lon": 78.8710, "place_type": "town"},

    # Tiruvannamalai
    {"name": "Tiruvannamalai", "district": "tiruvannamalai", "lat": 12.2253, "lon": 79.0747, "place_type": "city"},
    {"name": "Arani", "district": "tiruvannamalai", "lat": 12.6710, "lon": 79.2850, "place_type": "town"},
    {"name": "Cheyyar", "district": "tiruvannamalai", "lat": 12.6580, "lon": 79.5420, "place_type": "town"},

    # Cuddalore
    {"name": "Cuddalore", "district": "cuddalore", "lat": 11.7480, "lon": 79.7714, "place_type": "city"},
    {"name": "Chidambaram", "district": "cuddalore", "lat": 11.3990, "lon": 79.6930, "place_type": "town"},
    {"name": "Panruti", "district": "cuddalore", "lat": 11.7680, "lon": 79.5520, "place_type": "town"},
    {"name": "Neyveli", "district": "cuddalore", "lat": 11.6050, "lon": 79.4880, "place_type": "town"},

    # Viluppuram
    {"name": "Viluppuram", "district": "viluppuram", "lat": 11.9401, "lon": 79.4861, "place_type": "city"},
    {"name": "Tindivanam", "district": "viluppuram", "lat": 12.2350, "lon": 79.6520, "place_type": "town"},
    {"name": "Gingee", "district": "viluppuram", "lat": 12.2520, "lon": 79.4180, "place_type": "heritage_town"},

    # Namakkal
    {"name": "Namakkal", "district": "namakkal", "lat": 11.2189, "lon": 78.1674, "place_type": "city"},
    {"name": "Tiruchengode", "district": "namakkal", "lat": 11.3810, "lon": 77.8920, "place_type": "town"},
    {"name": "Rasipuram", "district": "namakkal", "lat": 11.4650, "lon": 78.1780, "place_type": "town"},
    {"name": "Kolli Hills", "district": "namakkal", "lat": 11.2480, "lon": 78.3380, "place_type": "hill_station"},

    # Kanchipuram
    {"name": "Kanchipuram", "district": "kanchipuram", "lat": 12.8342, "lon": 79.7036, "place_type": "city"},
    {"name": "Sriperumbudur", "district": "kanchipuram", "lat": 12.9690, "lon": 79.9480, "place_type": "town"},

    # Tiruvallur
    {"name": "Tiruvallur", "district": "tiruvallur", "lat": 13.1432, "lon": 79.9079, "place_type": "city"},
    {"name": "Avadi", "district": "tiruvallur", "lat": 13.1147, "lon": 80.1098, "place_type": "city"},
    {"name": "Poonamallee", "district": "tiruvallur", "lat": 13.0480, "lon": 80.1110, "place_type": "town"},

    # Ramanathapuram
    {"name": "Ramanathapuram", "district": "ramanathapuram", "lat": 9.3639, "lon": 78.8395, "place_type": "city"},
    {"name": "Rameswaram", "district": "ramanathapuram", "lat": 9.2876, "lon": 79.3129, "place_type": "town"},
    {"name": "Paramakudi", "district": "ramanathapuram", "lat": 9.5440, "lon": 78.5910, "place_type": "town"},

    # Virudhunagar
    {"name": "Virudhunagar", "district": "virudhunagar", "lat": 9.5680, "lon": 77.9624, "place_type": "city"},
    {"name": "Sivakasi", "district": "virudhunagar", "lat": 9.4532, "lon": 77.7977, "place_type": "city"},
    {"name": "Rajapalayam", "district": "virudhunagar", "lat": 9.4530, "lon": 77.5540, "place_type": "city"},

    # Theni
    {"name": "Theni", "district": "theni", "lat": 10.0104, "lon": 77.4768, "place_type": "city"},
    {"name": "Bodinayakanur", "district": "theni", "lat": 10.0120, "lon": 77.3520, "place_type": "town"},
    {"name": "Periyakulam", "district": "theni", "lat": 10.1200, "lon": 77.5450, "place_type": "town"},

    # Tenkasi
    {"name": "Tenkasi", "district": "tenkasi", "lat": 8.9594, "lon": 77.3146, "place_type": "city"},
    {"name": "Sankarankovil", "district": "tenkasi", "lat": 9.1720, "lon": 77.5340, "place_type": "town"},
    {"name": "Courtallam", "district": "tenkasi", "lat": 8.9280, "lon": 77.2690, "place_type": "waterfall_town"},

    # Pudukkottai
    {"name": "Pudukkottai", "district": "pudukkottai", "lat": 10.3797, "lon": 78.8208, "place_type": "city"},
    {"name": "Aranthangi", "district": "pudukkottai", "lat": 10.1650, "lon": 78.9980, "place_type": "town"},

    # Sivaganga
    {"name": "Sivaganga", "district": "sivaganga", "lat": 9.8433, "lon": 78.4809, "place_type": "city"},
    {"name": "Karaikudi", "district": "sivaganga", "lat": 10.0673, "lon": 78.7758, "place_type": "city"},

    # Nagapattinam & Mayiladuthurai & Tiruvarur
    {"name": "Nagapattinam", "district": "nagapattinam", "lat": 10.7654, "lon": 79.8424, "place_type": "port_town"},
    {"name": "Velankanni", "district": "nagapattinam", "lat": 10.6800, "lon": 79.8480, "place_type": "pilgrimage_town"},
    {"name": "Mayiladuthurai", "district": "mayiladuthurai", "lat": 11.1075, "lon": 79.6524, "place_type": "city"},
    {"name": "Tiruvarur", "district": "tiruvarur", "lat": 10.7725, "lon": 79.6365, "place_type": "city"},
    {"name": "Mannargudi", "district": "tiruvarur", "lat": 10.6650, "lon": 79.4450, "place_type": "town"},

    # Kallakurichi
    {"name": "Kallakurichi", "district": "kallakurichi", "lat": 11.7380, "lon": 78.9630, "place_type": "city"},
    {"name": "Ulundurpet", "district": "kallakurichi", "lat": 11.6920, "lon": 79.2880, "place_type": "town"},

    # Perambalur
    {"name": "Perambalur", "district": "perambalur", "lat": 11.2333, "lon": 78.8829, "place_type": "city"},

    # Ranipet & Tirupathur
    {"name": "Ranipet", "district": "ranipet", "lat": 12.9270, "lon": 79.3330, "place_type": "city"},
    {"name": "Arakkonam", "district": "ranipet", "lat": 13.0800, "lon": 79.6700, "place_type": "city"},
    {"name": "Tirupathur", "district": "tirupathur", "lat": 12.4950, "lon": 78.5680, "place_type": "city"},
    {"name": "Ambur", "district": "tirupathur", "lat": 12.7910, "lon": 78.7160, "place_type": "city"},
    {"name": "Vaniyambadi", "district": "tirupathur", "lat": 12.6820, "lon": 78.6180, "place_type": "city"}
]


def load_all_tiles() -> Dict[str, Dict]:
    """Scan all *_meta.json files in dataset_tamilnadu_districts."""
    tiles = {}
    for dist_dir in BASE_DIR.iterdir():
        if not dist_dir.is_dir() or dist_dir.name.startswith(".") or dist_dir.name.startswith("tamilnadu"):
            continue
        close_dir = dist_dir / "close_aerial_z14"
        if not close_dir.exists():
            continue
        for meta_file in close_dir.glob("*_meta.json"):
            try:
                with open(meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                tid = meta.get("tile_id", meta_file.stem.replace("_meta", ""))
                bbox = meta.get("bbox", [])
                if len(bbox) >= 4:
                    tiles[tid] = {
                        "tile_id": tid,
                        "district": dist_dir.name.lower(),
                        "bbox": bbox,  # [min_lon, min_lat, max_lon, max_lat]
                        "center_lat": meta.get("center_lat", (bbox[1] + bbox[3]) / 2),
                        "center_lon": meta.get("center_lon", (bbox[0] + bbox[2]) / 2),
                        "meta_path": str(meta_file)
                    }
            except Exception:
                pass
    return tiles


def build_place_indices():
    print("=" * 70)
    print("🗺️  BUILDING TAMIL NADU PLACE NAME SEARCH INDEX")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles = load_all_tiles()
    print(f"Loaded {len(tiles)} total tiles from dataset_tamilnadu_districts.")

    place_lookup: Dict[str, Any] = {}
    tile_places: Dict[str, Any] = {tid: {"tile_id": tid, "district": info["district"], "places": []} for tid, info in tiles.items()}

    # 1. Map registered Tamil Nadu places to tiles
    matched_places = 0
    for p in TAMIL_NADU_PLACES:
        p_name = p["name"]
        p_norm = normalize(p_name)
        p_lat = p["lat"]
        p_lon = p["lon"]
        p_dist = p["district"]

        # Find tiles containing this point
        covering_tiles = []
        for tid, tinfo in tiles.items():
            b = tinfo["bbox"]
            if b[0] <= p_lon <= b[2] and b[1] <= p_lat <= b[3]:
                covering_tiles.append(tid)

        # If not strictly within bbox (e.g. slight coordinate drift), find nearest tile in same district
        if not covering_tiles:
            dist_tiles = [tid for tid, tinfo in tiles.items() if tinfo["district"] == p_dist]
            if dist_tiles:
                def dist_sq(tid):
                    t = tiles[tid]
                    return (t["center_lat"] - p_lat) ** 2 + (t["center_lon"] - p_lon) ** 2
                best_tid = min(dist_tiles, key=dist_sq)
                covering_tiles = [best_tid]

        if covering_tiles:
            matched_places += 1
            entry = {
                "name": p_name,
                "district": p_dist,
                "lat": p_lat,
                "lon": p_lon,
                "place_type": p.get("place_type", "town"),
                "tiles": covering_tiles,
                "years_available": ["2020", "2022", "2024"]
            }
            place_lookup[p_norm] = entry

            # Also add to tile_places
            for tid in covering_tiles:
                tile_places[tid]["places"].append({
                    "name": p_name,
                    "place_type": p.get("place_type", "town"),
                    "lat": p_lat,
                    "lon": p_lon
                })

    # 2. Add District-level place entries so querying "karur", "coimbatore", etc. always works
    for dist_dir in BASE_DIR.iterdir():
        d_clean = dist_dir.name.lower()
        if not dist_dir.is_dir() or d_clean.startswith(".") or d_clean.startswith("tamilnadu"):
            continue
        dist_tiles = [tid for tid, tinfo in tiles.items() if tinfo["district"] == d_clean]
        if dist_tiles:
            lats = [tiles[t]["center_lat"] for t in dist_tiles]
            lons = [tiles[t]["center_lon"] for t in dist_tiles]
            c_lat = round(sum(lats) / len(lats), 5)
            c_lon = round(sum(lons) / len(lons), 5)

            d_norm = normalize(d_clean)
            if d_norm not in place_lookup:
                place_lookup[d_norm] = {
                    "name": d_clean.capitalize(),
                    "district": d_clean,
                    "lat": c_lat,
                    "lon": c_lon,
                    "place_type": "district_capital",
                    "tiles": dist_tiles[:10],
                    "years_available": ["2020", "2022", "2024"]
                }

    # 3. Write output files
    place_lookup_path = OUTPUT_DIR / "place_lookup.json"
    tile_places_path = OUTPUT_DIR / "tile_places.json"

    with open(place_lookup_path, "w", encoding="utf-8") as f:
        json.dump(place_lookup, f, indent=2)

    with open(tile_places_path, "w", encoding="utf-8") as f:
        json.dump(tile_places, f, indent=2)

    print(f"✅ Generated {len(place_lookup)} place lookup entries: {place_lookup_path}")
    print(f"✅ Generated {len(tile_places)} tile-to-place mappings: {tile_places_path}")
    print(f"✅ Place Index Build Complete!")


if __name__ == "__main__":
    build_place_indices()
