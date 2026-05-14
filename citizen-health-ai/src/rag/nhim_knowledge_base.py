"""
NHIM / NHM scheme knowledge base for RAG retrieval.

Each document has: id, title, tags (for boosting), content.
"""

NHIM_DOCUMENTS: list[dict] = [
    {
        "id": "pmjay_overview",
        "title": "Ayushman Bharat PM-JAY",
        "tags": ["insurance", "pmjay", "ayushman", "eligibility", "coverage", "hospital", "cashless"],
        "content": (
            "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (PM-JAY) is India's flagship health insurance scheme. "
            "Coverage: ₹5 lakh per family per year for secondary and tertiary hospitalisation. "
            "Eligibility: Based on SECC 2011 data — deprived rural families and specific occupational categories in urban areas. "
            "Income: No fixed income criterion — eligibility is SECC-based; combined with CMCHIS in Tamil Nadu for families earning up to ₹5 lakh/year. "
            "Documents: Ration card, Aadhaar / ABHA (Ayushman Bharat Health Account) card. "
            "Hospitals: Over 23,000 empanelled hospitals nationwide — government and private, all cashless. "
            "Pre-existing conditions covered from day 1. No premium to be paid by beneficiary. "
            "How to check eligibility: Call 14555 (toll-free), visit pmjay.gov.in, or ask Ayushman Mitra at any empanelled hospital. "
            "Karnataka equivalent: Ayushman Bharat – Arogya Karnataka (AB-ArK). "
            "Tamil Nadu: PM-JAY is merged with CMCHIS — single card covers both."
        ),
    },
    {
        "id": "cmchis",
        "title": "CMCHIS — Chief Minister's Comprehensive Health Insurance Scheme (Tamil Nadu)",
        "tags": ["cmchis", "insurance", "tamilnadu", "eligibility", "coverage", "private hospital"],
        "content": (
            "Chief Minister's Comprehensive Health Insurance Scheme (CMCHIS) is Tamil Nadu's state health insurance scheme. "
            "Coverage: ₹5 lakh per family per year. "
            "Eligibility: Families with annual income up to ₹72,000 (₹6,000/month) possessing a government-issued ration card. "
            "Documents required: Ration card, income certificate, Aadhaar card. "
            "Hospitals: ~1,090 empanelled hospitals — all government hospitals (free) and registered private hospitals. "
            "Procedures covered: 1,027+ surgical and medical treatments including cancer, cardiac surgery, dialysis, organ transplants. "
            "New conditions added 2023: 116 new diseases including haematological disorders, cochlear implants. "
            "How to apply: Visit nearest government hospital or Common Service Centre (CSC) with documents. "
            "Helpline: 104 (health helpline), 14555 (PM-JAY helpdesk). "
            "Overlap: Families eligible for both PM-JAY and CMCHIS receive combined coverage — CMCHIS acts as top-up."
        ),
    },
    {
        "id": "jssk",
        "title": "JSSK — Janani Shishu Suraksha Karyakram",
        "tags": ["maternal", "jssk", "delivery", "pregnancy", "free", "transport", "newborn"],
        "content": (
            "Janani Shishu Suraksha Karyakram (JSSK) guarantees zero-expense delivery and newborn care at all public health facilities. "
            "Entitlements for pregnant women: "
            "Free delivery — both normal and C-section; free medicines and consumables; free diagnostics (blood tests, ultrasound); "
            "free diet during stay (₹100/day normal delivery, ₹150/day C-section); free blood transfusion; "
            "free transport from home to facility, inter-facility referral, and back home after discharge; "
            "exempted from ALL user charges at government hospitals. "
            "Entitlements for sick newborns (up to 30 days): "
            "Free treatment at SNCU (Special Newborn Care Unit) or NBSU; free drugs, diagnostics, blood; free diet for mother. "
            "Who can avail: ALL pregnant women delivering at any government health facility — no income restriction. "
            "How to access: Present at Sub-Centre, PHC, CHC, or District Hospital. ASHA can assist with transport entitlement."
        ),
    },
    {
        "id": "pmsma",
        "title": "PMSMA — Pradhan Mantri Surakshit Matritva Abhiyan",
        "tags": ["maternal", "anc", "pmsma", "pregnancy", "checkup", "9th", "monthly"],
        "content": (
            "PMSMA provides free comprehensive antenatal checkups to pregnant women on the 9th of every month at PHCs/CHCs/Government Hospitals. "
            "Services offered on the 9th: Medical officer or specialist examination; abdominal examination; "
            "haemoglobin (Hb) test; blood pressure; blood glucose (gestational diabetes screening); urine albumin and sugar; "
            "ultrasound if not previously done; weight and BMI; IFA tablet provision; high-risk identification. "
            "Eligibility: All pregnant women in 2nd and 3rd trimester (14 weeks and above). "
            "High-risk cases identified and referred to specialists for management. "
            "PMSMA sticker placed on ANC card after each visit as proof. "
            "No registration required — walk in on the 9th of any month. Completely free of charge."
        ),
    },
    {
        "id": "immunization_schedule",
        "title": "Mission Indradhanush — Universal Immunisation Programme (UIP)",
        "tags": ["immunization", "vaccination", "children", "mission indradhanush", "vaccine", "bcg", "opv", "pentavalent"],
        "content": (
            "Mission Indradhanush ensures full immunisation for all children under 2 years and pregnant women. "
            "UIP vaccine schedule: "
            "At birth: BCG (tuberculosis), OPV-0, Hepatitis B birth dose. "
            "6 weeks: Pentavalent-1 (DPT+HepB+Hib), OPV-1, Rotavirus-1, IPV-1, PCV-1. "
            "10 weeks: Pentavalent-2, OPV-2, Rotavirus-2. "
            "14 weeks: Pentavalent-3, OPV-3, Rotavirus-3, IPV-2, PCV-2. "
            "9–12 months: Measles-Rubella (MR-1), JE-1 (endemic districts), PCV booster, Vitamin A-1. "
            "16–24 months: MR-2, DPT booster-1, OPV booster, JE-2, Vitamin A-2. "
            "5–6 years: DPT booster-2. 10 years: TT. 16 years: TT. "
            "Pregnant women: TT-1 and TT-2 during pregnancy. "
            "Intensified Mission Indradhanush (IMI): Special campaigns for left-out/drop-out children. "
            "ASHA tracks due vaccine dates and reminds families. All vaccines free at government facilities."
        ),
    },
    {
        "id": "rbsk",
        "title": "RBSK — Rashtriya Bal Swasthya Karyakram",
        "tags": ["child health", "rbsk", "screening", "disability", "birth defect", "school"],
        "content": (
            "RBSK screens and provides free management of 4D conditions in children aged 0–18 years. "
            "4D conditions: "
            "1. Defects at birth: Congenital heart disease, cleft lip/palate, club foot, neural tube defects. "
            "2. Deficiencies: Anaemia, vitamin A/D deficiency, SAM/MAM malnutrition. "
            "3. Diseases: Skin, ear-nose-throat, vision problems, dental caries. "
            "4. Developmental delays/disabilities: Autism, cerebral palsy, learning disability, hearing impairment. "
            "Process: Mobile Health Teams visit Anganwadi centres and government schools twice yearly. "
            "Positive cases referred to District Early Intervention Centres (DEICs). "
            "Free corrective surgeries, treatment, prosthetics, and assistive devices under ADIP scheme. "
            "Coverage: All children 0–18 years in government anganwadis and schools — completely free."
        ),
    },
    {
        "id": "108_ambulance",
        "title": "108 Emergency Ambulance and 102 Janani Express",
        "tags": ["emergency", "ambulance", "108", "102", "accident", "transport", "janani express"],
        "content": (
            "108 Emergency Medical Services: Free ambulance for all medical emergencies. "
            "Dial 108 — toll-free, 24×7 from any phone. "
            "Response time: Urban areas 15 minutes, rural areas 25 minutes. "
            "Services: Basic and Advanced Life Support, trained EMTs, free transport to nearest government hospital. "
            "Maternity priority: Pregnant women in labour get priority response. "
            "102 Janani Express: Dedicated free transport for pregnant women and sick newborns in rural areas (non-emergency). "
            "ASHA incentive: ASHA receives ₹100–300 for facilitating a 108/102 call for a maternity case. "
            "Both services are completely free — no charges to the patient or family."
        ),
    },
    {
        "id": "jsy",
        "title": "JSY — Janani Suraksha Yojana",
        "tags": ["maternal", "jsy", "cash", "delivery", "incentive", "institutional delivery"],
        "content": (
            "Janani Suraksha Yojana (JSY) provides cash incentives to promote institutional delivery. "
            "Cash benefit for mother in High Performing States (Tamil Nadu, Karnataka): "
            "Rural areas: ₹700 per institutional delivery. Urban areas: ₹600 per delivery. "
            "ASHA incentive: Rural ₹600, Urban ₹400 for facilitating an institutional delivery. "
            "Eligibility: All pregnant women; BPL families prioritised but now near-universal. "
            "Payment via Direct Bank Transfer (DBT) after hospital delivery verification. "
            "Documents: ASHA referral slip, hospital delivery certificate, bank account/Jan Dhan details. "
            "Tamil Nadu additional benefit: Dr. Muthulakshmi Reddy Maternity Benefit Scheme provides ₹18,000 in instalments "
            "for institutional delivery, ANC visits, and immunisation compliance. "
            "JSY + JSSK combination: Cash under JSY plus free services under JSSK complement each other."
        ),
    },
    {
        "id": "nhm_overview",
        "title": "NHM — National Health Mission Overview",
        "tags": ["nhm", "health", "programme", "scheme", "government", "asha", "primary health"],
        "content": (
            "National Health Mission (NHM) is India's flagship public health programme. "
            "Two sub-missions: NRHM (National Rural Health Mission) and NUHM (National Urban Health Mission). "
            "Key pillars: Strengthening PHC/CHC/District Hospitals; ASHA programme (1 per 1,000 population); "
            "Rogi Kalyan Samiti (RKS) for hospital management; Free drugs, diagnostics, diet and blood at government hospitals; "
            "National Ambulance Services (108, 102); Community-based monitoring. "
            "ASHA role: Community health worker who motivates families to use health services, accompanies pregnant women for delivery, "
            "tracks immunisations, conducts home visits, distributes ORS/IFA/contraceptives. "
            "Major programmes: RMNCH+A (reproductive, maternal, newborn, child, adolescent health); "
            "National TB Programme (NTP / NIKSHAY); National Malaria Elimination Programme; "
            "National HIV/AIDS Programme; Integrated Disease Surveillance Programme (IDSP); "
            "National Mental Health Programme; Free Drugs and Diagnostics Initiative."
        ),
    },
    {
        "id": "phc_hierarchy",
        "title": "Government Health Facility Hierarchy and Services",
        "tags": ["hospital", "phc", "chc", "facility", "referral", "opd", "location", "government hospital"],
        "content": (
            "Government health facility levels in Tamil Nadu and Karnataka: "
            "1. Sub-Centre (SC): Serves 3,000–5,000 rural population. ANM-based. Basic ANC, immunisation, family planning, first aid. "
            "2. Primary Health Centre (PHC): Serves 20,000–30,000 population. Medical Officer present. "
            "OPD, basic lab, delivery, immunisation. 24×7 PHCs designated for round-the-clock deliveries. "
            "3. Community Health Centre (CHC): Serves 80,000–1,20,000. 30-bed hospital with 4 specialists "
            "(Obstetrician, Paediatrician, Surgeon, Physician). FRU (First Referral Unit) for emergency obstetric care. "
            "4. Sub-District/Taluk Hospital: 100–300 beds, all specialities at taluk level. "
            "5. District Hospital (DH): 200–500 beds, blood bank, ICU, dialysis, oncology. "
            "6. Government Medical College Hospital (GMCH): 500+ beds, all super-specialities. "
            "Examples: Rajiv Gandhi GGH Chennai, JIPMER Puducherry, Victoria Hospital Bengaluru. "
            "Referral chain: SC → PHC → CHC → District Hospital → GMCH. "
            "ALL services are FREE at government facilities for eligible patients."
        ),
    },
    {
        "id": "poshan_sam",
        "title": "Poshan Abhiyaan — Nutrition and SAM/MAM Management",
        "tags": ["nutrition", "poshan", "icds", "anganwadi", "malnutrition", "sam", "mam", "rutf", "nrc"],
        "content": (
            "Poshan Abhiyaan (National Nutrition Mission) addresses malnutrition in children, adolescents, and mothers. "
            "Services via Anganwadi Centres: Supplementary nutrition (hot meals / take-home ration); "
            "monthly growth monitoring (weight, height); nutrition counselling; pre-school education (3–6 years); referrals. "
            "SAM (Severe Acute Malnutrition): MUAC < 11.5 cm or WFH < -3SD or bilateral pitting oedema. "
            "Community management: RUTF (Ready-to-Use Therapeutic Food) provided at home. "
            "NRC admission: SAM with medical complications, SAM < 6 months, SAM with poor appetite — free 14-day inpatient care with RUTF. "
            "MAM (Moderate Acute Malnutrition): MUAC 11.5–12.5 cm — managed via ICDS supplementary feeding. "
            "Poshan Tracker app used by AWW for real-time reporting. "
            "ASHA and ANM track MUAC monthly and refer SAM children to NRC."
        ),
    },
    {
        "id": "free_services",
        "title": "Free Drugs, Diagnostics and Diet at Government Hospitals",
        "tags": ["free", "drugs", "medicines", "diagnostics", "blood test", "lab", "free treatment"],
        "content": (
            "Free Drugs Service Initiative (FDSI) and Free Diagnostics Service Initiative under NHM. "
            "Free Drugs: All patients at government OPDs and IPDs receive free essential medicines. "
            "Tamil Nadu (TNMSC) and Karnataka (KSDL) maintain Essential Drug Lists. "
            "Includes antibiotics, antihypertensives, antidiabetics, antiepileptics, ARV drugs, iron-folic acid, ORS. "
            "Free Diagnostics: CBC, blood sugar, lipid profile, liver/kidney function tests, urine examination — free at PHC and above. "
            "X-ray and ultrasound at CHC and above. CT scan and MRI at district hospitals and GMCHs. "
            "Free Diet: All IPD patients receive free diet. Post-delivery mothers: ₹100/day (normal), ₹150/day (C-section) under JSSK. "
            "Free Blood: Blood banks at CHC and above — free for all BPL patients and maternity cases. "
            "Tamil Nadu extras: Makkalai Thedi Maruthuvam (Reaching Health to People) door-step health screening; "
            "TN free dialysis, free cancer treatment under CMCHIS."
        ),
    },
    {
        "id": "abha_digital_health",
        "title": "ABHA — Ayushman Bharat Health Account (Digital Health ID)",
        "tags": ["abha", "digital health", "health id", "records", "ayushman", "national health authority"],
        "content": (
            "ABHA (Ayushman Bharat Health Account) is a 14-digit unique health ID for every Indian citizen. "
            "Purpose: Secure digital storage and sharing of health records with consent. "
            "Benefits: Single ID for accessing PM-JAY benefits, storing prescriptions, lab reports, discharge summaries; "
            "no need to carry physical records; doctors can access history instantly with consent. "
            "How to create: Download Aarogya Setu or ABHA app; visit Common Service Centre; "
            "or register at any government hospital using Aadhaar or driving licence. "
            "Creation is free and voluntary. "
            "Linked schemes: PM-JAY card is linked to ABHA ID. "
            "Privacy: Records shared only with explicit patient consent. Patient can revoke access anytime. "
            "Helpline: 14477 (National Health Authority helpline)."
        ),
    },
    {
        "id": "rmncha",
        "title": "RMNCH+A — Reproductive, Maternal, Newborn, Child, Adolescent Health",
        "tags": ["rmncha", "maternal", "child health", "adolescent", "family planning", "contraception"],
        "content": (
            "RMNCH+A is the comprehensive reproductive and child health strategy under NHM. "
            "Reproductive health: Family planning counselling; free contraceptives (condoms, OCPs, MPA injection, IUCD, sterilisation); "
            "management of RTI/STI; medical termination of pregnancy (MTP) services at PHC and above. "
            "Maternal health: Full ANC (4 visits minimum), JSSK, JSY, PMSMA; skilled birth attendance; EmOC; postnatal care. "
            "Newborn care: SNCU, NBSU, NBCC (Newborn Care Corner) at delivery points; KMC (Kangaroo Mother Care) for LBW babies; "
            "HBNC (Home Based Newborn Care) visits by ASHA for 7 days post-delivery. "
            "Child health: IMNCI training for PHC staff; RBSK screening; nutrition management; diarrhoea/pneumonia management. "
            "Adolescent health: RKSK (Rashtriya Kishor Swasthya Karyakram); Peer Education Programme; "
            "Weekly Iron Folic Acid Supplementation (WIFS) for adolescent girls and boys; SABLA scheme."
        ),
    },
]
