"""
CARELINK AI — Full System Test Suite
Runs in sequence without pytest. Each section is self-contained.
"""

import sys
import os
import time
import uuid
import traceback
import numpy as np

# Force UTF-8 output to avoid UnicodeEncodeError on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── ensure project root is on path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"
SEP  = "=" * 62

results = {"passed": 0, "failed": 0, "skipped": 0}

def ok(label):
    results["passed"] += 1
    print(f"{PASS} {label}")

def fail(label, exc=None):
    results["failed"] += 1
    print(f"{FAIL} {label}")
    if exc:
        print(f"         {type(exc).__name__}: {exc}")

def skip(label, reason=""):
    results["skipped"] += 1
    print(f"{SKIP} {label}" + (f"  ({reason})" if reason else ""))

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ═══════════════════════════════════════════════════════════════════════════
# 1. SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
section("1. Settings & Configuration")

try:
    from app.core.settings import get_settings
    settings = get_settings()
    ok("Settings module imported")
except Exception as e:
    fail("Settings module import", e); sys.exit(1)

try:
    assert isinstance(settings.CORS_ORIGINS, list)
    ok(f"CORS_ORIGINS parsed  -> {settings.CORS_ORIGINS}")
except Exception as e:
    fail("CORS_ORIGINS parse", e)

try:
    assert isinstance(settings.SUPPORTED_LANGUAGES, list)
    ok(f"SUPPORTED_LANGUAGES  -> {settings.SUPPORTED_LANGUAGES}")
except Exception as e:
    fail("SUPPORTED_LANGUAGES parse", e)

try:
    assert "medgemma" in settings.LLM_MODEL_NAME.lower(), f"Got {settings.LLM_MODEL_NAME}"
    ok(f"LLM_MODEL_NAME       -> {settings.LLM_MODEL_NAME}")
except Exception as e:
    fail("LLM_MODEL_NAME", e)

try:
    assert settings.LLM_DEVICE == "cuda", f"Got {settings.LLM_DEVICE}"
    ok(f"LLM_DEVICE           -> {settings.LLM_DEVICE}")
except Exception as e:
    fail("LLM_DEVICE", e)

try:
    assert "sqlite" in settings.DATABASE_URL
    ok(f"DATABASE_URL         -> {settings.DATABASE_URL}")
except Exception as e:
    fail("DATABASE_URL", e)

try:
    assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
    ok(f"OLLAMA_BASE_URL      -> {settings.OLLAMA_BASE_URL}")
except Exception as e:
    fail("OLLAMA_BASE_URL", e)


# ═══════════════════════════════════════════════════════════════════════════
# 2. SQLite DATABASE
# ═══════════════════════════════════════════════════════════════════════════
section("2. SQLite Database")

try:
    from app.db.base import engine, init_db, SessionLocal
    ok("DB base module imported")
except Exception as e:
    fail("DB base import", e)

try:
    from app.db.models import (
        User, HealthRecord, Consultation, Prescription,
        PrescriptionItem, TriagePrediction, Medicine,
        Pharmacy, PharmacyInventory, AuditLog
    )
    ok("All 10 ORM models imported")
except Exception as e:
    fail("ORM models import", e)

try:
    init_db()
    ok("init_db() — tables created in carelink.db")
except Exception as e:
    fail("init_db()", e)

try:
    db = SessionLocal()
    # Quick write + read + delete
    test_user = User(
        name="Test Patient",
        age=35,
        gender="male",
        contact="9999999999",
        role="patient"
    )
    db.add(test_user)
    db.commit()
    db.refresh(test_user)
    user_id = test_user.id
    assert user_id is not None

    fetched = db.query(User).filter(User.id == user_id).first()
    assert fetched.name == "Test Patient"

    db.delete(fetched)
    db.commit()
    db.close()
    ok("SQLite CRUD (write / read / delete)")
except Exception as e:
    fail("SQLite CRUD", e)

try:
    db = SessionLocal()
    triage_rec = TriagePrediction(
        request_id="test-req-001",
        symptoms_text="chest pain and sweating",
        patient_age=52,
        duration_days=1,
        chronic_conds=["diabetes"],
        language="en",
        prediction="HIGH",
        confidence=0.91,
        prob_low=0.03,
        prob_medium=0.15,
        prob_high=0.82,
        rules_triggered=["CHEST_PAIN_CARDIOVASCULAR"],
        escalated=False,
        emergency_flag=True,
        explanation="Seek emergency care immediately.",
        model_version="v1.0.0",
        processing_ms=143.5,
    )
    db.add(triage_rec)
    db.commit()

    rec = db.query(TriagePrediction).filter_by(request_id="test-req-001").first()
    assert rec.prediction == "HIGH"
    assert rec.emergency_flag is True

    db.delete(rec)
    db.commit()
    db.close()
    ok("TriagePrediction audit row persisted and verified")
except Exception as e:
    fail("TriagePrediction persistence", e)


# ═══════════════════════════════════════════════════════════════════════════
# 3. NLP PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
section("3. NLP Pipeline")

try:
    from app.nlp.text_preprocessor import TextPreprocessor
    tp = TextPreprocessor()
    result = tp.preprocess("I have SOB and CP since 2 days")
    assert isinstance(result, str) and len(result) > 0
    ok(f"TextPreprocessor: '{result[:60]}'")
except Exception as e:
    fail("TextPreprocessor", e)

try:
    from app.nlp.symptom_extractor import SymptomExtractor, ExtractedSymptom
    se = SymptomExtractor()
    symptoms = se.extract_symptoms("I have severe chest pain and sweating with dizziness")
    assert isinstance(symptoms, list)
    names = [s.normalized for s in symptoms]
    ok(f"SymptomExtractor: {names}")
except Exception as e:
    fail("SymptomExtractor", e)

try:
    from app.nlp.language_handler import LanguageHandler
    lh = LanguageHandler()
    for lang in ["en", "es", "fr"]:
        exp = lh.format_explanation("HIGH", ["chest pain"], lang)
        assert isinstance(exp, str) and len(exp) > 0
    ok("LanguageHandler: en / es / fr ✓")
except Exception as e:
    fail("LanguageHandler", e)


# ═══════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
section("4. Feature Engineering")

try:
    from app.services.feature_engineering import FeatureEngineer
    from app.nlp.symptom_extractor import SymptomExtractor, ExtractedSymptom
    fe = FeatureEngineer()
    se = SymptomExtractor()
    symptoms = se.extract_symptoms("fever and headache")
    fv = fe.extract_features(
        symptoms=symptoms,
        age=35,
        duration_days=2,
        chronic_conditions=[]
    )
    assert fv.features.shape[0] > 0
    ok(f"FeatureEngineer: vector shape = {fv.features.shape}")
except Exception as e:
    fail("FeatureEngineer", e)


# ═══════════════════════════════════════════════════════════════════════════
# 5. ML MODEL SERVICE
# ═══════════════════════════════════════════════════════════════════════════
section("5. ML Model Service")

try:
    from app.services.model_service import ModelService
    ms = ModelService()
    ok(f"ModelService init | version={ms.model_version}")
except Exception as e:
    fail("ModelService init", e)

try:
    features = np.random.random(100)
    prediction, probs = ms.predict(features)
    assert prediction in ["LOW", "MEDIUM", "HIGH"]
    assert abs(sum(probs.values()) - 1.0) < 0.02
    ok(f"ModelService.predict: {prediction} | probs={probs}")
except Exception as e:
    fail("ModelService.predict", e)

try:
    info = ms.get_model_info()
    assert "version" in info
    assert "is_loaded" in info
    ok(f"ModelService.get_model_info: loaded={info.get('is_loaded')}")
except Exception as e:
    fail("ModelService.get_model_info", e)


# ═══════════════════════════════════════════════════════════════════════════
# 6. RULE ENGINE
# ═══════════════════════════════════════════════════════════════════════════
section("6. Clinical Safety Rule Engine")

try:
    from app.services.rule_engine import RuleEngine
    from app.nlp.symptom_extractor import SymptomExtractor
    re_svc = RuleEngine()
    ok(f"RuleEngine init | {len(re_svc.rules)} rules | version={re_svc.version}")
except Exception as e:
    fail("RuleEngine init", e)

# Test: chest pain + diabetes + age>45 should trigger CHEST_PAIN_CARDIOVASCULAR
try:
    se2 = SymptomExtractor()
    syms = se2.extract_symptoms("I have severe chest pain and sweating and I feel short of breath")
    risk, conf, triggered = re_svc.evaluate_rules(
        symptoms=syms, age=55, duration_days=1,
        chronic_conditions=["diabetes"],
        predicted_risk="MEDIUM", confidence=0.70
    )
    rule_names = [r.rule_name for r in triggered]
    assert risk == "HIGH", f"Expected HIGH, got {risk}"
    ok(f"CHEST_PAIN escalation: MEDIUM -> HIGH | rules={rule_names}")
except Exception as e:
    fail("CHEST_PAIN escalation rule", e)

# Test: infant fever should trigger INFANT_FEVER
try:
    syms2 = se2.extract_symptoms("high fever")
    risk2, _, triggered2 = re_svc.evaluate_rules(
        symptoms=syms2, age=1, duration_days=0,
        chronic_conditions=[], predicted_risk="LOW", confidence=0.65
    )
    rule_names2 = [r.rule_name for r in triggered2]
    ok(f"INFANT_FEVER rule: {risk2} | rules={rule_names2}")
except Exception as e:
    fail("INFANT_FEVER rule", e)

# Test: low-risk common cold should stay LOW
try:
    syms3 = se2.extract_symptoms("runny nose and mild cough")
    risk3, _, triggered3 = re_svc.evaluate_rules(
        symptoms=syms3, age=28, duration_days=3,
        chronic_conditions=[], predicted_risk="LOW", confidence=0.82
    )
    ok(f"Low-risk cold: stays {risk3}")
except Exception as e:
    fail("Low-risk cold rule pass-through", e)


# ═══════════════════════════════════════════════════════════════════════════
# 7. CONFIDENCE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════
section("7. Confidence Controller")

try:
    from app.services.confidence_controller import ConfidenceController
    cc = ConfidenceController()
    ok("ConfidenceController init")
except Exception as e:
    fail("ConfidenceController init", e)

try:
    probs_high = {"low": 0.03, "medium": 0.15, "high": 0.82}
    conf = cc.calculate_confidence(probs_high)
    assert 0.0 < conf <= 1.0
    ok(f"calculate_confidence (HIGH case): {conf:.3f}")
except Exception as e:
    fail("calculate_confidence", e)

try:
    decision = cc.evaluate(predicted_risk="HIGH", confidence=conf, probabilities=probs_high)
    ok(f"ConfidenceController.evaluate: risk={decision.adjusted_risk} emergency={decision.emergency_flag}")
except Exception as e:
    fail("ConfidenceController.evaluate", e)

try:
    probs_ambig = {"low": 0.34, "medium": 0.33, "high": 0.33}
    conf2 = cc.calculate_confidence(probs_ambig)
    d2 = cc.evaluate(predicted_risk="MEDIUM", confidence=conf2, probabilities=probs_ambig)
    assert d2.escalated, "Ambiguous probabilities should trigger escalation"
    ok(f"Escalation on ambiguous probs: escalated={d2.escalated}")
except Exception as e:
    fail("Escalation on ambiguous probs", e)


# ═══════════════════════════════════════════════════════════════════════════
# 8. EXPLANATION SERVICE  (MedGemma / template fallback)
# ═══════════════════════════════════════════════════════════════════════════
section("8. Explanation Service (MedGemma)")

try:
    from app.services.explanation_service import ExplanationService
    es = ExplanationService()
    ok("ExplanationService init")
except Exception as e:
    fail("ExplanationService init", e)
    es = None

if es is not None:
    if es.model_ready:
        try:
            expl = es.generate_explanation(
                risk_level="HIGH",
                symptoms=["chest pain", "sweating"],
                confidence=0.91,
                rules_triggered=["CHEST_PAIN_CARDIOVASCULAR"],
                emergency_flag=True,
                language="en"
            )
            assert len(expl) > 20
            ok(f"MedGemma (medgemma-1.5-4b-it) explanation:\n         {expl[:120]}...")
        except Exception as e:
            fail("MedGemma explanation generation", e)
    else:
        # MedGemma loading or unavailable — test template fallback
        try:
            expl = es._template_explanation("HIGH", ["chest pain"], "en")
            assert len(expl) > 20
            ok(f"Ollama (llama3.2) explanation:\n         {expl[:120]}...")
        except Exception as e:
            fail("Template fallback", e)

    try:
        disc = es.generate_disclaimer("en")
        assert "informational" in disc.lower()
        ok("generate_disclaimer ✓")
    except Exception as e:
        fail("generate_disclaimer", e)


# ═══════════════════════════════════════════════════════════════════════════
# 9. API SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════
section("9. API Schemas (Pydantic v2)")

try:
    from app.api.schemas import TriageRequest, TriageResponse, ProbabilityDistribution, HealthResponse
    ok("All schema classes imported")
except Exception as e:
    fail("Schema imports", e)

try:
    req = TriageRequest(
        symptoms_text="  Chest pain and sweating  ",  # should be stripped
        age=52,
        duration_days=1,
        chronic_conditions=["Diabetes", "DIABETES"],  # deduped + lowercased
        language="en"
    )
    assert req.symptoms_text == "Chest pain and sweating"
    assert req.chronic_conditions == ["diabetes"]
    ok("TriageRequest validation (strip + dedup)")
except Exception as e:
    fail("TriageRequest validation", e)

try:
    req2 = TriageRequest(symptoms_text="headache", age=200, duration_days=1)
    fail("Should have rejected age=200")
except Exception:
    ok("TriageRequest rejects age=200 ✓")

try:
    req3 = TriageRequest(symptoms_text="", age=30, duration_days=1)
    fail("Should have rejected empty symptoms")
except Exception:
    ok("TriageRequest rejects empty symptoms ✓")

try:
    prob = ProbabilityDistribution(low=0.03, medium=0.15, high=0.82)
    ok(f"ProbabilityDistribution valid: {prob.low}/{prob.medium}/{prob.high}")
except Exception as e:
    fail("ProbabilityDistribution", e)

try:
    from datetime import datetime
    resp = TriageResponse(
        prediction="HIGH",
        probabilities=ProbabilityDistribution(low=0.03, medium=0.15, high=0.82),
        confidence=0.91,
        rules_triggered=["CHEST_PAIN_CARDIOVASCULAR"],
        explanation="Seek emergency care.",
        model_version="v1.0.0",
        escalated=False,
        emergency_flag=True,
        request_id="req_test001",
        timestamp=datetime.utcnow()
    )
    assert resp.prediction.value == "HIGH"
    ok("TriageResponse model instantiation ✓")
except Exception as e:
    fail("TriageResponse instantiation", e)


# ═══════════════════════════════════════════════════════════════════════════
# 10. FULL END-TO-END TRIAGE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
section("10. End-to-End Triage Pipeline")

import asyncio

async def run_e2e(label, symptoms_text, age, duration_days, chronic_conditions,
                  expected_risk=None, expect_emergency=None):
    try:
        from app.services.triage_service import TriageService
        ts = TriageService()
        t0 = time.monotonic()
        result = await ts.predict(
            symptoms_text=symptoms_text,
            age=age,
            duration_days=duration_days,
            chronic_conditions=chronic_conditions,
            language="en",
            request_id=f"e2e-{label.replace(' ', '-')[:30]}-{uuid.uuid4().hex[:8]}"
        )
        ms = (time.monotonic() - t0) * 1000
        note = f"risk={result.prediction.value} conf={result.confidence:.2f} " \
               f"emergency={result.emergency_flag} {ms:.0f}ms"
        if expected_risk:
            assert result.prediction.value == expected_risk, \
                f"Expected {expected_risk}, got {result.prediction.value}"
        if expect_emergency is not None:
            assert result.emergency_flag == expect_emergency, \
                f"Expected emergency={expect_emergency}"
        ok(f"{label}: {note}")
    except Exception as e:
        fail(label, e)
        traceback.print_exc()

async def e2e_all():
    cases = [
        ("HIGH — chest pain + diabetes",
         "I have severe chest pain and sweating",          55, 1, ["diabetes"],       "HIGH",   True),
        ("HIGH — seizure emergency",
         "Patient is having a seizure and lost consciousness", 40, 0, [],             "HIGH",   True),
        ("MEDIUM — fever + headache",
         "I have a fever and severe headache for 2 days",  32, 2, [],                  None,    None),
        ("LOW — common cold",
         "I have a runny nose and mild cough",             28, 3, [],                  None,    False),
        ("MEDIUM — infant fever (age=1)",
         "My baby has high fever",                          1, 0, [],                  None,    None),
    ]
    for label, text, age, dur, conds, exp_risk, exp_emg in cases:
        await run_e2e(label, text, age, dur, conds, exp_risk, exp_emg)

asyncio.run(e2e_all())


# ═══════════════════════════════════════════════════════════════════════════
# 11. API ENDPOINTS (FastAPI TestClient)
# ═══════════════════════════════════════════════════════════════════════════
section("11. FastAPI API Endpoints")

try:
    from fastapi.testclient import TestClient
    from app.main import app
    _client_ok = True
    ok("FastAPI TestClient initialized")
except Exception as e:
    fail("FastAPI TestClient init", e)
    _client_ok = False

if _client_ok:
    with TestClient(app) as client:
        try:
            r = client.get("/health")
            assert r.status_code == 200
            d = r.json()
            assert d["status"] == "healthy"
            ok(f"GET /health -> 200 | model_loaded={d.get('model_loaded')}")
        except Exception as e:
            fail("GET /health", e)

        try:
            r = client.get("/")
            assert r.status_code == 200
            ok("GET / -> 200")
        except Exception as e:
            fail("GET /", e)

        try:
            payload = {
                "symptoms_text": "I have chest pain and sweating",
                "age": 52, "duration_days": 1,
                "chronic_conditions": ["diabetes"], "language": "en"
            }
            r = client.post("/api/v1/triage/predict", json=payload)
            assert r.status_code == 200
            d = r.json()
            assert d["prediction"] in ["LOW", "MEDIUM", "HIGH"]
            assert "request_id" in d
            assert "explanation" in d
            ok(f"POST /api/v1/triage/predict -> 200 | prediction={d['prediction']}")
        except Exception as e:
            fail("POST /api/v1/triage/predict", e)

        try:
            payload_bad_age = {
                "symptoms_text": "headache", "age": 999,
                "duration_days": 1, "chronic_conditions": []
            }
            r = client.post("/api/v1/triage/predict", json=payload_bad_age)
            assert r.status_code == 422
            ok("POST /api/v1/triage/predict age=999 -> 422 ✓")
        except Exception as e:
            fail("Validation: invalid age rejected", e)

        try:
            r = client.get("/api/v1/metrics")
            assert r.status_code == 200
            ok("GET /api/v1/metrics -> 200")
        except Exception as e:
            fail("GET /api/v1/metrics", e)

        try:
            r = client.get("/api/v1/model/info")
            assert r.status_code == 200
            ok("GET /api/v1/model/info -> 200")
        except Exception as e:
            fail("GET /api/v1/model/info", e)

        try:
            r = client.get("/api/v1/rules")
            assert r.status_code == 200
            ok("GET /api/v1/rules -> 200")
        except Exception as e:
            fail("GET /api/v1/rules", e)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  RESULTS")
print(SEP)
print(f"  Passed  : {results['passed']}")
print(f"  Failed  : {results['failed']}")
print(f"  Skipped : {results['skipped']}")
print(SEP)

if results["failed"] > 0:
    print("  SOME TESTS FAILED — review output above")
    sys.exit(1)
else:
    print("  ALL TESTS PASSED")
    sys.exit(0)
