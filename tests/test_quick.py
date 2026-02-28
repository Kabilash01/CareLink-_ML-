"""Quick test without loading heavy models."""
import sys
print("Testing imports...")

try:
    from app.core.settings import get_settings
    print("✅ Settings module loaded")
    
    settings = get_settings()
    print(f"✅ Settings initialized: ENV={settings.ENVIRONMENT}")
    print(f"   CORS_ORIGINS: {settings.CORS_ORIGINS}")
    print(f"   SUPPORTED_LANGUAGES: {settings.SUPPORTED_LANGUAGES}")
    
    from app.api.schemas import TriageRequest, TriageResponse
    print("✅ API schemas loaded")
    
    from app.services.rule_engine import RuleEngine
    print("✅ Rule engine loaded")
    
    from app.nlp.text_preprocessor import TextPreprocessor
    print("✅ Text preprocessor loaded")
    
    print("\n✅ All critical modules loaded successfully!")
    print("\n🚀 Service is ready to run. Start with:")
    print("   venv\\Scripts\\activate")
    print("   python -m uvicorn app.main:app --reload")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
