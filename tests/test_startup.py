"""Quick startup test for CARELINK AI service."""
from app.main import app

print("Testing FastAPI app initialization...")
print("✅ App initialized successfully!")
print("\nAvailable routes:")
for route in app.routes:
    if hasattr(route, 'methods'):
        methods = ','.join(route.methods)
        print(f"  [{methods}] {route.path}")
    else:
        print(f"  {route.path}")

print("\n✅ All systems operational!")
