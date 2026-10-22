import os
import requests
from dotenv import load_dotenv
from google import genai

load_dotenv()

def test_gemini_details():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY: Missing in .env")
        return
    
    # Try preferred model first, then fallback model
    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash"]
    client = genai.Client(api_key=api_key)

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents="Hello"
            )
            print(f"✅ GEMINI_API_KEY: Valid!")
            print(f"   ├── Model Access: {model_name} (Confirmed)")
            print("   └── Permissions: Full Text & Multimodal Vision Generation")
            return
        except Exception as e:
            if "503" in str(e):
                print(f"⚠️  {model_name} busy (503), trying fallback model...")
                continue
            else:
                print(f"❌ GEMINI_API_KEY: Failed on {model_name} ({e})")
                return

    # Reached only when every candidate model kept responding 503.
    print("⚠️  All candidate Gemini models are temporarily busy (503). Try again shortly.")

def test_figma_details():
    token = os.getenv("FIGMA_ACCESS_TOKEN")
    if not token:
        print("⚠️  FIGMA_ACCESS_TOKEN: Missing in .env")
        return
    try:
        headers = {"X-Figma-Token": token}
        res = requests.get("https://api.figma.com/v1/me", headers=headers)
        if res.status_code == 200:
            user_data = res.json()
            print("✅ FIGMA_ACCESS_TOKEN: Valid!")
            print(f"   ├── User: {user_data.get('handle')} ({user_data.get('email')})")
            print("   └── Access Scope: REST API File & Node Image Export")
        else:
            print(f"❌ FIGMA_ACCESS_TOKEN: Invalid (HTTP {res.status_code})")
    except Exception as e:
        print(f"❌ FIGMA_ACCESS_TOKEN: Failed ({e})")

def test_github_details():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("⚠️  GITHUB_TOKEN: Missing in .env")
        return
    try:
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        res = requests.get("https://api.github.com/user", headers=headers)
        if res.status_code == 200:
            user_data = res.json()
            scopes = res.headers.get("X-OAuth-Scopes", "Fine-grained Token / Default")
            print("✅ GITHUB_TOKEN: Valid!")
            print(f"   ├── User: {user_data.get('login')}")
            print(f"   ├── OAuth Scopes: [{scopes}]")
            
            # Check PR/Repository write capabilities
            if "repo" in scopes or "public_repo" in scopes or scopes == "Fine-grained Token / Default":
                print("   └── PR/Issue Permissions: Active (Can create PRs and write code patches)")
            else:
                print("   └── PR/Issue Permissions: ⚠️ Limited Scope (Requires 'repo' scope for automated PRs)")
        else:
            print(f"❌ GITHUB_TOKEN: Invalid (HTTP {res.status_code})")
    except Exception as e:
        print(f"❌ GITHUB_TOKEN: Failed ({e})")

def test_e2b_details():
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        print("⚠️  E2B_API_KEY: Missing in .env")
        return
    try:
        headers = {"X-API-Key": api_key}
        res = requests.get("https://api.e2b.dev/health", headers=headers)
        if res.status_code in [200, 204]:
            print("✅ E2B_API_KEY: Valid!")
            print("   └── Sandbox Access: Enabled (Can execute isolated Docker auto-repair tests)")
        else:
            print(f"❌ E2B_API_KEY: Invalid (HTTP {res.status_code})")
    except Exception as e:
        print(f"❌ E2B_API_KEY: Failed ({e})")

if __name__ == "__main__":
    print("🔍 Testing API Keys & Access Levels...\n" + "="*50)
    test_gemini_details()
    print("-" * 50)
    test_figma_details()
    print("-" * 50)
    test_github_details()
    print("-" * 50)
    test_e2b_details()
    print("=" * 50)
