import os
import sys
import re
from huggingface_hub import HfApi, create_repo

def clean_token(raw_token):
    token = raw_token.strip().strip("'").strip('"')
    match = re.search(r'(hf_[A-Za-z0-9]+)', token)
    if match:
        return match.group(1)
    return token

def main():
    print("=" * 65)
    print("   SENTINEL AUDIO - PERMANENT CLOUD DEPLOYMENT (HUGGING FACE)")
    print("=" * 65)
    print()
    print("This script will upload your full-stack website to a permanent")
    print("24/7 cloud server with 16 GB RAM (100% Free on Hugging Face Spaces).")
    print()
    
    while True:
        raw_token = input("1. Paste your Hugging Face Token (starts with hf_...): ").strip()
        hf_token = clean_token(raw_token)
        
        if not hf_token:
            print("[ERROR] Token cannot be empty. Please paste your token.")
            continue

        api = HfApi(token=hf_token)
        try:
            user_info = api.whoami()
            username = user_info["name"]
            print(f"\n✅ Successfully authenticated as @{username}!")
            break
        except Exception as e:
            print(f"\n❌ Authentication failed: {e}")
            print("Troubleshooting tips:")
            print("  • Go to: https://huggingface.co/settings/tokens")
            print("  • Click 'Create new token' -> choose 'Write' permissions.")
            print("  • Make sure to copy the entire string starting with 'hf_...'")
            print()
            retry = input("Do you want to try pasting the token again? (y/n): ").strip().lower()
            if retry != 'y':
                return

    # Ask for Space Name
    default_name = "ai-voice-scam-detector"
    custom_name = input(f"\n2. Enter Space name (press Enter for '{default_name}'): ").strip()
    repo_name = custom_name if custom_name else default_name
    
    # Format valid repo name (only letters, numbers, hyphens)
    repo_name = re.sub(r'[^a-zA-Z0-9_-]', '-', repo_name)
    repo_id = f"{username}/{repo_name}"

    # Create Space if it does not exist
    print(f"\nCreating/Verifying Docker Space: {repo_id}...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            token=hf_token,
            exist_ok=True
        )
        print(f"[OK] Space '{repo_id}' is ready.")
    except Exception as e:
        print(f"Note: {e}")

    # Upload directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nUploading website files and AI models to {repo_id}...")
    print("This will take around 1-3 minutes depending on your internet connection...")
    
    try:
        api.upload_folder(
            folder_path=current_dir,
            repo_id=repo_id,
            repo_type="space",
            token=hf_token,
            ignore_patterns=["__pycache__", "*.pyc", ".git", ".venv", "venv", "env", "deploy_to_hf.py", "scratch", ".system_generated"]
        )
        print("\n" + "=" * 65)
        print("🎉 SUCCESS! YOUR WEBSITE IS DEPLOYED!")
        print("=" * 65)
        print(f"\nYour Permanent Space Link:")
        print(f"👉 https://huggingface.co/spaces/{repo_id}")
        print(f"\nDirect Full-Screen Web App Link:")
        print(f"👉 https://{username}-{repo_name.replace('_', '-')}.hf.space")
        print("=" * 65)
    except Exception as e:
        print(f"\n[ERROR] Upload failed: {e}")

if __name__ == "__main__":
    main()
