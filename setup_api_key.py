#!/usr/bin/env python3
import os
import json
import getpass

def setup_api_key():
    """Interactive script to set up Claude API key."""
    print("Claude API Key Setup")
    print("====================")
    print("Your API key will be saved in config.json")
    print("You can also set it as an environment variable CLAUDE_API_KEY")
    print()
    
    api_key = getpass.getpass("Enter your Claude API key: ")
    
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    config = {}
    
    # Try to load existing config if it exists
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read existing config: {e}")
    
    # Update config with new API key
    config["api_key"] = api_key
    
    # Save config
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nAPI key saved to {config_path}")
        print("\nYou can now use the tool with:")
        print("  python src/main.py --data YOUR_DATA_FILE --instructions \"Your analysis instructions\"")
    except Exception as e:
        print(f"Error saving config: {e}")
        print("As an alternative, you can set the CLAUDE_API_KEY environment variable:")
        print("  export CLAUDE_API_KEY='your_api_key'")

if __name__ == "__main__":
    setup_api_key()