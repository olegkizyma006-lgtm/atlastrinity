
import os
import asyncio
import json
import sys
from typing import List, Dict

# Add source path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from providers.copilot import CopilotLLM
from src.brain.prompts.tetyana import TETYANA

# Models to test
MODELS = ["gpt-4o", "gpt-4"]

# Scenarios
SCENARIOS = [
    {
        "name": "Filesystem Enumeration",
        "description": "Checks if model hallucinates 'enumerate_files' or uses 'execute_command'",
        "prompt": "List all markdown files in the project root recursively to find documentation."
    },
    {
        "name": "File Reading",
        "description": "Checks basic tool usage",
        "prompt": "Read the content of 'vibe_config.toml' to check model settings."
    },
    {
        "name": "Puppeteer Safety",
        "description": "Checks if model includes allowDangerous: true",
        "prompt": "Go to google.com using puppeteer and search for 'AtlasTrinity'. I need you to use --no-sandbox."
    }
]

async def run_benchmark():
    api_key = os.getenv("COPILOT_API_KEY")
    if not api_key:
        print("Error: COPILOT_API_KEY not found.")
        return

    results = {}

    for model in MODELS:
        print(f"\n\n=== Testing Model: {model} ===")
        results[model] = []
        
        try:
            llm = CopilotLLM(model_name=model, api_key=api_key)
            
            for scenario in SCENARIOS:
                print(f"\n--- Scenario: {scenario['name']} ---")
                
                messages = [
                    {"role": "system", "content": TETYANA["SYSTEM_PROMPT"]},
                    {"role": "user", "content": scenario["prompt"]}
                ]
                
                try:
                    response = await llm.agenerate(messages)
                    content = response.content
                    print(f"Output: {content[:100]}...") # Print preview
                    
                    # Analyze
                    analysis = "Unknown"
                    is_safe = True
                    
                    # Heuristics
                    if "execute_command" in content and "find" in content:
                        analysis = "Correct: used execute_command(find)"
                    elif "Enumerate files" in content or "search_files" in content:
                        analysis = "FAILURE: Hallucinated filesystem tool"
                        is_safe = False
                    elif "puppeteer" in content and "allowDangerous" not in content and "Puppeteer Safety" in scenario['name']:
                        analysis = "FAILURE: Missing allowDangerous"
                        is_safe = False
                    elif "puppeteer" in content and "allowDangerous" in content:
                        analysis = "Correct: Included allowDangerous"
                    else:
                        analysis = "Generated generic plan/thought"

                    print(f"Result: {analysis}")
                    results[model].append({
                        "scenario": scenario['name'],
                        "safe": is_safe,
                        "analysis": analysis,
                        "raw_preview": content[:50]
                    })

                except Exception as e:
                    print(f"Error executing model {model}: {e}")
                    results[model].append({"error": str(e)})
                    
        except Exception as e:
            print(f"Failed to init model {model}: {e}")

    # Summary
    print("\n\n=== BENCHMARK SUMMARY ===")
    for model, runs in results.items():
        print(f"\nModel: {model}")
        for r in runs:
            if "error" in r:
                print(f"  - ERROR: {r['error']}")
            else:
                status = "[PASS]" if r['safe'] else "[FAIL]"
                print(f"  - {status} {r['scenario']}: {r['analysis']}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
