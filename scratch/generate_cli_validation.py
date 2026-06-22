import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from run_experiment3 import CLIWrapper

def generate_validation():
    print("Initializing CLI Wrapper without clear screen...")
    cli = CLIWrapper(clear_screen=False)
    queries = ["Burning Umbrella", "Operation Kitty", "Lazarus Cryptocurrency"]
    
    rep_path = r"C:\Users\atmak\.gemini\antigravity-ide\brain\21851985-707e-4cab-bf5e-714d21b4905d\production_cli_validation.md"
    
    # Keep original stdout
    original_stdout = sys.stdout
    
    print("Executing queries and routing output to markdown artifact...")
    with open(rep_path, "w", encoding="utf-8") as f:
        sys.stdout = f
        
        print("# PRODUCTION CLI VALIDATION REPORT\n")
        
        for q in queries:
            cli.execute_and_print(q)
            print("\n")
            
        sys.stdout = original_stdout
        
    print(f"Validation complete. File saved to: {rep_path}")

if __name__ == "__main__":
    generate_validation()
