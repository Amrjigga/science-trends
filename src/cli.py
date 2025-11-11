# Command-line interface for running the complete science trends pipeline
# Executes all modules in sequence: preprocess , embed  , cluster , label , name , aggregate , visualize

import subprocess, sys, os

# Run the complete pipeline in the correct order
def run():
    python_exe = sys.executable  
    
    steps = [
        "preprocess.py",
        "embed.py", 
        "cluster.py",
        "label_topics.py",
        "name_topics.py",          
        "aggregate_trends.py",
        "visualize.py",
    ]
    
    for step in steps:
        print(f"Running {step}...")
        subprocess.check_call([python_exe, os.path.join("src", step)])

if __name__ == "__main__":
    run()
