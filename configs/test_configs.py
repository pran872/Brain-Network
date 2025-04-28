import sys
import os
import glob
import subprocess

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

errored_out = dict()

def run_tests():
    # dir = "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/round_0_seed_tests"
    
    pts = [
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p43.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p44.json",
        "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p45.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p46.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p47.json",
        "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p48.json",
        "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p51.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p52.json",
    ]
    # for config in glob.glob(os.path.join(dir, "**", "*.json"), recursive=True):
    for config in pts:
        print(f"\nTesting config: {config}")
        try:
            result = subprocess.run(
                ["python", "source/simple_cnn.py", "--config", config, "-d"],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            errored_out[config] = e
            print(f"Error in config: {config}")
            print("stdout:", e.stdout)
            print("stderr:", e.stderr)
            continue
        print("Done")

    print("Errored out")
    for key, value in errored_out.items():
        print()
        print(key)
        print(value)

if __name__ == "__main__":
    run_tests()