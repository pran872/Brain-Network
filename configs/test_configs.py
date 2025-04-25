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
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/renet_configs/config_resnet_p10.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/renet_configs/config_resnet_p11.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/renet_configs/config_resnet_p12.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p13.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p14.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p15.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/brainit_configs/config_brainit_p16.json",
        "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/brainit_configs/config_brainit_p17.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/brainit_configs/config_brainit_p18.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/renet_configs/config_resnet_p19.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/renet_configs/config_resnet_p20.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/renet_configs/config_resnet_p21.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p22.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p23.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/zoomvit_configs/config_zoomvit_p24.json",
        # "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/brainit_configs/config_brainit_p25.json",
        "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/brainit_configs/config_brainit_p26.json",
        "/Users/pranathipoojary/Imperial/FYP/Brain-Network/configs/stanford_dogs/brainit_configs/config_brainit_p27.json"
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