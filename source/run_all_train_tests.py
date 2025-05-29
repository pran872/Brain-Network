import json
import argparse
from argparse import Namespace

try:
    import train
    import test
except ModuleNotFoundError:
    import source.train
    import source.test

def parse_args():
    parser = argparse.ArgumentParser(description="Pass a config file or enter debug mode")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=False, 
        default=None,
        help="Config file. If not provided, defaults will be used."
    )
    parser.add_argument(
        "-d", "--debug", 
        action="store_true",
        required=False,
        help="Run on debug mode"
    )
    parser.add_argument(
        "--run_all_tests",
        action="store_true",
        required=False,
        help="Runs accuracy tests," \
                "sample efficiency tests: 0.1, 0.25, 0.5," \
                "and robustness tests: FGSM, PGD, gaussian noise with epsilon [0.01, 0.05, 0.1, 0.2]"
    )
    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, TypeError) as e:
        config = {}
    
    return config, args.debug, args.run_all_tests

def main():
    config, debug, run_all_tests = parse_args()

    tests_to_run = {"acc": [True]}
    # tests_to_run = {}
    if run_all_tests:
        tests_to_run["sample_eff"] = [0.1, 0.25, 0.5]
        tests_to_run["robustness"] = [0.01, 0.05, 0.1, 0.2] # Values arent used btw
    for test_type, values in tests_to_run.items():
        if test_type == "acc":
            config["run_name"] += "_acc"
            args_to_parse = [config, debug]
            log_dir = train.main(args_to_parse)
            config["run_name"] = config["run_name"].replace("_acc", "")
        elif test_type == "sample_eff":
            for val in values:
                suffix = f"_sample_eff_{str(val).replace('.', '_')}"
                config["run_name"] += suffix
                config["dataset"]["downsample_fraction"] = val
                args_to_parse = [config, debug]
                train.main(args_to_parse)
                config["run_name"] = config["run_name"].replace(suffix, "")
        elif test_type == "robustness":
            args = Namespace(
                run_folder=log_dir,
                config=None,
                model=None,
                debug=debug,
                run_default_attacks=True
            )
            args_to_parse = args, debug, True
            test.main(args_to_parse)


if __name__ == "__main__":
    main()