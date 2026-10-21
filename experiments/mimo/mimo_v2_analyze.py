"""Import-safe entry point; scientific script retained in experiments/_legacy/."""


def run(config=None, argv=None):
    from experiment_runtime import run_script

    return run_script("mimo_v2_analyze", config=config, argv=argv)


if __name__ == "__main__":
    import sys

    run(argv=sys.argv[1:])
