"""Provides a simple command-line wrapper to reproduce key
experiments."""
import os
import sys
import argparse

from antibody_scoring.data_prep import data_retrieval as dataretr
from antibody_scoring import evaluations


class ReconfigParser(argparse.ArgumentParser):
    """Reconfigure argparse's parser so an automatic error message
    is generated if no args supplied."""
    def error(self, message):
        self.print_help()
        sys.exit(2)


def gen_arg_parser():
    """Build the command line arg parser."""
    parser = ReconfigParser(description="Use this command line app to "
                "run the experiments described in the paper and reproduce key "
                "results. Output is printed to the console and written to "
                "the log files under results.")
    parser.add_argument("--retrieve_data", action="store_true",
            help="Retrieve the raw data from Zenodo and other resources")
    parser.add_argument("--engelhart", action="store_true",
            help="Run evals on the data from Engelhart et al.")
    parser.add_argument("--mason", action="store_true",
            help="Run evals on the data from Mason et al.")
    parser.add_argument("--desautels", action="store_true",
            help="Run evals on the data from Desautels et al.")
    parser.add_argument("--cognano", action="store_true",
            help="Run evals on the cognano dataset.")
    parser.add_argument("--il6", action="store_true",
            help="Run evals on the il6 dataset.")
    return parser



def get_raw_data(project_dir):
    """Retrieve data from links from which it was originally downloaded (for
    datasets which are not included)."""
    os.chdir(project_dir)
    dataretr.retrieve_engelhart_dataset(project_dir)
    #dataretr.retrieve_desautels_dataset(project_dir)
    dataretr.retrieve_mason_dataset(project_dir)
    #dataretr.retrieve_cognano_dataset(project_dir)
    dataretr.retrieve_il6_dataset(project_dir)


def main():
    """Entry point for all dataset building tasks."""
    home_dir = os.path.dirname(os.path.abspath(__file__))
    parser = gen_arg_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.retrieve_data:
        get_raw_data(home_dir)

    if args.engelhart:
        evaluations.engelhart_eval(home_dir)

    if args.mason:
        evaluations.mason_eval(home_dir)

    if args.desautels:
        evaluations.desautels_eval(home_dir)

    if args.cognano:
        evaluations.cognano_eval(home_dir)

    if args.il6:
        evaluations.il6_eval(home_dir)


if __name__ == "__main__":
    main()
