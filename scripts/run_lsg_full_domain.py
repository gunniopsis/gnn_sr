"""Evaluate the supplied Burnett fold-1 LSG; never fit model parameters."""

import argparse

from gnnsr.lsg_full_domain.evaluation import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(run(resume=args.resume))
