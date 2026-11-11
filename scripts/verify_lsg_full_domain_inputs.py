"""Hash all inputs once after transfer; submission uses the cached receipt."""

from gnnsr.lsg_full_domain.evaluation import ROOT, verify

if __name__ == "__main__":
    print(verify(ROOT, full_hash=True))
    print("All Burnett fold-1 LSG inputs verified. No fitting or evaluation performed.")
