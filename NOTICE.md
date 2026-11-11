# Attribution and external materials

This repository accompanies Roland Thomas Clemson's November 2026 University of Sydney thesis, supervised by Dr Viraj Herath.

The Burnett River hydraulic simulations, geometry, saved partitions, EOF representations and LSG model artifacts are supplied research inputs associated with the Fraehr et al. work cited in the thesis. They are not redistributed here. `gnnsr.lsg` and `gnnsr.lsg_full_domain` provide evaluation adapters for those externally supplied artifacts; this repository does not retrain the LSG model or redistribute the authors' entire repository.

The implementation uses PyTorch, PyTorch Geometric, NumPy, SciPy and the geospatial packages listed in `pyproject.toml`. Their licences apply to those dependencies. Optional LSG inference uses GPflow and TensorFlow.

No new software licence has been selected by this packaging operation. This file is attribution, not a grant of rights or a substitute for a licence. Do not infer that the licence of another student repository applies to this project. Dataset/model access and redistribution permissions must be obtained from their owners.
