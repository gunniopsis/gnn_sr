# Method and evaluation protocol

## Model

Each timestep is an independent supervised example. The network predicts a standardized residual correction to coarse WSE projected onto fine cells. Coarse and fine graphs have bidirectional hydraulic/spatial neighbour edges; a directed containing-parent link transfers coarse information to each fine node, with the original nearest-parent fallback retained where necessary.

The encoder-processor-decoder uses two-linear-layer MLPs, sum aggregation, residual node updates, LayerNorm and LeakyReLU. Width sets both latent dimension and MLP hidden-layer size. The adopted model has width 64, two coarse layers, one coarse-to-fine fusion stage, two fine layers and 140,481 trainable parameters.

## Training

- Fold 1 and seed 42; read the original saved partition without rebuilding it.
- Fit input/output normalization using training data only, then freeze it for validation and testing.
- Adam, initial learning rate 0.001, zero weight decay, no dropout and gradient clipping at 1.0.
- Validate every epoch and replace `best.pt` only on strictly lower validation depth RMSE.
- ReduceLROnPlateau: patience 10, factor 0.5, minimum learning rate 0.00001.
- Early stopping: 30 consecutive non-improving epochs.
- Full-domain training: maximum 100 epochs, no epoch-cap extension. Save a complete checkpoint every epoch, plus `best.pt` and `latest.pt`.
- AOI stages retain their original longer epoch limits and improvement-based cap extension. The full-domain 100-epoch limit does not apply to those searches.

The composite objective adds a differentiable soft-wetness extent term to standardized residual MSE. Temperature controls the smoothness around the fixed 0.03 m wet/dry threshold. The adopted values are T=0.3 m and lambda=0.01. Calibration uses training samples only; it neither prunes candidates nor selects on held-out performance.

## Spatial and temporal support

The 1 km AOI has 2,314 fine target cells. Width and loss stages retain their original two-coarse/three-fine-hop support. The depth sweep uses a common four-coarse/six-fine-hop graph with 362 coarse nodes and 3,271 fine nodes. Only unchanged AOI target cells contribute to loss and metrics.

Full-domain training uses all 780,785 fine cells. The main LF/GNN-SR/LSG comparison uses the original fold-1 mask with 769,709 cells, joined by global cell identifiers. It is a supplied training-derived mask, not a mask fitted to predictions or held-out errors. All 18 held-out events retain 383 hourly timesteps, hours 48-430.

The detailed event-16 depth-distribution and hourly RMSE analysis additionally excludes cells that remain dry in HF, LF and GNN-SR throughout the event. The original study's resulting wet union had 609,316 cells; this is an observed result, not a selection constraint imposed on a new model.

## Report metrics and baselines

Use the unchanged functions in `src/gnnsr/metrics.py` for AvgRMSE, FI, peak-depth R2, AvgPeakDiff and maximum-extent CSI. The report tables also include POD, RFA and timing of maximum simultaneous inundation area. AvgPeakDiff uses reference minus prediction, so positive means underprediction. The detailed signed GIS/histogram exports use prediction minus reference, so positive means overprediction; their metadata records that formula. Wet means depth at least 0.03 m after physical output processing.

LF is the original HEC-RAS WSE projected through the same saved parent mapping and fine terrain as the GNN inputs. It is not a newly executed hydraulic simulation. LSG's five principal comparison metrics come from the supplied `Validation_results.npz`; additional saved-LSG inference provides predictions, extent counts and area timing. Keep measured inference times separate from rescoring and file-export times.

The 2/2 depth choice is a computational tradeoff documented in the report; the AOI 4/4 configuration had the lowest validation RMSE. Do not reinterpret test performance as the model-selection criterion.
