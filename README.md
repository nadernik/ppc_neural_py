# Python Codes - Neuron Analysis
## Structure

```
Python Codes/
├── requirements.txt                    # Python dependencies
├── Neuron Analysis/                    # Main analysis package
│   ├── __init__.py
│   ├── Neurons/                       # Core neuron analysis functions
│   │   ├── __init__.py
│   │   ├── spike_analysis.py          # Spike analysis and PSTH
│   │   ├── psth_analysis.py           # PSTH plotting and visualization
│   │   └── tdt_utils.py               # TDT data utilities
│   ├── TDT/                           # TDT data processing
│   │   ├── __init__.py
│   │   ├── tdt_reader.py              # TDT data reading functions
│   │   └── tdt_processing.py          # TDT data processing and filtering
│   ├── dPCA-master/                   # Demixed PCA implementation
│   │   ├── __init__.py
│   │   └── dpca.py                    # Main dPCA algorithm
│   ├── tSNE_matlab/                   # tSNE dimensionality reduction
│   │   ├── __init__.py
│   │   └── tsne.py                    # tSNE implementation
│   └── NeurometricExample/            # Neurometric analysis examples
│       ├── __init__.py
│       └── neurometric_example.py     # ROC analysis examples
└── demo.py                            # Demo script
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. For TDT data processing, you may need the TDT Python SDK:
```bash
pip install tdt
```

## Key Features

### Spike Analysis (`Neurons/spike_analysis.py`)
- `align_spikes_to()`: Align spikes to events and create spike matrices
- `get_psth()`: Calculate Peri-Stimulus Time Histograms with smoothing
- `get_roc()`: Calculate ROC curves and AUC
- `get_selectivity()`: Calculate selectivity indices
- `get_tuning()`: Calculate tuning curves

### PSTH Analysis (`Neurons/psth_analysis.py`)
- `plot_spike_psth()`: Plot PSTH with raster plots
- `plot_spike_raster()`: Plot spike rasters
- `plot_population_response()`: Plot population responses
- `plot_tuning_curve()`: Plot tuning curves
- `plot_roc_curve()`: Plot ROC curves

### TDT Data Processing (`TDT/`)
- `tdt2mat()`: Read TDT tank data
- `sev2mat()`: Read SEV files
- `tdt_filter()`: Filter TDT data
- `ecog_tdt_data2matlab()`: Process ECoG data

### dPCA (`dPCA-master/dpca.py`)
- `dpca()`: Main dPCA algorithm
- `dpca_explained_variance()`: Calculate explained variance
- `dpca_plot()`: Plot dPCA results

### tSNE (`tSNE_matlab/tsne.py`)
- `tsne()`: tSNE dimensionality reduction
- `x2p()`: Convert distances to probabilities
- `tsne_d()` and `tsne_p()`: Simplified tSNE variants

### Neurometric Analysis (`NeurometricExample/`)
- `run_neurometric_analysis()`: Complete neurometric analysis
- `calculate_roc_curve()`: Calculate ROC curves
- `plot_neurometric_function()`: Plot neurometric functions

## Usage Examples

### Basic Spike Analysis
```python
import numpy as np
from Neuron_Analysis.Neurons.spike_analysis import align_spikes_to, get_psth

# Example data
spike_times = np.random.exponential(100, 1000)
event_times = np.arange(0, 10000, 1000)

# Align spikes to events
spike_matrix, aligned_spikes = align_spikes_to(
    spike_times, event_times, pre=-500, post=1000, bin_size=10
)

# Calculate PSTH
psth, sem = get_psth(spike_matrix, bin_size=10, w=50, kernel='gauss')
```

### PSTH Plotting
```python
from Neuron_Analysis.Neurons.psth_analysis import plot_spike_psth

# Plot PSTH with raster
fig, axes = plot_spike_psth(
    spike_times, event_times, 
    pre=-500, post=1000, 
    title="Example PSTH"
)
```

### TDT Data Processing
```python
from Neuron_Analysis.TDT.tdt_reader import tdt2mat
from Neuron_Analysis.TDT.tdt_processing import tdt_filter

# Read TDT data
tdt_data = tdt2mat("MyTank", "Block-1")

# Filter data
filtered_data = tdt_filter(
    tdt_data['streams']['LFP1']['data'],
    fs=1000,
    low_freq=1,
    high_freq=100,
    filter_type='bandpass'
)
```

### dPCA Analysis
```python
from Neuron_Analysis.dPCA_master.dpca import dpca, dpca_plot

# Example data (neurons x conditions x time)
X = np.random.randn(50, 8, 100)

# Run dPCA
W, V, which_marg = dpca(X, num_comps=5)

# Plot results
dpca_plot(X, W, V, which_marg, plot_type='traces')
```

### tSNE Dimensionality Reduction
```python
from Neuron_Analysis.tSNE_matlab.tsne import tsne

# Example data
X = np.random.randn(1000, 100)

# Run tSNE
Y = tsne(X, no_dims=2, perplexity=30)
```

### Neurometric Analysis
```python
from Neuron_Analysis.NeurometricExample.neurometric_example import run_neurometric_analysis

# Run complete analysis
run_neurometric_analysis()
```

## Contributing

When adding new functions or modifying existing ones:
1. Follow the existing code style
2. Add comprehensive docstrings
3. Include type hints
4. Test with example data

## License

This code is based on the original implementation by Nader Nikbakht. 