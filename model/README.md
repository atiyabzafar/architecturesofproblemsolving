# Running the model 

This file is created as a guideline for using the model. Follow the instructions given in this file to perform simulations for the models and generate plots used in the publication. 

## Setup

Create and activate a virtual environment, then install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Folder setup

Put everything in the samemodel folder and run from there

## Order of running

```
0  py "261001 analysis.py" networks                       (optional: prints the network table and KIN4)

1  SIMULATIONS
1a py "261001 analysis.py" run
1b py "261109 results runs.py" "output/261109 results" all
1c py "261109 frustration.py" > "output/261109 results/frustration.txt"

2  FIGURES AND NUMBERS
2a py "261109 results figures.py" "output/261109 results" "output/261001 timeseries.npz" > "output/261109 results/results_numbers.txt"
2b py "261111 fig_s6_meanfield.py" "output/261109 results"
2c py "261001 analysis.py" figures

```

