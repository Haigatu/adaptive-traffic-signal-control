# Adaptive Traffic Signal Control — Results

Open **Results_Evidence.ipynb** in JupyterLab or VS Code. All tables and figures are saved. **Results_Evidence.html** provides a browser-readable copy.

The notebook reconstructs results from 258 saved test runs, checks metric definitions and paired comparisons, and recomputes two forecasting baselines from PeMS08 data. It analyses completed experiments; it does not rerun training or SUMO.

## Run the analysis

Open a terminal in this folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m jupyterlab Results_Evidence.ipynb
```

Select **Restart Kernel → Run All**, or execute individual cells with Shift+Enter. Python 3.11+ is required. Internet access is needed to install dependencies, but not to run the analysis. SUMO and PyTorch are not required for this notebook.

Keep `evidence/` beside the notebook. It contains the original experiment artifacts and source snapshots; these are preserved in their original form. `audit_tables/` contains exported analysis tables.
