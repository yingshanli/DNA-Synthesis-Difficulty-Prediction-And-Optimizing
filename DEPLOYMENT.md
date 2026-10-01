# Online deployment (Streamlit Community Cloud)

This folder contains the minimal files required to deploy the DNA synthesis-duration prediction app.

## Files

- `streamlit_app.py` — public Streamlit application
- `xgboost_best_model.pkl` — trained XGBoost pipeline
- `requirements.txt` — Python dependencies, including the ViennaRNA Python interface

## Secondary-structure calculation

The web deployment uses the official ViennaRNA Python interface rather than calling the
`RNAfold` command-line executable. The app loads the built-in **DNA Mathews 2004**
energy parameter set using `RNA.params_load_DNA_Mathews2004()` and calculates the
minimum-free-energy structure at **37 °C**.

This avoids a separate Linux `apt` dependency and keeps the deployment consistent
across local and Streamlit Cloud environments.

## Local test

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run streamlit_app.py
```

Then open the local URL printed by Streamlit.

## Deploy on Streamlit Community Cloud

1. Put these files in the root of a GitHub repository (or copy them into the root of the paper repository).
2. Sign in to Streamlit Community Cloud using GitHub.
3. Create a new app and select the repository.
4. Set the entrypoint to:
   `streamlit_app.py`
5. Choose a Python version compatible with the packages above.
6. Click **Deploy**.

The app can then be assigned a public `*.streamlit.app` URL.

## Publication note

The application predicts synthesis duration from sequence-derived features.
Predictions and synonymous optimization results are computational estimates and
should not be interpreted as experimentally confirmed synthesis performance.
