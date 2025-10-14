# WFP Campaign Dashboard

This is a Streamlit app for tracking contact penetration by district and pass.

## Files
- `wfp_campaign_dashboard_app.py` — the Streamlit app
- `wfp_dashboard_template.csv` — plug-and-play CSV template
- `requirements.txt` — Python dependencies
- `README.md` — this file with setup instructions

## Quick Start

1. **Create & activate a virtual environment (recommended)**
   - macOS / Linux
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell)
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate
     ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run wfp_campaign_dashboard_app.py
   ```
   If needed, specify a port:
   ```bash
   streamlit run wfp_campaign_dashboard_app.py --server.port 8502
   ```

4. **Use the dashboard**
   - Upload your CSV (start with `wfp_dashboard_template.csv` to verify)
   - Filter by District, Pass, and Contact Method
   - Download processed tables from the buttons at the bottom

## CSV Schema

Required header (no extra columns needed):
```
date,campaign_name,district,precinct_id,precinct_name,pass_number,contact_method,attempts,target_attempts
```

- `date` must be YYYY-MM-DD
- `district` is an integer 1–12
- `pass_number` is an integer
- `contact_method` (e.g., Door, Phone, Text)
- `attempts` numeric
- `target_attempts` numeric (max per district+pass+precinct)

## Tier Mapping (auto-assigned)
- **Tier 1:** districts 2, 5, 7, 8
- **Tier 2:** districts 3, 4, 6, 9, 10, 11, 12
- Anything else (e.g., District 1) → Unassigned

## Troubleshooting
- **File not found**: Make sure you run `streamlit run` from the same folder or pass the full path.
- **ModuleNotFoundError**: Install deps with `pip install -r requirements.txt`.
- **Port busy**: Use `--server.port 8502`.
