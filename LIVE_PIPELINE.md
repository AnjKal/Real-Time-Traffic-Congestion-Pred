# Run the live congestion forecast UI

Run every command from the repository root, in Command Prompt (not PowerShell).

## One-time setup

1. Install **64-bit Python 3.7.9** (for the model) and **any Python 3.10 or newer** (for the map page).
   Get a free TomTom API key (https://developer.tomtom.com).
2. Clone and enter the folder:

       git clone -b live_data https://github.com/AnjKal/Real-Time-Traffic-Congestion-Pred.git
       cd Real-Time-Traffic-Congestion-Pred

3. Model environment (Python 3.7):

       py -3.7 -m venv .venv37
       .venv37\Scripts\activate.bat
       pip install -r requirements.txt
       pip install -r requirements_pipeline.txt
       deactivate

4. Map environment (Python 3.10+; use the version you installed, e.g. `py -3.11`):

       py -3.11 -m venv .venv-web
       .venv-web\Scripts\activate.bat
       pip install -r requirements_webapp.txt
       python -m webapp.osm_graph
       deactivate

   The last command downloads the road map once (a few minutes, needs internet) and saves it to
   `data\graph\bangalore_core.pkl`. If it fails with a 502 ("at capacity"), run it again. It never
   needs repeating.
5. Create your key file: `copy .env.example .env`, then open `.env` and put your own TomTom key
   after `TOMTOM_API_KEY=`. Never commit `.env`.

## Every run (two Command Prompt windows)

- Window 1:

      .venv37\Scripts\activate.bat
      python start_pipeline.py

  Wait for `Model ready`.
- Window 2:

      .venv-web\Scripts\activate.bat
      python run_webapp.py

  Open http://127.0.0.1:8010/

## What to expect

- Pipeline test mode: readings every 2 minutes, so the first forecast appears after about 24 minutes
  (12 readings). The model was trained on 15-minute readings, so this checks the pipeline, not accuracy.
- Polling stops by itself after 48 minutes (2,400 TomTom calls, under the free limit of 2,500 a day).
- Type `quit` in Window 1 to stop.
- Ports 8001 and 8010 must be free.
- "Compute Routes" says the road map has not been built if step 4 was skipped.
