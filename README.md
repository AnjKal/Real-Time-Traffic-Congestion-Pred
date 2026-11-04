## One-time setup

Install Python 3.7.9 (for the model) and Python 3.10 or newer (for the map page)-https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe](https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe?utm_source=chatgpt.com. Get a TomTom API key.
Clone, then go into the folder:

git clone -b <branch> https://github.com/AnjKal/Real-Time-Traffic-Congestion-Pred.git
cd Real-Time-Traffic-Congestion-Pred
Model environment (Command Prompt):

py -3.7 -m venv .venv37
.venv37\Scripts\activate.bat
python --version
pip install -r requirements.txt
pip install protobuf==3.20.3
pip install -r requirements_pipeline.txt
## Map-page environment (new Command Prompt):

python -m venv .venv-web
.venv-web\Scripts\activate.bat
pip install -r requirements_webapp.txt
python -m webapp.osm_graph
Copy .env.example to .env and paste the TomTom key.
Every run

Terminal 1: .venv37\Scripts\activate.bat, then python start_pipeline.py. Wait for "Model ready".
Terminal 2: .venv-web\Scripts\activate.bat, then python run_webapp.py. Open http://127.0.0.1:8010/.
The first forecast appears after about 24 minutes, and polling stops after 48.
Then, in Terminal 1, type quit and run python evaluate_live.py --step-min 2
