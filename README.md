Steps to Run the Project
1. Clone the repository
git clone -b training https://github.com/AnjKal/Real-Time-Traffic-Congestion-Pred.git
cd Real-Time-Traffic-Congestion-Pred


2. Install Python 3.7.9

Verify installation:
python --version


3. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1


4. Install dependencies
pip install tensorflow==1.15.0 numpy==1.18.5 pandas==1.1.5 scipy scikit-learn pyyaml matplotlib tables h5py


5. Add the raw dataset

Create data/raw/ and place these two files inside it:
- tomtom_traffic_raw.csv
- bangalore_traffic_spots.csv
6. Generate the 15-minute dataset
python scripts/generate_bangalore_15min_data.py


7. Train the DCRNN model
python dcrnn_train.py --config_filename=data/model/dcrnn_bangalore_15min_smoke.yaml --use_cpu_only


8. Evaluate the model
python evaluate_bangalore.py


9. Compare DCRNN with the persistence baseline
python evaluate_compare_bangalore.py