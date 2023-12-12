# Detection Phishing Messages Training Pipeline

### Installing and Running

#### 1. Clone repository
`git clone https://github.com/frek17/phishing-ml-pipeline.git`

#### 2. Download Python 3.8.10
Example, for MacOS - https://www.python.org/downloads/macos/

#### 3. Open terminal in repo directory, create virtual environment from downloaded python and run
`pip install -r requirements.txt`

#### 4. Model parameters
Select proper model parameters in `config.py`

#### 5. Run pipeline
`python train_ml_model.py`

#### 6. Final model and metrics
If all steps were successful you have your trained model in `<model_path>` and metric for test data in log file or in `<model_path/README.md -- model-index -- metrics>`