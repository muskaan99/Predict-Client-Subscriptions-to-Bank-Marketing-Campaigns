
The main.py file is used to run the best models from Supervised Learning and Semi Supervised Learning experiments respectively. The original dataset can be found at data\Bank_dataset\bank-additional-full.csv. To run the main.py file, the test data needs to be loaded from the /data folder. The requirements.txt file lists all Python libraries that the main.py mode depends on.

## Steps to run main.py file

1) Use the command:
`pip install -r requirements.txt`

2)use the command `python3 main.py` to run the main.py file.


The Bank_Marketing.ipynb file was written using the Colab IDE. It contains the Exploratory Data Analysis, training and testing codes.

The SL.py file contains the following Supervised Learning Algorithms: Trivial model, Baseline Model, Logistic Regression, Decision Tree, SVC, MLP and Random Forest Algorithms.

The SSL.py file contains the Semi Supervised Learning Algorithms: Prop 1NN, Expectation Maximization and Label Propagation. The experiments have been performed for 10%,20%, 30% and 40% of labeled data. Since the model file for S3VM is approx 1.5GB, we have attached the pickle file of the model for only 40% labeled data in the folder \models\semi_supervised_models\40%_lab_data.

The pickle files for all the models can be found under the model folder which is subdivided into supervised and semi supervised categories.

The code for S3VM has been taken from the following github repository: https://github.com/NekoYIQI/QNS3VM.