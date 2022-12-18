
## Details about Bank_Marketing.ipynb file
The Bank_Marketing.ipynb file was written using the Colab IDE. It contains the Exploratory Data Analysis, training and testing codes for both Supervised and Semi Supervised Learning Algorithms.

### The following Supervised Algorithms were implemented with the help of Grid Search:
- Trivial Baseline
- Non Trivial Baseline
- Logistic Regression
- Support Vector Machine
- Decision Tree Classifier
- Multi Layer Perceptron
- Random Forest Classifier

The SL.py file contains the following Supervised Learning Algorithms: Trivial model, Baseline Model, Logistic Regression, Decision Tree, SVC, MLP and Random Forest Algorithms.

### The following Semi-Supervised Algorithms were implemented:
- Propagating Nearest Neighbour
- Label Propagation
- Expectation Maximization
- Semi Supervised Support Vector Machine(S3VM)

The SSL.py file contains the Semi Supervised Learning Algorithms: Prop 1NN, Expectation Maximization and Label Propagation. The experiments have been performed for 10%,20%, 30% and 40% of labeled data. Since the model file for S3VM is approx 1.5GB it isn't available in the model files folder.

### The notebook also contains experiments conducted for:
i) Noisy train data.

ii) Logistic regression model for both Supervised and Semi Supervised Learning techniques.

## Performance Metrics
Specificity and Sensitivity
 
## Details about main.py file
The main.py file is used to run the best models from Supervised Learning and Semi Supervised Learning experiments respectively. The original dataset can be found at data\Bank_dataset\bank-additional-full.csv. To run the main.py file, the test data needs to be loaded from the /data folder. The requirements.txt file lists all Python libraries that the main.py mode depends on.

## Steps to run main.py file

1) Use the command:
`pip install -r requirements.txt`

2) Use the command `python3 main.py` to run the main.py file.


The pickle files for all the models can be found under the saved_model_files folder which is subdivided into supervised and semi supervised categories.

The code for S3VM has been taken from the following github repository: https://github.com/NekoYIQI/QNS3VM.

## Project Report
The project report can be found [here](https://github.com/muskaan99/Predict-Client-Subscriptions-to-Bank-Marketing-Campaigns/blob/main/PROJECT%20REPORT.pdf) 
