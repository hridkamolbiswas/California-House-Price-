# California House Pricing

## Dataset

* contains n = 8 columns (features) and 20,640 samples (rows)
* In higher order moments the number of features are changed as follows:
    * for moment, d = 2: C(n+d, d) = C(8+2, 2) = 45
    * for moment, d = 3: C(n+d, d) = C(8+3, 3) = 165




## Execution tips:

* clone the repos
* python -m venv env (create virtual env)
* .\env\Scripts\activate (for windows , activate the virtual env)
* navigate to House_Price : cd House_Price
* pip install - r requirements.txt (install modules inside the venv)
* python main.py

## Save images and model

* few images are saved in images folder
* best model that gives the minimum error or test data <br> is saved in
model_pickle folder as .pickle file
