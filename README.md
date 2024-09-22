# Financial transaction anomaly detection

This project aims to develop a machine learning model for
detecting anomaly in financial transaction. 
In this repository we added both *ipynb file* and 
*streamlit* project that helps the end user to
maintain the machine learning model without any coding
experience.

## How to run the project:

* For running the ipynb file, using colab would be enough.
If you do not want to upload the ipynb file on the colab for
 running the model, I've added the url for the files:

## Notebooks

| Notebook              | Description            |
| --------------------- | ---------------------- |
| [Model Training](https://colab.research.google.com/drive/10KFp6mUgYjCpHfekwaDKDLyM4yig0hh4?usp=drive_link) | Train a machine learning model using Python and relevant libraries. |
| [Data Visualization](https://colab.research.google.com/drive/16kGRFRWHqOFjJ5vSYaSGidObIbPWVOVb?usp=drive_link) | Visualize data and model outcomes using various plotting tools. |



# Streamlit project
In our project we also added a small UI feature that will help any non-technical
user to train and visdualize the model. For this I've added a requirement.txt
file. This project must be run from a local VM as it will open a random port
to run on a localhost. 

## Setting Up a Python Virtual Environment
### Prerequisites

- Python 3.x installed on your machine.
- Ensure `pip` is installed and up to date by running the following command:

    ```bash
    pip install --upgrade pip
    ```

### Step 1: Create a Virtual Environment

To create a virtual environment, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the directory where you want to create your virtual environment.
3. Run the following command to create the virtual environment:

    ```bash
    python -m venv venv
    ```

   - `venv` is the name of your virtual environment. You can choose any name for it, but `venv` is commonly used.

### Step 2: Activate the Virtual Environment

Once the virtual environment is created, activate it using the following command based on your operating system:

- On **Windows**:

    ```bash
    venv\Scripts\activate
    ```

- On **macOS** and **Linux**:

    ```bash
    source venv/bin/activate
    ```

Once activated, you should see the virtual environment’s name (e.g., `(venv)`) in your terminal prompt.

### Step 3: Install Dependencies

To install the required dependencies listed in `requirements.txt`, use the following command:

```bash
pip install -r requirements.txt
```

# Run the project
If you successfully create the virtual env for this project you can simply
run the project by using this command:
``streamlit run app.py``