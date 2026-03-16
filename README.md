# Gojek App Review Sentiment Analysis

This project aims to perform sentiment analysis on user reviews of the **Gojek** application scraped directly from the Google Play Store. The project pipeline includes data collection (web scraping), text preprocessing (NLP), Exploratory Data Analysis (EDA), feature extraction, and Machine Learning modeling to predict and classify user sentiment (Positive, Neutral, or Negative).

## 📂 Directory Structure

Here is an explanation of the main files in this project:

- `scraping_gojek_sentiment_analysis.ipynb`
  A Jupyter Notebook containing the script to scrape Gojek app review data from the Google Play Store. This script utilizes the `google-play-scraper` library to pull up to 500,000 recent reviews.
- `gojek_sentiment_analysis.csv`
  The raw dataset extracted from the scraping script. It contains user reviews along with ratings, dates, and other metadata.
- `notebook.ipynb`
  The main Jupyter Notebook containing the complete Data Science workflow, including:
  - **Function Helper:** A collection of functions for text cleaning.
  - **Preprocessing Data:** Text cleaning (such as case folding, punctuation/URL removal, stopword removal, and Indonesian stemming).
  - **Feature Extraction:** Processing text data into numerical representations using feature extraction methods (such as TF-IDF or word vector conversion).
  - **Data Splitter:** Splitting the dataset into training data and testing data.
  - **Modeling & Evaluation:** Training various Machine Learning algorithms and evaluating model performance (Accuracy, Precision, Recall, and F1-Score).
- `requirements.txt`
  A list of all required Python packages and library dependencies (including pandas, scikit-learn, google-play-scraper, etc.) to run the entire project smoothly.

## 🛠 Installation and Setup

Follow the steps below to set up your local environment to run the project.

1. **Ensure Python is Installed:** Minimum recommended Python version is 3.8 or higher.
2. **Setup Virtual Environment (Recommended):**
   ```bash
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On Linux/Mac:
   source env/bin/activate
   ```
3. **Install Requirements Dependencies:**
   Run the following command in the terminal to install all necessary packages.
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **(Optional) Scrape New User Data:**
   - If you want to update the dataset, open `scraping_gojek_sentiment_analysis.ipynb`.
   - Run all cells (*Run All*) to pull the latest data. The output will overwrite `gojek_sentiment_analysis.csv` by default.
2. **Running Main Modeling and Analysis:**
   - Run Jupyter Notebook / Jupyter Lab in your local environment.
   - Open `notebook.ipynb`. 
   - You are advised to run the notebook sequentially ("Kernel -> Restart & Run All") from *Function Helper* to *Modeling* to observe the data flow from preprocessing to Machine Learning accuracy metrics.

## 📊 Technologies Used
- **Python**: Main Programming Language.
- **Jupyter Notebook**: Interactive computing environment.
- **Pandas & NumPy**: For DataFrame-based data manipulation and analysis.
- **Scikit-Learn**: Main module for Machine Learning models, evaluation metrics, and TF-IDF extraction.
- **google-play-scraper**: Extracts app reviews directly from the Public Play Store API.

---
*Created for Data Analysis and Natural Language Processing purposes.*
