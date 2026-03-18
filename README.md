# Gojek App Review Sentiment Analysis

This project aims to conduct sentiment analysis on user reviews of the Gojek app collected directly from the Google Play Store. The project workflow includes data collection (web scraping), text preprocessing (NLP), Exploratory Data Analysis (EDA), feature extraction, and Machine Learning modeling to predict and classify user sentiment (Positive, Neutral, or Negative). The final results were obtained using model fine-tuning techniques, with a **validation accuracy over 92%**.

## Dataset Description

Data acquisition was performed using web scraping techniques with the google-play-scraper library. Data collection focused on the com.gojek.app application, with language and country localization set to Indonesia. The collected data is dynamic, sorted by “newest” with a total of 500,000 review entries to ensure the sentiment is relevant to the current version of the app.
## Dataset Summary
*   **File Name:** gojek_sentiment_analysis.csv
*   **Total Rows:** 513,394 reviews (excluding header).
*   **Data Source:** This dataset contains user review data for the Gojek application scraped from the Google Play Store.
*   **Main Purpose:** Commonly used for sentiment analysis, customer satisfaction analysis, or text classification.
*   [**Link Dataset**](https://www.kaggle.com/datasets/hendrimardani/gojek-sentiment-analysis-text)

## Column Explanations and Data Types
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| reviewId | String (Object) | Unique ID for each review (UUID). |
| userName | String (Object) | The name of the user account that provided the review. |
| userImage | String (Object) | URL to the user's profile picture. |
| content | String (Object) | The main review text written by the user (Indonesian). |
| score | Integer (int64) | Star rating given (scale 1 to 5). |
| thumbsUpCount| Integer (int64) | Number of other users who marked this review as "Helpful". |
| reviewCreatedVersion | String (Object) | The version of the Gojek app used when the review was created. |
| at | Datetime (Object)| Date and time the review was submitted by the user. |
| replyContent | String (Object) | Official reply text from the Gojek CS team (if any). |
| repliedAt | Datetime (Object)| Date and time when the Gojek team provided a reply. |
| appVersion | String (Object) | Current application version (often the same as reviewCreatedVersion). |

## Additional Information
*   **Missing Values:** The `replyContent` and `repliedAt` columns have many missing values because not all reviews are replied to by the Gojek team. The `reviewCreatedVersion` and `appVersion` columns also have some missing values.
*   **Language:** Most reviews are written in Indonesian with various language styles (formal, polite, or slang/informal).
*   **Rating Scale:** The `score` column is the primary indicator of sentiment (1-2 is usually negative, 3 is neutral, 4-5 is positive).


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
