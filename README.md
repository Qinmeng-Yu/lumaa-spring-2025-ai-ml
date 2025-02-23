# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

## Author

**Name**: Qinmeng Yu

**Contact**: qinmeng.yu@duke.edu

****

## **Overview**
This project implements a **content-based recommendation system** that suggests movies similar to a user's input query. Given a short text description of movie preferences, the system **compares the query** with movie descriptions in a dataset and returns the **top 5 most similar movies**.

The system uses **TF-IDF vectorization** and **cosine similarity** to find the most relevant matches.

## **Dataset**
- **Source**: [IMDb Dataset of Top 1000 Movies and TV Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)
- **File**: `imdb_top_1000.csv`
- **Size**: 1000 movies with metadata, including **title** and **overview (plot summary)**
- **Usage**: The system uses the **"Overview"** column to calculate movie similarity.

## **Installation & Setup**

### **1. Prerequisites**
Ensure you have **Python 3.12** installed.

### **2. Create Virtual Environment (Optional)**
It's recommended to use a **virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### **3. Install Dependencies**
Run the following command to install required packages:
```bash
pip install -r requirements.txt
```

#### **`requirements.txt`**
```plaintext
numpy
pandas
scikit-learn
```

## **How to Run the Recommendation System**
1. **Run the script with a query**:
   ```bash
   python recommend.py "I love thrilling action movies set in space, with a comedic twist."
   ```
2. **Example Output**:
   ```
   Top Recommendations:
   1. Amarcord (Similarity: 0.29)
   2. The Incredibles (Similarity: 0.16)
   3. The Man Who Would Be King (Similarity: 0.15)
   4. Aliens (Similarity: 0.14)
   5. Barton Fink (Similarity: 0.14)
   ```

Due to the small dataset size, cosine similarity scores may be relatively low. Additionally, cosine similarity may not be the most effective method for capturing the relationship between the **"Overview"** provided in the IMDb dataset and user queries.

## **Project Structure**

```
📂 content-based-recommendation/
│── dataset/
│   ├── imdb_top_1000.csv  # Movie dataset
│── recommend.py           # Main recommendation script
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
```

## **How It Works**
1. **Load Dataset**: Reads `imdb_top_1000.csv` and extracts movie **titles** and **descriptions**.
2. **Preprocess Text**: Converts text to **lowercase**, removes punctuation, and applies **TF-IDF vectorization**.
3. **Compute Similarity**: Uses **cosine similarity** to compare user input with movie descriptions.
4. **Return Top-N Results**: Outputs the **top 5 most similar** movies.

## **Example Queries**
You can try different queries to test the system:
```bash
python recommend.py "I love horror movies with supernatural elements."
python recommend.py "I enjoy romantic comedies with emotional depth."
python recommend.py "I prefer action-packed spy thrillers."
```

## **Expected Salary**

**Hourly**: $25 – $40 per hour

## **License**

This project is open-source and free to use. The dataset belongs to **IMDb** and is provided by **Kaggle**.