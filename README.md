# 🎵 Song Lyrics-Based Music Recommender System

This project is a **content-based music recommendation system** that recommends songs using **song lyrics similarity**. It uses **TF-IDF Vectorization** and **Cosine Similarity** to recommend songs that are lyrically similar to the one you input.

---

## 📁 Dataset

The dataset used is \[`spotify_millsongdata.csv`] which contains:

* `artist`: Name of the artist
* `song`: Title of the song
* `link`: (dropped)
* `text`: Full song lyrics

A sample of **5000 songs** was taken to optimize performance during development.

---

## 🧠 How it Works

The system works in the following steps:

1. **Load and preprocess the dataset**:

   * Lowercase text
   * Remove newline characters and unwanted symbols
   * Apply stemming using `PorterStemmer`

2. **Vectorization**:

   * Lyrics are converted into vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
   * Common stop words (like *is*, *and*, *the*) are removed.

3. **Similarity Calculation**:

   * **Cosine Similarity** is calculated between all song lyrics.
   * Based on the similarity score, top 5 most similar songs are recommended.

---

## 🔧 Dependencies

Install the following Python libraries:

```bash
pip install pandas nltk scikit-learn
```

You will also need to download the NLTK tokenizer:

```python
import nltk
nltk.download('punkt')
```

---

## 🚀 How to Run

```python
# Step 1: Load the dataset
df = pd.read_csv('/content/spotify_millsongdata.csv')

# Step 2: Preprocessing and sampling
df = df.sample(5000).drop('link', axis=1).reset_index(drop=True)

# Step 3: Clean text
df['text'] = df['text'].str.lower().replace(r'^\w\s', ' ', regex=True).replace(r'\n', ' ', regex=True)

# Step 4: Tokenize and stem
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def token(txt):
    token = nltk.word_tokenize(txt)
    a = [stemmer.stem(w) for w in token]
    return " ".join(a)

df['text'] = df['text'].apply(lambda x: token(x))

# Step 5: TF-IDF and similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidf.fit_transform(df['text'])
similar = cosine_similarity(matrix)

# Step 6: Recommendation function
def recommender(song_name):
    if song_name not in df['song'].values:
        return f"❌ '{song_name}' not found in dataset."

    idx = df[df['song'] == song_name].index[0]
    distance = sorted(list(enumerate(similar[idx])), reverse=True, key=lambda x: x[1])

    song = []
    for s_id in distance[1:6]:
        song.append(df.iloc[s_id[0]]['song'])

    return song

# Example:
recommender("I'm Into You")
```

---

## 📌 Example Output

```python
recommender("I'm Into You")
# Output:
# ['I'm Into Something Good', 'You're Beautiful', 'You Belong with Me', 'Into You', 'Love Me Like You Do']
```

---

## 📊 Future Improvements

* Add artist filtering
* Integrate Spotify API to play audio previews
* Web interface using Streamlit or Flask
* Support fuzzy matching for misspelled song titles

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 🧑‍💻 Author

* **Bishal Shah**
* CMR Institute of Technology, AIML Student
* [shahbishal.com.np](https://shahbishal.com.np)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
