

```markdown
# 🎬 Hybrid Recommender System using MovieLens 100K

A lightweight and interpretable hybrid recommender system built using the [MovieLens 100K](https://grouplens.org/datasets/movielens/) dataset. This model combines collaborative filtering with content-based metadata (user and item features) to make rating predictions. It's particularly effective in cold-start scenarios and is designed for simplicity, transparency, and fast computation.

---

## 📌 Project Overview

This project implements a baseline **Hybrid Collaborative Filtering** approach using:
- **User Metadata**: Age, Gender, Occupation
- **Item Metadata**: Movie Genre indicators
- **Rating Data**: 100,000 user-movie interactions

The model extends the `AlgoBase` class from the [Surprise](http://surpriselib.com/) library and predicts ratings using a **rule-based formula**:

```

prediction = global\_mean
\+ 0.5 × mean(user\_features)
\+ 0.5 × mean(item\_features)

````

---

## 📊 Performance

| Metric          | Cross-Validation | Test Set     |
|----------------|------------------|--------------|
| **RMSE**       | 1.1261           | 1.1239       |
| **MAE**        | 0.9452           | 0.9420       |

- Evaluation: 5-fold cross-validation and 80/20 train-test split
- No learnable parameters → Highly interpretable and efficient

---

## ✅ Features

- ✅ Hybrid architecture (Collaborative + Content-Based)
- ✅ No training or optimization required
- ✅ Excellent for **cold-start** problems
- ✅ High interpretability
- ✅ Lightweight and fast to run

---

## 🧠 Use Cases

- 🎯 **Cold-start** recommendation systems
- 🔍 Applications requiring **explainable AI**
- 💻 Systems deployed in **resource-constrained** environments

---


## 📈 Model Comparison

| Feature                | Hybrid Baseline Model (This Work) | Traditional CF (e.g., KNN, SVD) |
| ---------------------- | --------------------------------- | ------------------------------- |
| **Uses Metadata**      | ✅ Yes                             | ❌ No                            |
| **Cold Start Support** | ✅ Yes                             | ❌ No                            |
| **Training Required**  | ❌ No                              | ✅ Yes                           |
| **Interpretability**   | ✅ High                            | ❌ Low (latent dimensions)       |
| **Performance Tuning** | ❌ Not applicable                  | ✅ Required                      |

---

## 🧪 Evaluation Metrics

* **Root Mean Square Error (RMSE)**: Penalizes large errors; lower is better.
* **Mean Absolute Error (MAE)**: Measures absolute difference between predicted and actual ratings.

---

## 💡 Future Enhancements

* 🔧 Learn feature weights using linear regression or attention
* 🤖 Upgrade to deep hybrid models with embeddings
* 🧩 Handle nonlinear interactions with more complex logic

---

## 🧑‍💻 Author

**Sri Satya Sai Immani**
Graduate Student, Case Western Reserve University
📍 Cleveland, Ohio, USA

---

## 📚 References

* [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/)
* [Surprise Library](http://surpriselib.com/)
* [Scikit-learn](https://scikit-learn.org/)
* [Pandas](https://pandas.pydata.org/)

---



