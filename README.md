# Spam Email Detection using Deep Learning (LSTM)


An advanced NLP and Deep Learning repository implementing a Long Short-Term Memory (LSTM) network to classify emails into Spam or Ham (Non-Spam) with high precision.

---

## 📊 Dataset & Class Balancing
The initial raw dataset consisted of 5,171 heavily imbalanced samples. Training on skewed classes forces deep learning models to over-predict the majority class. 
* To ensure optimal model objectivity, **Downsampling** was applied to match the minority class exactly.
* The training corpus was restructured into a balanced subset of **2,998 unique email streams** (50% Spam / 50% Ham).


---

## 🛠️ NLP Preprocessing Pipeline
Raw emails include noise that degrades network optimization. The following preprocessing pipeline was created using `NLTK` and Python's string manipulations:
1. **Header Stripping:** Discarded generic structure labels like `"Subject:"`.
2. **Punctuation Clearing:** Removed special string characters via a custom translation matrix mapping.
3. **Stop Word Removal:** Tokenized strings to lower-case and removed generic stop words (e.g., "the", "is", "at") to keep only high-weight keywords.

### WordCloud Insights

---

## 🧠 Model Architecture
Text files were vectorized into numerical matrices and bound to a sequence length limit of 100 via post-padding configurations.

The architecture comprises:
* **Embedding Layer:** maps token indices into a 32-dimensional semantic vector space.
* **LSTM Layer (16 units):** captures the directional word order and contextual dependencies across text structures.
* **Dense Layer (32 units, ReLU):** performs dense feature grouping.
* **Output Layer (1 unit, Sigmoid):** outputs binary probability scores.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=100),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

---

## 📈 Performance & Convergence
Using `ReduceLROnPlateau` to decrease the learning rate when loss plateaus and `EarlyStopping` to catch optimal validation metrics, the training achieved rapid convergence:

* **Training Final Accuracy:** ~97.12%
* **Test Accuracy:** **95.83%**
* **Test Loss:** **0.1861**
