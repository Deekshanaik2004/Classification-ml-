# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 2: Sample dataset (you can replace this with your own CSV file)
data = {
    'email': [
        "Win money now!", 
        "Limited time offer, click here!", 
        "Hi, how are you?", 
        "Let’s meet for lunch tomorrow.", 
        "Free entry in 2 a weekly contest", 
        "Your invoice is attached", 
        "Congratulations! You've won!", 
        "Reminder for your appointment"
    ],
    'label': [1, 1, 0, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.25, random_state=42)

# Step 4: Convert text into numerical vectors
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Step 5: Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Step 6: Make predictions
predictions = model.predict(X_test_counts)

# Step 7: Evaluate
print("Accuracy:", accuracy_score(y_test, predictions))

# Step 8: Try with your own email
your_email = ["You have won a prize! Click to claim."]
your_email_counts = vectorizer.transform(your_email)
your_prediction = model.predict(your_email_counts)
print("Your email is:", "Spam" if your_prediction[0] == 1 else "Not Spam")
