import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split

# Define hyperparameters

OUTPUT_PATH = "./matrix_factorization_results/results_" + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + ".csv"
NUM_EPOCHS = 1500
NUM_FACTORS = 50
LEARNING_RATE = 0.0001
LAMBDA = 0.01

# First we get our data
ratings_names = ["UserID", "MovieID", "Rating", "Timestamp"]
ratings_df = pd.read_csv("data/ml-100k/u.data", header=None, sep="\t", names=ratings_names)

# Ensure the UserIDs and MovieIDs are consecutive
assert ratings_df["UserID"].max() == len(ratings_df["UserID"].unique()), "UserIDs are not consecutive"
assert ratings_df["MovieID"].max() == len(ratings_df["MovieID"].unique()), "MovieIDs are not consecutive"

# Randomly initialize our factor matrices, where the row index corresponds to (user id/movie_id - 1)
# Here we shall sample from a Gaussian distribution, but feel free to sample from other distributions
rng = np.random.default_rng()
users = rng.standard_normal((ratings_df["UserID"].max(), NUM_FACTORS))
movies = rng.standard_normal((ratings_df["MovieID"].max(), NUM_FACTORS))

# Now we get our train and validation data ready
train_df, val_df = train_test_split(ratings_df, stratify=ratings_df["UserID"])

train_loss = []
val_loss = []

for epoch in range(NUM_EPOCHS):

    # Initialize loss for this epoch
    current_train_loss = 0.0
    current_val_loss = 0.0

    # shuffle the training data at the start of each epoch to ensure stochaisticity
    train_data = np.array(list(zip(train_df["UserID"], train_df["MovieID"], train_df["Rating"])))
    rng.shuffle(train_data)
    for user_id, movie_id, rating in tqdm(train_data, desc=f"Training epoch {epoch}/{NUM_EPOCHS-1}", total=len(train_df)):
        error = rating - np.dot(users[user_id-1], movies[movie_id-1])
        user_grad = error * movies[movie_id-1] - LAMBDA * users[user_id-1]
        movie_grad = error * users[user_id-1] - LAMBDA * movies[movie_id-1]
        users[user_id-1] += LEARNING_RATE * user_grad
        movies[movie_id-1] += LEARNING_RATE * movie_grad
    
    # calculate train loss
    for user_id, movie_id, rating in list(zip(train_df["UserID"], train_df["MovieID"], train_df["Rating"])):
        train_square_error = (rating - np.dot(users[user_id-1], movies[movie_id-1]))**2 
        regularization = LAMBDA * (np.dot(users[user_id-1], users[user_id-1]) + np.dot(movies[movie_id-1], movies[movie_id-1]))
        current_train_loss += train_square_error + regularization

    print(f"\tTrain loss: {current_train_loss}")

    # validate
    for user_id, movie_id, rating in tqdm(list(zip(val_df["UserID"], val_df["MovieID"], val_df["Rating"])), desc="Validating", total=len(val_df)):
        val_square_error = (rating - np.dot(users[user_id-1], movies[movie_id-1]))**2
        current_val_loss += val_square_error

    print(f"\tVal loss: {current_val_loss}")

    train_loss.append(current_train_loss)
    val_loss.append(current_val_loss)

eval_df = pd.DataFrame({"Train Loss": train_loss, "Val Loss": val_loss})
eval_df.to_csv(OUTPUT_PATH, index=False)