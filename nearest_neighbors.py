import numpy as np
import pandas as pd
import pgeocode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

# Define hyperparameters
NUM_NEIGHBORS = 5

# Gather ratings data
ratings_names = ["UserID", "ItemID", "Rating", "Timestamp"]
ratings_df = pd.read_csv("data/ml-100k/u.data", header=None, sep="\t", names=ratings_names)

# Gather movie data
movie_names = [
    "MovieID", "MovieTitle", "ReleaseDate", "VideoReleaseDate", "IMDbURL", "unknown", "Action", "Adventure", 
    "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movie_df = pd.read_csv("data/ml-100k/u.item", header=None, sep="|", encoding="latin-1", names=movie_names)

# Gather user data
user_names = ["UserID", "Age", "Gender", "Occupation", "ZipCode"]
user_df = pd.read_csv("data/ml-100k/u.user", header=None, sep="|", names=user_names)

# Get lat and long from user zip codes
# This will produce some NaNs since some are outside US and pgeocode just misses on some
# Make sure to filter the NaNs ot before scaling 
nomi = pgeocode.Nominatim('us')
user_df[["latitude", "longitude"]] = nomi.query_postal_code(user_df["ZipCode"].to_list())[["latitude", "longitude"]]
cleansed_user_df = user_df[~user_df.isna().any(axis=1)]

# One hot encode each user's categorical features
occupation_oh_enc = OneHotEncoder()
occupation_oh_enc.fit(cleansed_user_df[["Gender", "Occupation"]])
categorical_enc = occupation_oh_enc.transform(cleansed_user_df[["Gender", "Occupation"]]).toarray()

# Insert the one hot encoded features back into user data
full_user_enc = np.concat([
    np.expand_dims(cleansed_user_df["Age"], axis=-1), 
    categorical_enc, 
    np.array(cleansed_user_df[["latitude", "longitude"]])
], axis=-1)

# Ensure each feature is on the same scale
# NOTE: StandardScaler is sensitive to outliers, try other scalers to see how it affects performance
# In fact if you examine the lat, long distributions you will see they contain outliers (perhaps RobustScaler here)
# NOTE: We are also making the choice to scale the 1-hot encoded data. I'm not sure if this is traditional, but perhaps
# play around with scaling the non-categorical features first, then 1-hot encode
scaler = StandardScaler()
scaled_full_user_enc = pd.DataFrame(scaler.fit_transform(full_user_enc))
scaled_full_user_enc = pd.concat([
    cleansed_user_df[["UserID"]].reset_index(drop=True), 
    pd.DataFrame(scaler.fit_transform(full_user_enc))
], axis=1)

# Now we'll split the data into "train" and "test"
# We'll sample n ratings from each user to be our "test" set
# Then we use our algorithm to guess the ratings for each of these queries and see how close we got
user_ratings_df = pd.merge(ratings_df, movie_df, "inner", left_on="ItemID", right_on="MovieID")[["UserID", "MovieID", "Rating"]]
cleansed_user_ratings_df = user_ratings_df[user_ratings_df["UserID"].isin(scaled_full_user_enc["UserID"])]
train_df, test_df = train_test_split(cleansed_user_ratings_df, test_size=0.2, stratify=cleansed_user_ratings_df["UserID"])

# Calculate the k nearest neighbors for each user (we get the similarity score too)
# NOTE: We use distances here but this is probably not very good.
# Try experimenting with different notions of similarity (e.g. cosine, Pearson correlation, etc) instead of raw distance
nbrs = NearestNeighbors(n_neighbors=NUM_NEIGHBORS+1).fit(scaled_full_user_enc.drop(columns=["UserID"]))
distances, neighbors_indices = nbrs.kneighbors(scaled_full_user_enc.drop(columns=["UserID"]))
neighbors = scaled_full_user_enc["UserID"].to_numpy()[neighbors_indices][:, 1:]
distances = distances[:, 1:]
neighbor_df = pd.DataFrame({
    'UserID': scaled_full_user_enc.loc[np.repeat(np.arange(len(user_df)), NUM_NEIGHBORS-1), "UserID"], # possible bug when NUM_NEIGHBORS <= 1, need to investigate
    'NeighborUserID': neighbors.ravel(),
    'Distance': distances.ravel()
})

# For each UserID, MovieID in test_df we get the neighbors and similarities 
# and use their weighted sum as the rating
test_neighbor_df = pd.merge(test_df, neighbor_df, "inner", "UserID")
test_ratings_df = pd.merge(test_neighbor_df, train_df, "left", left_on=["NeighborUserID", "MovieID"], right_on=["UserID", "MovieID"])
test_ratings_df = test_ratings_df[~test_ratings_df["Rating_y"].isna()]
test_ratings_df["WeightedRating"] = test_ratings_df["Distance"]*test_ratings_df["Rating_y"]
test_ratings_df = test_ratings_df[["UserID_x", "MovieID", "Distance", "WeightedRating"]]
test_ratings_df = test_ratings_df.rename(columns={"UserID_x":"UserID"})

prediction_df = test_ratings_df.groupby(by=["UserID", "MovieID"]).sum()
prediction_df["NormalizedWeightedRating"] = prediction_df["WeightedRating"] / prediction_df["Distance"]
prediction_df = prediction_df.reset_index()
prediction_df = prediction_df.drop(columns=["Distance"])

eval_df = pd.merge(test_df, prediction_df, "left", on=["UserID", "MovieID"])
eval_df = eval_df.rename(columns={"Rating": "ActualRating"})

# With the actual rating and predicted rating for each user/movie pair in test_df
# we can evaluate how well the algorithm did and output to csv
eval_df.to_csv("evaluation.csv", index=False)
