# ALiRec
Learning recommender systems

- What kind of recommender system?
    - Collaborative filtering, matrix factorization, two tower, GNN

## Two Tower Considerations
- What kinds of towers do we want: user/item, item/item, more than two towers, something else?
- Data engineering
    - What features are input to the towers
        - Review the dataset. How can we define user features? How can we define item features?
        - How do we represent our input data (how to get embeddings for image, text, structured)?
- Training
    - All towers should be trained jointly, end to end
    - Tower architectures (MFP, RNN, transformers, etc)?
    - Could use pointwise (log loss), pairwise (bpr), or contrastive losses
    - How to measure similarity between tower outputs (dot product, cosine similarity, etc)
- Candiate Generation (two tower + ANN to generate candidate recommendations)
    - Offline: Precompte all embeddings for user and item using two towers. When recommending to user, lookup their embedding and then use ANN algo (e.g. faiss, scANN, etc) to generate candidates
    - Online: Precompute item embeddings. Compute user embedding in real time, then use ann method to retrieve candidates
- Ranker (after candidate generation, rank the candidates)
    - What model to use? Typically a deep nn like DCN, DeepFM, Deep & Wide, etc
    - What features does it take
        - Make sure to include cross-features of the user and item sets since the two tower model struggles with these
- Variations
    - Item only tower
        - Used for related item recommendations. Follows basic structure of the two tower

## Collaboartive Filtering
- Data
    - The data in the CF approach can be modeled as a weighted, undirected, bipartite grpah. Users are one color and items are another. Edges between users and items represent some interaction between them (click, purchase, etc), while the weight can represent the strength of that interaction (review, rating, number of clicks, etc)
        - From "Recommender Systems: A Primer", they define collaborative filtering algorithms as those whose only input is the user-item rating matrix
        - In other words, collaborative filtering follows the idea that users who have similar tastes will continue to have similar tastes in the future
    - The benefit of this data model is that the recommender system does not have to understand the users or items themselves. 
    - Suffers from cold start, scalability, and sparsity
        - Cold start refers to the fact that a user initially has no interaction history with the items, which makes recommending items difficult
        - Scalability refers to the fact that the number of users and items in many settings can be in the millions or tens of millions
        - Sparsity refers to the fact that of these millions of users and items, each user will interact with a very small amount of items (our bipartite graph is usually a sparse matrix) meaning we usually do not have enough data using a pure collaborative filtering approach to make accurate recommendations

## Content-Based Filtering
- Data
    - The data in the CBF approach is more traditional structured data. Items are defined by a set of features (name, description, location, etc), which are used to build representations (tf-idf, neural networks, etc) of the items. 
        - In other words, the goal of content based approaches is to consider a single user's preferences to determine that user's future preferences
- Training
    - We can think of training for the CBF approach as learning user specific classifiers. 
