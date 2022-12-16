## **Book Recommendation Project**
***
![image](https://th.bing.com/th/id/R.ca1231dc9deeda5bc6aec9ccfd1069ec?rik=NWH7mSSmVGsqqA&pid=ImgRaw&r=0)


 
#### Overview
Recommendation systems are widely used to recommend products to the end users that are most appropriate. This system uses features of collaborative filtering to produce efficient and effective recommendations. The collaborative recommendation is probably the most familiar, most widely implemented, and most mature of the technologies. Collaborative recommender systems aggregate ratings of objects, recognize commonalities between users on the basis of their ratings and generate new recommendations.
A book recommendation system is a type of recommendation system where we have to recommend similar books to the reader based on his interest. The books recommendation system is used by online websites which provide eBooks like google play books, open library, good Read, etc.
We build an Ontology and knowledge base to present our ideas. After that, we build a chatbot to mimic our final project idea.

#### Problem Formulation
Sometimes it’s hard to find the suitable book, mobile phone, clothes, or places that provide all your needs. You have to go to many places or search a lot in websites, this process wastes time, money and effort, maybe it ends that you didn’t find what you search for. What if you want to read a book that matches your personality, thoughts and you can really enjoy it. It’s complicated, isn’t it?
  Books recommender system makes it easier for you to recommend books based on what you want.

#### Methodolgy
1. Content-based filtering: are based on a description of the item and a profile of the user’s preferences.These methods are best suited to situations where there is known data on an item (name, location, description, etc.), but not on the user. Content-based recommenders treat recommendation as a user-specific classification problem and learn a classifier for the user’s likes and dislikes based on an item’s features.
2. Colaborative filtering: based on user rating and consumption to group similar users together, then to recommend products/services to users. 

![image](https://th.bing.com/th/id/R.324f09a5286c0f8fbc256cd759309e82?rik=2hK%2bvSMXQFSEaQ&pid=ImgRaw&r=0)


#### Requirements
Please setup this packages to enable you to run the Colab code:
```python
!pip install surprise
```
You need to run this code in cmd in the same path of final_project.py is located in GUI code:
```python
Streamlit run final_project.py
```

#### Content-based filtering
![image](https://th.bing.com/th/id/R.11e2d61c19692fa99209f54010dc48a9?rik=f3D%2fW3mQ2uD7vQ&pid=ImgRaw&r=0)
Content-based filtering methods are based on a description of the item and a profile of the user’s preferences.These methods are best suited to situations where there is known data on an item (name, location, description, etc.), but not on the user. 
Content-based recommenders treat recommendation as a user-specific classification problem and learn a classifier for the user’s likes and dislikes based on an item’s features.
In this system, keywords are used to describe the items and a user profile is built to indicate the type of item this user likes. These algorithms try to recommend items that are similar to those that a user liked in the past, or is examining in the present. In particular, various candidate items are compared with items previously rated by the user and the best-matching items are recommended. This approach has its roots in information retrieval and information filtering research.
To create a user profile, the system mostly focuses on two types of information:
    1. A model of the user’s preference.
    2. A history of the user’s interaction with the recommender system.

#### Colaborative filtering
![image](https://www.digitalvidya.com/wp-content/uploads/2019/12/1_mM089Lta5X6zkUkULcO9aA_2eb032e471550e902d447becfb1036ed.png)
- Collaborative filtering is based on the idea that similar people (based on the data) generally tend to like similar things. It predicts which item a user will like based on the item preferences of other similar users. 
- Collaborative filtering uses a user-item matrix to generate recommendations. This matrix contains the values that indicate a user’s preference for a given item. These values can represent either explicit feedback (direct user ratings) or implicit feedback (indirect user behavior such as listening, purchasing, and watching).
    o	Explicit Feedback: The amount of data that is collected from the users when they choose to do so. Many times, users choose not to provide data for the user. So, this data is scarce and sometimes costs money.  For example, ratings from the user.
    o	Implicit Feedback: In implicit feedback, we track user behavior to predict their preference.

•	Classification:
1.	Singular Value Decomposition (SVD): 
A method from linear algebra that has been generally used as a dimensionality reduction technique in machine learning. SVD is a matrix factorization technique, which reduces the number of features of a dataset by reducing the space dimension from N-dimension to K-dimension (where K<N). In the context of the recommender system, the SVD is used as a collaborative filtering technique. It uses a matrix structure where each row represents a user, and each column represents an item. The elements of this matrix are the ratings that are given to items by users.

2.	Non-Negative matrix factorization (NMF): 
Non-Negative Matrix Factorization is a state-of-the-art feature extraction algorithm. NMF is useful when there are many attributes, and the attributes are ambiguous or have weak predictability. By combining attributes, NMF can produce meaningful patterns, topics, or themes.
Each feature created by NMF is a linear combination of the original attribute set. Each feature has a set of coefficients, which are a measure of the weight of each attribute on the feature. There is a separate coefficient for each numerical attribute and for each distinct value of each categorical attribute. The coefficients are all non-negative.
3.	NormalPredictor: 
The Normal Predictor algorithm predicts the blank score based on the score matrix of the training set. First, assume that the distribution of the entire score obeys the normal distribution. Based on this assumption, the expectation and variance of the predicted value are obtained by calculating the maximum likelihood estimation of the training set.

•	Cluster:
- Co-clustering:
Co-clustering is rather a recent paradigm for unsupervised data analysis, but it has become increasingly popular because of its potential to discover latent local patterns, otherwise unapparent by usual unsupervised algorithms such as k-means. Wide deployment of co-clustering, however, requires addressing several practical challenges such as data transformation, cluster initialization, scalability, and so on.

##### GUI (User Interface):
Asking the user if he/she has an ID or not. If the user said yes, ywe ask him/her again to enter his/ her ID. After that, the ID is passed to the collaborative filtering function that we established and returns to the user sort of books he/her may like. If the user doesn’t have an ID and said no in the first question. We ask him/her to enter a book he/she wants to read. Then the book is returned to the content-based function that we established and returns to the user the similar books based on cosine similarity.


