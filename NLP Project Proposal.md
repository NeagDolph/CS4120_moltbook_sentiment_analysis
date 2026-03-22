CS-4120   
Final Project Proposal  
Group Members: Neil Agrawal, Ernie Chae, Cody Ho, Andrew Lotocki

Project Overview and Goals:  
In this project, we will look at posts from the Moltbook platform, a social network similar to Reddit, where AI agents interact with each other without human involvement. First, we will generate additional metadata from post text, including sentiment and polarity scores, word and sentence counts, and vectorized text embeddings, then combine these with existing features such as topic and toxicity labels. Next, we will normalize the resulting dataset and apply linear regression as well as random forest models to identify which post features are most strongly associated with engagement. We can identify this by measuring the number of upvotes, downvotes, and comments. We will then build feedforward neural networks to predict engagement metrics and evaluate how effectively linguistic features can be used to predict post-engagement, and detail our results.   
   
Data Description:  
The dataset we will be using is Moltbook, which is [available](https://huggingface.co/datasets/TrustAIRLab/Moltbook/tree/main) through Hugging Face. Each row represents a single post and includes a unique ID, a topic label, a toxicity score, and the post content stored as a dictionary, with there being over 44,000 posts in total. This dictionary includes the post text, author information, creation timestamp, and engagement statistics. A more detailed field description is available [here](https://huggingface.co/datasets/TrustAIRLab/Moltbook/blob/main/README.md).

Research Question:   
Which features in a post on the Moltbook platform are associated with higher engagement in the form of upvotes, downvotes, and comments, and how can these features be used to create more posts with a high engagement rating?

Models:  
We will first create additional metadata from the raw post text, including sentiment and polarity scores from NLTK, post length metrics such as word count and sentence count, and text vector representations using the Ollama text embedding “Nomic Embed Text”. These features will then be combined with existing metadata to complete our data. Then, we will build a traditional linear regression model and random forest regressors to analyze feature importance and identify which linguistic characteristics are most strongly associated with engagement. These models will be implemented using scikit-learn with standard hyperparameters, including tuning the number of trees and the depth of each tree. For prediction, we will build a feedforward neural network using Keras. The model will take text embeddings and engineered features as input and output the predicted counts for upvotes, downvotes, and comments. Our neural network architecture will be designed by the group, while the underlying computation will be done by Keras. We will experiment with different hidden layer sizes and activation functions. 

Tools/libraries/packages:   
The project will be developed in Python using NLTK for sentiment and linguistic feature extraction, scikit-learn for feature scaling and regression models, and Keras with TensorFlow for neural network modeling. Matplotlib will be used for visualization. Hugging Face will be used for dataset access, and Ollama with the Nomic-Embed-Text model will be used to generate text embeddings, with PyTorch used as needed for experimentation or additional models.

How we will evaluate the model:  
Model performance will be evaluated on a test data subset. For regression tasks predicting upvotes, downvotes, and comment counts, we will report mean squared error and R² scores, using ReLU activations in hidden layers and linear activation in the output layer. We will also define a binary classification task for high and low engagement, based on whether engagement exceeds the dataset average. For this model, we will use a sigmoid output activation and evaluate performance using accuracy, precision, recall, and F1 score.

Visualizations and Results:  
We will include feature importance bar plots for our regression and random forest models, a correlation heatmap between text features and engagement metrics, and predicted-versus-actual scatter plots for the neural network. For our engagement classification task, we will use confusion matrices and summary metric plots to show model performance.

Timeline:  
Week 1 will focus on data exploration, cleaning, metadata creation, and feature extraction.  
Spring break  
Week 2 will involve training regression and random forest models and analyzing feature importance.  
Week 3 will focus on building and tuning the neural network and completing the first check-in write-up.  
Week 4 will include the check-in write-up due on 3/23, followed by a group meeting. During this week, any necessary changes will be completed based on check-in and meeting feedback.  
Week 5 will focus on completing neural network tuning and polishing experiments. During this week, we will also create visualizations for the upcoming presentation.  
Week 6 will focus on preparing the presentation slideshow and beginning the final report write-up.  
Week 7 is presentation week, during which we will present and continue writing the final report.  
Week 8 will focus on finalizing the report before the April 22nd deadline.

Individual Contributions:  
Andrew is responsible for procuring the data, generating metadata, and working to fine-tune our regression models, along with the corresponding write-up components of these tasks. Cody will take responsibility regarding the neural network training and tuning, running experiments to produce the needed visualizations for the team’s slides, and write up relating to the neural network modeling.  Neil is responsible for the embedding pipeline (converting the post text into embeddings using Nomic-Embed-Text) and creating visual maps of the data to find clusters, and using the team’s findings to provide recommendations on how to increase post engagement. Ernie will be responsible for the training pipeline, as well as the final write-up and slides, ensuring that all our numerical data aligns with our experiments.

Feedback 

### **Comments:**

Seems like a pretty cool project and I liked the various models you're thinking about\! One thing I want to check with the professor is whether analyzing an AI-generated dataset fits within the scope of the project, so I'll let you know when I hear back. In the meantime, it's worth thinking about whether the NLP component is substantial enough since the core task is really engagement prediction, so make sure your report frames the linguistic feature extraction and analysis as the central NLP contribution. Also, your research question asks how features can be used to create more engaging posts, but your methodology only covers prediction as well as the feature analysis but doesn't really answer that part, so you may want to either refine the research question or think about how you'd actually address it otherwise

Akshitha Bhashetty, Mar 1 at 8:23pm  
TA Mentor: Akshitha Bhashetty

Update from Terra: analyzing an AI-generated dataset is appropriate for this project, looks interesting\!

Terra Blevins, Mar 9 at 1:40pm

First model: Extract the sentiment based on the number of upvotes and replies a comment has

- Models we want to use:  
  - FFNN to put between 3 classes of engagement (\# upvote, \# downvote, num comments)  
  - 

Second model: From there, get the features that affect the sentiment?

- 

For Monday:

Task 1: Check-in Writeup and Meeting (10 points)  
Turn in a 1-2 page report describing the progress on the project. You can think of this as the first draft of  
your final report. The writeup should be single-spaced, with 1-margins and in 11/12pt, easily legible font  
(e.g., Times New Roman, Arial). It should contain the following sections: (i) Introduction, where you  
describe the goals of your project and provide an overview of the work to be done; (ii) Data, where you  
describe the datasets you are using including relevant statistics (e.g., number of samples, class  
distribution, etc…); (iii) Models, where you describe the models you will be using (including the  
baselines); and (iv) Preliminary Results, such as the performance of your baseline models.

Task 1: Check-in Writeup and Meeting (10 points)  
Turn in a 1-2 page report describing the progress on the project. You can think of this as the first draft of  
your final report. The writeup should be single-spaced, with 1-margins and in 11/12pt, easily legible font  
(e.g., Times New Roman, Arial). It should contain the following sections: (i) Introduction, where you  
describe the goals of your project and provide an overview of the work to be done; (ii) Data, where you  
describe the datasets you are using including relevant statistics (e.g., number of samples, class  
distribution, etc…); (iii) Models, where you describe the models you will be using (including the  
baselines); and (iv) Preliminary Results, such as the performance of your baseline models.

