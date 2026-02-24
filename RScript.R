## Title: MovieLens Recommendation System Project 
## Sub-title: HarvardX PH125.9x - Data Science: Capstone
## Author: Li Man Hon
## Website: https://github.com/JoeLiMH/movielens

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(ggplot2)
library(knitr)
library(tidyr)


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

############################################################
#Exploratory Data Analysis
############################################################

#Overview
#Column names of the dataset
names(edx)
#Numbers of unique movies and users in the dataset
edx %>% summarize(n_movies = n_distinct(movieId), n_users = n_distinct(userId))

#Movie Effects
#Plot average ratings by movie in the dataset
edx %>% group_by(movieId) %>% 
  summarize(avg_rating = sum(rating)/n()) %>%
  ggplot(aes(avg_rating)) + 
  geom_histogram(bins = 30, color = I("black")) +
  labs(x = "Average rating", y = "Number of movies")

#Plot number of ratings by movie in the dataset
edx %>% count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = I("black")) +
  scale_x_log10() +
  labs(x = "Movies (log10)", y = "Number of ratings")

#User Effects
#Plot average ratings by user in the dataset
edx %>% group_by(userId) %>% 
  summarize(avg_rating = sum(rating)/n()) %>%
  ggplot(aes(avg_rating)) + 
  geom_histogram(bins = 30, color = I("black")) +
  labs(x = "Average rating", y = "Number of users")

#Plot number of ratings by user in the dataset
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = I("black")) +
  scale_x_log10() +
  labs(x = "Users (log10)", y = "Number of ratings")

#Genre Effects
#Individual genres and corresponding average ratings the dataset
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarise(Avg_Rating = mean(rating), count = n()) %>%
  arrange(desc(Avg_Rating))

#Plot average rating by genres combinations (number of rating >= 100,000)
edx %>% group_by(genres) %>% 
  summarize(n = n(), avg_rating = mean(rating), se = sd(rating)/sqrt(n())) %>% 
  filter(n >= 100000) %>%
  mutate(genres = reorder(genres, -avg_rating)) %>% 
  ggplot(aes(x = genres, y = avg_rating, ymin = avg_rating - 2*se, ymax = avg_rating + 2*se)) + 
  geom_point() +
  geom_errorbar() +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  labs(x = "Genre combination", y = "Average Rating")

############################################################
#Modeling, Prediction and Evaluating with RMSE
############################################################


########################################
#Construct Loss function (RMSE)
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

########################################
#Benchmark model

#Calculate the overall average rating across all movies
mu_hat <- mean(edx$rating, na.rm = TRUE)
#Calculate RMSE of this model based on overall average rating
naive_rmse <- RMSE(final_holdout_test$rating, mu_hat)

#Save the result into result tibble
rmse_result <- tibble(method = "Overall rating average", RMSE = naive_rmse)
rmse_result %>% kable(col.names = c("Method", "RMSE"))

########################################
#Modeling movie effects

#Estimate movie effects (b_i)
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(movieId = movieId[1], mu_hat = mu_hat, b_i = mean(rating - mu_hat))
#Predict ratings adjusting for movie effects
pred_bi <- final_holdout_test %>% 
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu_hat + b_i) %>% 
  pull(pred)
#Calculate RMSE of this model based on movie effects
movie_rmse = RMSE(final_holdout_test$rating, pred_bi)
#Save the result into result tibble
rmse_result <- add_row(rmse_result, method = "Movie Effect", RMSE = movie_rmse)
rmse_result %>% kable(col.names = c("Method", "RMSE"))

########################################
#Modeling movie & user effects

#Estimate user effects (b_u)
user_avgs <- edx %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu_hat - b_i))
#Predict ratings adjusting for movie & user effects
pred_bu <- final_holdout_test %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  summarize(pred = mu_hat + b_i + b_u) %>%
  pull(pred)
#Calculate RMSE of this model based on movie & user effects
user_rmse = RMSE(final_holdout_test$rating, pred_bu)
#Save the result into result tibble
rmse_result <- add_row(rmse_result, method = "Movie & User Effect", RMSE = user_rmse)
rmse_result %>% kable(col.names = c("Method", "RMSE"))

########################################
#Modeling movie, user & genre effect

#Estimate genre effects (b_g)
genre_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u))
#Predict ratings adjusting for movie, user & genre effects
pred_genre <- final_holdout_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genre_avgs, by = "genres") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  pull(pred)
#Calculate RMSE of this model based on movie, user & genre effects
genre_rmse <- RMSE(final_holdout_test$rating, pred_genre)
#Save the result into result tibble
rmse_result <- add_row(rmse_result, method = "Movie, User & Genre Effect", RMSE = genre_rmse)
rmse_result %>% kable(col.names = c("Method", "RMSE"))

########################################
#Regularized movie, user & genre effects

#Tuning lambda
#Generate a sequence of values for lambda ranging from 0 to 10 with 0.1 interval
lambdas <- seq(0, 10, 0.1)
#Tune the value of optimal lambda
reg_rmses <- sapply(lambdas, function(lambda) {
  b_i <- edx %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu_hat)/(n() + lambda))
  b_u <- edx %>% 
    left_join(b_i, by = "movieId") %>% 
    group_by(userId) %>% 
    summarize(b_u = sum(rating - mu_hat - b_i)/(n() + lambda))
  b_g <- edx %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu_hat - b_i - b_u)/(n() + lambda))
  preds <- edx %>% 
    left_join(b_i, by = "movieId") %>% 
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu_hat + b_i + b_u + b_g) %>% 
    summarize(rmse = RMSE(edx$rating, pred)) %>%
    pull(rmse)
})
#Optimal of lambda, which minimizes RMSE
lambda <- lambdas[which.min(reg_rmses)]

#Check with plotting RMSEs against lambdas
ggplot(mapping = aes(x = lambdas, y = reg_rmses)) + 
  geom_point(alpha = 0.5)

#Estimate regularized movie, user and genre effects with optimized lambda value
b_i_reg <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu_hat)/(n() + lambda))
b_u_reg <- edx %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  group_by(userId) %>% 
  summarize(b_u = sum(rating - mu_hat - b_i)/(n() + lambda))
b_g_reg <- edx %>%
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_hat - b_i - b_u)/(n() + lambda))
#Predict ratings adjusting for regularized movie, user & genre effects
pred_reg <- final_holdout_test %>% 
  left_join(b_i_reg, by = "movieId") %>% 
  left_join(b_u_reg, by = "userId") %>%
  left_join(b_g_reg, by = "genres") %>%
  mutate(pred = mu_hat + b_i + b_u + b_g) %>%
  pull(pred)
#Calculate RMSE of this model based on regularized movie, user & genre effects
reg_rmse <- RMSE(final_holdout_test$rating, pred_reg)
#Save the result into result tibble
rmse_result <- add_row(rmse_result, method = "Regularized Movie, User & Genre Effect", RMSE = reg_rmse)
rmse_result %>% kable(col.names = c("Method", "RMSE"))
