library(e1071)  # Для Naive Bayes
library(caret)  # Для оцінки точності
library(dplyr)  # Для маніпуляцій з даними
library(ggplot2)  # Для побудови графіків

# Функція для навчання та класифікації Naive Bayes
naive_bayes_classifier <- function(train_data, test_data, target) {
  model <- naiveBayes(as.formula(paste(target, "~ .")), data = train_data)
  predictions <- predict(model, test_data)
  return(predictions)
}

# Функція для оцінки точності
evaluate_accuracy <- function(predictions, true_labels) {
  accuracy <- sum(predictions == true_labels) / length(true_labels)
  return(accuracy)
}

data(iris)
iris_data <- iris

wine <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = FALSE)
names(wine) <- c("Class", "Alcohol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280/OD315_of_diluted_wines", "Proline")
wine$Class <- as.factor(wine$Class)

breast_cancer <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header = FALSE)
names(breast_cancer) <- c("ID", "Diagnosis", paste0("V", 1:30))
breast_cancer$Diagnosis <- as.factor(breast_cancer$Diagnosis)
breast_cancer <- breast_cancer %>% select(-ID)  # Видаляємо стовпець з ID

heart_disease <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", header = FALSE)
names(heart_disease) <- c("Age", "Sex", "Cp", "Trestbps", "Chol", "Fbs", "Restecg", "Thalach", "Exang", "Oldpeak", "Slope", "Ca", "Thal", "Target")
heart_disease$Target <- as.factor(ifelse(heart_disease$Target > 0, 1, 0))  # Бінаризуємо цільову змінну

# Функція для створення ансамблю методом bagging
generate_bagging_ensemble <- function(data, target, n_classifiers = 10, sample_size = 0.8) {
  classifiers <- list()
  for (i in 1:n_classifiers) {
    sample_indices <- sample(seq_len(nrow(data)), size = round(sample_size * nrow(data)), replace = TRUE)
    sample_data <- data[sample_indices, ]
    model <- naiveBayes(as.formula(paste(target, "~ .")), data = sample_data)
    classifiers[[i]] <- model
  }
  return(classifiers)
}

# Функція для генерації ансамблю методом cross-validation
generate_cv_ensemble <- function(train_data, target, n_classifiers = 10) {
  folds <- createFolds(train_data[[target]], k = n_classifiers)
  classifiers <- list()
  for (i in seq_along(folds)) {
    fold_train_data <- train_data[-folds[[i]], ]
    classifiers[[i]] <- naiveBayes(as.formula(paste(target, "~ .")), data = fold_train_data)
  }
  return(classifiers)
}

# Просте голосування
simple_voting <- function(classifiers, test_data, target) {
  predictions <- sapply(classifiers, function(clf) predict(clf, test_data))
  final_prediction <- apply(predictions, 1, function(row) {
    names(sort(table(row), decreasing = TRUE))[1]
  })
  return(final_prediction)
}


weighted_voting <- function(classifiers, weights, test_data, target) {
  predictions <- sapply(classifiers, function(clf) predict(clf, test_data))
  weighted_votes <- matrix(0, nrow = nrow(predictions), ncol = length(unique(test_data[[target]])))
  colnames(weighted_votes) <- unique(test_data[[target]])
  for (i in 1:ncol(predictions)) {
    for (j in 1:nrow(predictions)) {
      weighted_votes[j, predictions[j, i]] <- weighted_votes[j, predictions[j, i]] + weights[i]
    }
  }
  final_prediction <- apply(weighted_votes, 1, function(row) {
    colnames(weighted_votes)[which.max(row)]
  })
  return(final_prediction)
}

# (Weighted Majority)
weighted_majority <- function(classifiers, test_data, target, eta = 0.5) {
  n <- length(classifiers)
  weights <- rep(1, n)
  predictions <- sapply(classifiers, function(clf) predict(clf, test_data))
  final_prediction <- rep(NA, nrow(test_data))
  for (i in 1:nrow(test_data)) {
    weighted_votes <- rep(0, length(unique(test_data[[target]])))
    names(weighted_votes) <- unique(test_data[[target]])
    for (j in 1:n) {
      weighted_votes[predictions[i, j]] <- weighted_votes[predictions[i, j]] + weights[j]
    }
    final_prediction[i] <- names(which.max(weighted_votes))
    if (final_prediction[i] != test_data[i, target]) {
      weights <- weights * (eta ^ (predictions[i, ] != test_data[i, target]))
    }
  }
  return(final_prediction)
}

# Функція для проведення експериментів
experiment <- function(data, target, train_sizes) {
  results <- data.frame()
  
  for (train_size in train_sizes) {
    set.seed(123)
    train_indices <- sample(seq_len(nrow(data)), size = round(train_size * nrow(data)))
    train_data <- data[train_indices, ]
    test_data <- data[-train_indices, ]
    
    # Базовий класифікатор
    base_predictions <- naive_bayes_classifier(train_data, test_data, target)
    base_accuracy <- evaluate_accuracy(base_predictions, test_data[[target]])
    results <- rbind(results, data.frame(train_size, Method = "Base Classifier", Accuracy = base_accuracy))
    
    # Ансамбль методом bagging з простим голосуванням
    bagging_ensemble <- generate_bagging_ensemble(train_data, target)
    bagging_predictions <- simple_voting(bagging_ensemble, test_data, target)
    bagging_accuracy <- evaluate_accuracy(bagging_predictions, test_data[[target]])
    results <- rbind(results, data.frame(train_size, Method = "Bagging + Simple Voting", Accuracy = bagging_accuracy))
    
    # Ансамбль методом cross-validation з простим голосуванням
    cv_ensemble <- generate_cv_ensemble(train_data, target)
    cv_predictions <- simple_voting(cv_ensemble, test_data, target)
    cv_accuracy <- evaluate_accuracy(cv_predictions, test_data[[target]])
    results <- rbind(results, data.frame(train_size, Method = "Cross-Validation + Simple Voting", Accuracy = cv_accuracy))
    
    # Взвешене голосування для bagging
    bagging_weights <- sapply(bagging_ensemble, function(clf) {
      pred <- predict(clf, train_data)
      acc <- sum(pred == train_data[[target]]) / nrow(train_data)
      return(acc)
    })
    bagging_weighted_predictions <- weighted_voting(bagging_ensemble, bagging_weights, test_data, target)
    bagging_weighted_accuracy <- evaluate_accuracy(bagging_weighted_predictions, test_data[[target]])
    results <- rbind(results, data.frame(train_size, Method = "Bagging + Weighted Voting", Accuracy = bagging_weighted_accuracy))
    
    # Взвешене голосування для cross-validation
    cv_weights <- sapply(cv_ensemble, function(clf) {
      pred <- predict(clf, train_data)
      acc <- sum(pred == train_data[[target]]) / nrow(train_data)
      return(acc)
    })
    cv_weighted_predictions <- weighted_voting(cv_ensemble, cv_weights, test_data, target)
    cv_weighted_accuracy <- evaluate_accuracy(cv_weighted_predictions, test_data[[target]])
    results <- rbind(results, data.frame(train_size, Method = "Cross-Validation + Weighted Voting", Accuracy = cv_weighted_accuracy))
  }
  
  return(results)
}

# Параметри експерименту
train_sizes <- seq(0.5, 0.9, by = 0.1)

# Проведення експериментів для кожного набору даних
iris_results <- experiment(iris_data, "Species", train_sizes)
wine_results <- experiment(wine, "Class", train_sizes)
breast_cancer_results <- experiment(breast_cancer, "Diagnosis", train_sizes)
heart_disease_results <- experiment(heart_disease, "Target", train_sizes)

# Функція для побудови графіків результатів
plot_results <- function(results, title) {
  ggplot(results, aes(x = train_size * 100, y = Accuracy, color = Method)) +
    geom_line() + geom_point() +
    labs(title = title, x = "Training Set Size (%)", y = "Accuracy") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Побудова графіків результатів для кожного набору даних
plot_results(iris_results, "Iris Dataset - Naive Bayes with Ensembles")
plot_results(wine_results, "Wine Dataset - Naive Bayes with Ensembles")
plot_results(breast_cancer_results, "Breast Cancer Dataset - Naive Bayes with Ensembles")
plot_results(heart_disease_results, "Heart Disease Dataset - Naive Bayes with Ensembles")

cat("\014")
rm(list=ls())
