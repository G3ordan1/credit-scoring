library(fairml)
library(fmf)
library(data.table)

set.seed(69)
gmsc <- fread("datasets/GMSC/cs-training.csv", drop = "V1") # cs-test not used as only kaggle has the true probabilities
gmscSplit <- sample(c(TRUE, FALSE), nrow(gmsc), replace = TRUE, prob = c(0.7, 0.3))
gmsc_train <- gmsc[gmscSplit, ]
gmsc_test <- gmsc[!gmscSplit, ]
gmsc_y_train <- gmsc_train[, 1]
gmsc_X_train <- gmsc_train[, -1]
gmsc_y_test <- gmsc_test[, 1]
gmsc_X_test <- gmsc_test[, -1]

# AC There are 6 numerical and 8 categorical attributes, all normalized to [-1,1] v2,3,7,10,13,14 are continuous.
# A6 is only 1,2,3,4,5,7,8,9 no 6 Maybe there should be?
australian <- as.data.table(australian)
ausSplit <- sample(c(TRUE, FALSE), nrow(australian), replace = TRUE, prob = c(0.75, 0.25))
ac_train <- australian[ausSplit, ]
ac_test <- australian[!ausSplit, ]
ac_y_train <- ac_train[, 1]
ac_X_train <- ac_train[, -1]
ac_y_test <- ac_test[, 1]
ac_X_test <- ac_test[, -1]

# df_names <- c("ac_X_train", "ac_y_train", "ac_y_test", "ac_X_test")
#
# for (name in df_names) {
#   df <- get(name)
#   rownames(df) <- NULL
#   assign(name, df)
# }
#
sort(unique(ac_train[, 6]))

tc <- fread("datasets/UCI_Credit_Card.csv", drop = "ID")
tcSplit <- sample(c(TRUE, FALSE), nrow(tc), replace = TRUE, prob = c(0.7, 0.3))
tc_train <- tc[tcSplit, ]
tc_test <- tc[!tcSplit, ]
tc_y_train <- tc_train[, 24]
tc_X_train <- tc_train[, -24]
tc_y_test <- tc_test[, 24]
tc_X_test <- tc_test[, -24]

gc <- as.data.table(german.credit)
gcSplit <- sample(c(TRUE, FALSE), nrow(gc), replace = TRUE, prob = c(0.7, 0.3))
gc_train <- gc[gcSplit, ]
gc_test <- gc[!gcSplit, ]
gc_y_train <- gc_train[, 20]
gc_X_train <- gc_train[, -20]
gc_y_test <- gc_test[, 20]
gc_X_test <- gc_test[, -20]
