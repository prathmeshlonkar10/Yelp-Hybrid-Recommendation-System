# ---------- Method Description ---------- #
"""
The recommendation system uses item-based collaborative filtering along with the Model-based recommendation system. 
The predictions of these 2 recommendation systems are then combined using the weighted hybrid recommendation system.
I decided to use model-based CF as the primary recommender because it gave me better performance over the item-based CF.
I have added additional 11 user features from the user.json dataset, and also considered the features from tip.json dataset.

These additional features are as follows:
User features:
1) compliment_hot
2) compliment_more
3) compliment_profile
4) compliment_cute
5) compliment_list
6) compliment_note
7) compliment_plain
8) compliment_cool
9) compliment_funny
10) compliment_writer
11) compliment_photos

Tip features:
1) likes

These features give a better idea about the user profile and helps in making better predictions for the user.
And it helped me to improve my RMSE score as compared to HW3.
"""
# ---------- Error Distribution ---------- #
"""
>=0 and <1: 102240
>=1 and <2: 32784
>=2 and <3: 6193
>=3 and <4: 825
>=4: 2
"""
# ---------- RMSE ---------- #
"""
0.9798584200645939
"""
# ---------- Execution time ----------- #
"""
369.84602546691895s
"""
# Please note that the error distribution, RMSE, and execution time has been calculated on the validation dataset.  

# ========== CODE START ========== #

# Importing the spark context and other libraries.
import os
import sys
import time
import json
import numpy as np
from pyspark import SparkContext
from xgboost import XGBRegressor

# Initializing python version.
#os.environ['PYSPARK_PYTHON'] = sys.executable
#os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Accessing system variables.
train_folderpath = sys.argv[1]
test_filepath = sys.argv[2]
output_filepath = sys.argv[3]

# Creating the spark context.
sc = SparkContext("local[*]", "Task_Competition_Hybrid")
sc.setLogLevel("ERROR")

# Ideation.
# For the hybrid recommendation, I have considered implementing the weighted hybrid method.
# In this, we will consider an alpha, and taken a weighted sum of item-based RS and model-based RS.
# First, we will find the item-based prediction result, then the model-based prediction result.
# Post that, we will combine these 2 prediction results.

# ========== Function Definitions ========== #
def rdd_to_dict(rdd, depth=1):
    if depth == 1:
        # Converting the rdd with key and value (value-set) format to a re-usable dictionary format.
        new_dict = {}
        for key, value in rdd.collect():
            new_dict[key] = value
        return new_dict
    else:
        # Converting the rdd with key and tuple (depth=2) value format to a re-usable dictionary format.
        new_dict = {}
        for key, t_val in rdd.collect():
            tmp_dict = {}
            for val in t_val:
                tmp_dict[val[0]] = val[1]
            new_dict[key] = tmp_dict
        return new_dict

def item_CF_pred(t_bsn, t_usr):
    # This function handles the item-based Collaborative Filtering process.
    # Pearson correlation is used for the weight calculation.
    # Default rating for cold start issue is kept as 3.0

    # ========== COLD START ========== #
    # The cold start can occur when the user or the business does not exist in the training data (utility matrix) but is present in the test dataset.
    if t_usr not in usr_bsn_dict.keys():
        # If the test user does not exist in training data (means that this user has never given rating before), we will assign a default rating.
        return 3.0
    if t_bsn not in bsn_usr_dict.keys():
        # If the test business does not exist in training data (means no user has rated this business before), we can consider the average rating of the test user.
        return usr_avg_dict[t_usr]

    # ========== Weights Calculation ========== #
    # We need to calulate weights (Pearson correlation) for the weighted average (rating prediction).
    # We need to calculate this weight for every other business (all pairs of other businessID with test businessID) that the target user has rated.

    # List to capture required weights for current test data (businessID & userID).
    cur_wgt_list = []

    # Accessing each businessID rated by thee current user.
    for i_bsn in usr_bsn_dict[t_usr]:
        # Creating pair for weights calculation.
        bsn_pair = tuple(sorted((t_bsn, i_bsn)))
        if bsn_pair in bsn_wgt.keys():
            # If the weight for this pair has already been calculated for some other test data, good to pick from the master dictionary.
            cur_wgt = bsn_wgt[bsn_pair]
        else:
            # If not, we need to calculate it.
            # To calculate, we will consider the co-rated users for this pair.
            # Means users that have rated both these items (businessIDs).
            cr_usr = bsn_usr_dict[t_bsn].intersection(bsn_usr_dict[i_bsn])

            # Now, we can calculate the pearson correlation. 
            # However, we need atleast 2 users who have co-rated the business-pair to calculate the pearson correlation.
            # If there is just 1, we can calculate the weight using similarity based on absolute difference in average ratings for the business.
            if len(cr_usr) <= 1:
                cur_wgt = (5.0 - abs(bsn_avg_dict[t_bsn] - bsn_avg_dict[i_bsn])) / 5
            
            # Altough we can use Pearson correlation for 2 users, but using similarity is more robust for sparse utility matrix.
            elif len(cr_usr) == 2:
                cr_usr = list(cr_usr)
                w_1 = (5.0 - abs(float(bsn_usr_rtg_dict[t_bsn][cr_usr[0]]) - float(bsn_usr_rtg_dict[i_bsn][cr_usr[0]]))) / 5
                w_2 = (5.0 - abs(float(bsn_usr_rtg_dict[t_bsn][cr_usr[1]]) - float(bsn_usr_rtg_dict[i_bsn][cr_usr[1]]))) / 5
                cur_wgt = (w_1 + w_2) / 2
            else:
                # Pearson correlation calculation.
                # For this, we will need ratings for the co-rated users.
                usr_rtg_1 = []
                usr_rtg_2 = []

                # Accessing each co-rated user.
                for i_usr in cr_usr:
                    # Collecting the list of all the co-rated ratings for both businessID.
                    usr_rtg_1.append(float(bsn_usr_rtg_dict[t_bsn][i_usr]))
                    usr_rtg_2.append(float(bsn_usr_rtg_dict[i_bsn][i_usr]))
                
                # We can calculate the co-rated average now.
                avg_1 = sum(usr_rtg_1) / len(usr_rtg_1)
                avg_2 = sum(usr_rtg_2) / len(usr_rtg_2)

                # We can now implement the formula for pearson correlation.
                numerator = sum(r_1 * r_2 for r_1, r_2 in zip([r1 - avg_1 for r1 in usr_rtg_1], [r2 - avg_2 for r2 in usr_rtg_2]))
                denominator = ((sum([(r1 - avg_1)**2 for r1 in usr_rtg_1]))**0.5) * ((sum([(r2 - avg_2)**2 for r2 in usr_rtg_2]))**0.5)

                # Check for denominator.
                if denominator == 0:
                    cur_wgt = 0
                else:
                    cur_wgt = numerator / denominator

            # Storing the weight for future references.
            bsn_wgt[bsn_pair] = cur_wgt

        # Storing the obtained weight to the list along with the rating for current business given by test user.
        cur_wgt_list.append((float(bsn_usr_rtg_dict[i_bsn][t_usr]), cur_wgt))
    
    # Instead of all the weight values, we will consider only the top 15 values.
    cur_wgt_list_std = sorted(cur_wgt_list, key=lambda val: -val[1])[:15]
    
    # Now we can implement the formula for Prediction (weighted average of ratings).
    numerator = sum([rtg * wgt for rtg, wgt in cur_wgt_list_std])
    denominator = sum([abs(wgt) for rtg, wgt in cur_wgt_list_std])

    # Check for denominator.
    if denominator == 0:
        # Return default value if denominator is 0.
        return 3.0
    else:
        # Prediction for rating.
        return numerator / denominator

def dataset_feature_check(usr, bsn, json_review_dict, json_usr_dict, json_bsn_dict, json_tip_dict):
    # user review features.
    if usr in json_review_dict.keys():
        rtg_useful_cnt = json_review_dict[usr][0]
        rtg_funny_cnt = json_review_dict[usr][1]
        rtg_cool_cnt = json_review_dict[usr][2]
    else:
        rtg_useful_cnt = None
        rtg_funny_cnt = None
        rtg_cool_cnt = None
    
    # user features.
    if usr in json_usr_dict.keys():
        usr_review_cnt = json_usr_dict[usr][0]
        usr_fans = json_usr_dict[usr][1]
        usr_avg_stars = json_usr_dict[usr][2]
        if json_usr_dict[usr][3][0] == "None":
            usr_elite_years = 0
        else:
            usr_elite_years = len(json_usr_dict[usr][3])
        usr_useful = json_usr_dict[usr][4]
        usr_funny = json_usr_dict[usr][5]
        usr_cool = json_usr_dict[usr][6]
        usr_compliment_hot = json_usr_dict[usr][7]
        usr_compliment_more = json_usr_dict[usr][8]
        usr_compliment_profile = json_usr_dict[usr][9]
        usr_compliment_cute = json_usr_dict[usr][10]
        usr_compliment_list = json_usr_dict[usr][11]
        usr_compliment_note = json_usr_dict[usr][12]
        usr_compliment_plain = json_usr_dict[usr][13]
        usr_compliment_cool = json_usr_dict[usr][14]
        usr_compliment_funny = json_usr_dict[usr][15]
        usr_compliment_writer = json_usr_dict[usr][16]
        usr_compliment_photos = json_usr_dict[usr][17]
    else:
        usr_review_cnt = None
        usr_fans = None
        usr_avg_stars = None
        usr_elite_years = None
        usr_useful = None
        usr_funny = None
        usr_cool = None
        usr_compliment_hot = None
        usr_compliment_more = None
        usr_compliment_profile = None
        usr_compliment_cute = None
        usr_compliment_list = None
        usr_compliment_note = None
        usr_compliment_plain = None
        usr_compliment_cool = None
        usr_compliment_funny = None
        usr_compliment_writer = None
        usr_compliment_photos = None

    # business features.
    if bsn in json_bsn_dict.keys():
        bsn_stars = json_bsn_dict[bsn][0]
        bsn_review_cnt = json_bsn_dict[bsn][1]
        bsn_latitude = json_bsn_dict[bsn][2]
        bsn_longitude = json_bsn_dict[bsn][3]
    else:
        bsn_stars = None
        bsn_review_cnt = None
        bsn_latitude = None
        bsn_longitude = None

    # tip features.
    if usr in json_tip_dict.keys():
        tip_likes_cnt = json_tip_dict[usr]
    else:
        tip_likes_cnt = None
    
    # Item-based prediction feature.
    itm_pred_rtg = item_CF_pred(bsn, usr)

    return [rtg_useful_cnt, rtg_funny_cnt, rtg_cool_cnt, usr_review_cnt, usr_fans, usr_avg_stars, usr_elite_years, usr_useful, usr_funny, usr_cool, usr_compliment_hot, usr_compliment_more, usr_compliment_profile, usr_compliment_cute, usr_compliment_list, usr_compliment_note, usr_compliment_plain, usr_compliment_cool, usr_compliment_funny, usr_compliment_writer, usr_compliment_photos, bsn_stars, bsn_review_cnt, bsn_latitude, bsn_longitude, tip_likes_cnt, itm_pred_rtg]

def model_CF_pred(csv_train_rdd):
    # Ideation.
    # For model-based recommendation system, we will need to utilize the additional user and business parameters to predict the ratings.
    # For this, I have used the user & business parameters from review, user, & business files.
    # We will build an XGBRegressor model from this feature data and predict the ratings to develop the model-based Recommendation system.

    # The csv_train_rdd dataset has only userID, businessID, and ratings information, and no features information. 
    # Let's utilize other datasets (review, user, business) for feature information.

    # ========== Building DATA features ========== #

    # Picking businessID, useful, funny, cool as the attributes from review_train.json file.
    json_review = sc.textFile(train_folderpath + "/review_train.json").map(lambda line: json.loads(line)).map(lambda line: (line["user_id"], (int(line["useful"]), int(line["funny"]), int(line["cool"])))).groupByKey().mapValues(list)
    #print("json_review head", json_review.take(5))

    # Converting the rdd to dictionary for re-usable format.
    json_review_dict = {}
    for usr, val_list in json_review.collect():
        # We will take total counts of useful, funny, and cool for each userID because there are too many values for each userID.
        useful_cnt = 0
        funny_cnt = 0
        cool_cnt = 0
        #total_cnt = 0
        for each_val in val_list:
            useful_cnt = useful_cnt + each_val[0]
            funny_cnt = funny_cnt + each_val[1]
            cool_cnt = cool_cnt + each_val[2]
            #total_cnt = total_cnt + 1
        json_review_dict[usr] = (useful_cnt, funny_cnt, cool_cnt)
    #print("json_review_dict", json_review_dict)

    # Similarly, creating relevant feature-set with user & business data.
    json_usr = sc.textFile(train_folderpath + "/user.json").map(lambda line: json.loads(line)).map(lambda line: (line["user_id"], (int(line["review_count"]), int(line["fans"]), float(line["average_stars"]), line["elite"].split(","), int(line["useful"]), int(line["funny"]), int(line["cool"]), int(line["compliment_hot"]), int(line["compliment_more"]), int(line["compliment_profile"]), int(line["compliment_cute"]), int(line["compliment_list"]), int(line["compliment_note"]), int(line["compliment_plain"]), int(line["compliment_cool"]), int(line["compliment_funny"]), int(line["compliment_writer"]), int(line["compliment_photos"]))))
    json_bsn = sc.textFile(train_folderpath + "/business.json").map(lambda line: json.loads(line)).map(lambda line: (line["business_id"], (float(line["stars"]), int(line["review_count"]), line["latitude"], line["longitude"], int(line["is_open"]))))
    json_tip = sc.textFile(train_folderpath + "/tip.json").map(lambda line: json.loads(line)).map(lambda line: (line["user_id"], (int(line["likes"]))))

    # Converting user and business rdds to dictionary for re-usable format.
    json_usr_dict = rdd_to_dict(json_usr)
    json_bsn_dict = rdd_to_dict(json_bsn)
    json_tip_dict = rdd_to_dict(json_tip)

    # ========== TRAINING DATA Preparation ========== #

    # Now that the features are collected from the respective datasets, we can merge all this information to create the training set.
    train_X = []
    train_Y = []

    # Looping through the train set.
    for usr, bsn, rtg in csv_train_rdd.collect():
        # Collecting the rating value as Y-label.
        train_Y.append(rtg)
        
        # Consoliadting all features.
        train_X.append(dataset_feature_check(usr, bsn, json_review_dict, json_usr_dict, json_bsn_dict, json_tip_dict))

    # Creating the numpy arrays for training set.
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    # ========== TESTING DATA Preparation ========== #

    # Now let's prepare the test dataset in the same way.
    csv_test_rdd = sc.textFile(test_filepath)
    # Separating the header.
    csv_test_header = csv_test_rdd.first()
    csv_test_rdd = csv_test_rdd.filter(lambda line: line != csv_test_header).map(lambda line: (line.split(",")[0], line.split(",")[1]))

    test_X = []
    usr_bsn_list = []

    for usr, bsn in csv_test_rdd.collect():
        usr_bsn_list.append((usr, bsn))

        # Consoliadting all features.
        test_X.append(dataset_feature_check(usr, bsn, json_review_dict, json_usr_dict, json_bsn_dict, json_tip_dict))

    test_X = np.array(test_X)

    # ========== XGBRegressor Model ========== #

    # Creating an instance of the XGBRegressor.
    model = XGBRegressor(n_estimators=350, eta=0.01, max_depth=8, random_state=10, subsample=0.8, colsample_bytree=0.6, reg_alpha=0.005, reg_lambda=0.01, min_child_weight=100)

    # Fitting the model on training data.
    model.fit(train_X, train_Y)

    # Predicting the test data.
    pred_test_Y = model.predict(test_X)

    # Body for writing the CSV output result.
    #output_body = "user_id, business_id, prediction\n"

    # Storing the output in the required format.
    #for idx in range(len(usr_bsn_list)):
    #    output_body = output_body + "{},{},{}\n".format(usr_bsn_list[idx][0], usr_bsn_list[idx][1], pred_test_Y[idx])
    
    return usr_bsn_list, pred_test_Y

def calculate_ed_and_rmse():
    # Taking the predictions from the output file.
    with open(output_filepath) as ifp:
        pred = ifp.readlines()[1:]

    # Taking the ground truth from test file (can be validation file as well).
    with open(test_filepath) as ifp:
        truth = ifp.readlines()[1:]

    # Making error distribution brackets.
    err_dist = {">=0 and <1": 0, ">=1 and <2": 0, ">=2 and <3": 0, ">=3 and <4": 0, ">=4": 0}

    # Initializing RMSE value.
    sq_err = 0

    # Calculation for error distribution and RMSE.
    for i in range(len(pred)):
        err = float(pred[i].split(",")[2]) - float(truth[i].split(",")[2])
        sq_err += err ** 2

        # Error distribution.
        if 1 > abs(err):
            err_dist[">=0 and <1"] += 1
        elif 2 > abs(err) >= 1:
            err_dist[">=1 and <2"] += 1
        elif 3 > abs(err) >= 2:
            err_dist[">=2 and <3"] += 1
        elif 4 > abs(err) >= 3:
            err_dist[">=3 and <4"] += 1
        else:
            err_dist[">=4"] += 1

    # Calculating actual rmse.
    rmse = (sq_err / (len(pred))) ** (0.5)

    # Printing the calculation summary.
    print("\nError Distribution:")
    for key, val in err_dist.items():
        print("{}: {}".format(key, val))
    
    print("\nRMSE:\n{}".format(rmse))

# ========== TIMER START ========== #
st = time.time()

# ========== ITEM_BASED RECOMMENDATION SYSTEM ========== #

# Input file.
csv_rdd = sc.textFile(train_folderpath + "/yelp_train.csv")

# Separating the header.
csv_header = csv_rdd.first()
csv_rdd = csv_rdd.filter(lambda line: line != csv_header).map(lambda line: line.split(","))
#print("csv_rdd head", csv_rdd.take(5))

# Ideation.
# In Item-based CF, we will have to create a Utility matrix of business IDs (items) and user IDs (user).
# This matrix will contain the ratings provided by that user to that item.
# This matrix will act as our training data.
# To predict the rating of a business, we have to calculate pearson correlation (weights) of that item with other items.
# Then take a weighted average across the other ratings given by the target user.

# Spliting the CSV to obtain businessID, userID, and user-ratings.
csv_split_rdd = csv_rdd.map(lambda line: (line[1], line[0], line[2]))

# ========== Utility Matrix ========== #
# Along with utility matrix, we will create a few more variables to capture the relevant information as required.

# Accessing business, user, and ratings values from splitted CSV rdd.
bsn_usr = csv_split_rdd.map(lambda line: (line[0], line[1])).groupByKey().mapValues(set)
usr_bsn = csv_split_rdd.map(lambda line: (line[1], line[0])).groupByKey().mapValues(set)
#print("bsn_usr head", bsn_usr.take(5))
#print("usr_bsn head", usr_bsn.take(5))

# Converting the bsn_usr and usr_bsn rdd to dictionary (re-usable format).
bsn_usr_dict = rdd_to_dict(bsn_usr)
usr_bsn_dict = rdd_to_dict(usr_bsn)
#print("bsn_usr_dict", bsn_usr_dict)
#print("usr_bsn_dict", usr_bsn_dict)

# Collecting the average scores across businessID and userID to use as a default case.
bsn_avg = csv_split_rdd.map(lambda line: (line[0], float(line[2]))).aggregateByKey((0,0), (lambda x, y: (x[0]+y, x[1]+1)), (lambda x, y: (x[0]+y[0], x[1]+y[1]))).map(lambda line: (line[0], line[1][0]/line[1][1]))
usr_avg = csv_split_rdd.map(lambda line: (line[1], float(line[2]))).aggregateByKey((0,0), (lambda x, y: (x[0]+y, x[1]+1)), (lambda x, y: (x[0]+y[0], x[1]+y[1]))).map(lambda line: (line[0], line[1][0]/line[1][1]))
#print("bsn_avg head", bsn_avg.take(5))
#print("usr_avg head", usr_avg.take(5))

# Converting the bsn_avg and usr_avg rdd to dictionary (re-usable format).
bsn_avg_dict = rdd_to_dict(bsn_avg)
usr_avg_dict = rdd_to_dict(usr_avg)
#print("bsn_avg_dict", bsn_avg_dict)
#print("usr_avg_dict", usr_avg_dict)

# Collecting ratings for businessID given by the corresponding users.
# This is the actual UTILITY MATRIX.
bsn_usr_rtg = csv_split_rdd.map(lambda line: (line[0], (line[1], line[2]))).groupByKey().mapValues(set)
#print("bsn_usr_rtg head", bsn_usr_rtg.take(5))

# Converting the bsn_usr_rtg rdd to dictionary (re-usable format).
bsn_usr_rtg_dict = rdd_to_dict(bsn_usr_rtg, 2)
#print("bsn_usr_rtg_dict", bsn_usr_rtg_dict)

# ========== Test-data rating prediction ========== #

# Input test file (This can also be validation file).
test_csv_rdd = sc.textFile(test_filepath)

# Separating the header.
test_csv_header = test_csv_rdd.first()
test_csv_rdd = test_csv_rdd.filter(lambda line: line != test_csv_header)
#print("test_csv_rdd head", test_csv_rdd.take(5))

# Spliting test CSV and accessing businessID and userID values.
test_bsn_usr = test_csv_rdd.map(lambda line: (line.split(",")[1], line.split(",")[0]))

# Ideation.
# Now the pair of businessID and userID in test_bsn_usr is our testing data.
# We need to predict the user-rating for this pair.
# We will calculate the weights using pearson correlation and take weighted average to predict.

# Dictionary to store the weights of businessID pairs.
bsn_wgt = {}

# Accessing each line of test data and predicting the rating with Item-based CF RS and storing in variable.
item_CF_pred_list = []
for line in test_bsn_usr.collect():
    pred_rtg = item_CF_pred(line[0], line[1])
    item_CF_pred_list.append(pred_rtg)

# ========== MODEL-BASED RECOMMENDATION SYSTEM ========== #
    
usr_bsn_list, pred_test_Y = model_CF_pred(csv_rdd)

# ========== Prediction Output ========== #

# Body for writing the CSV output result.
output_body = "user_id, business_id, prediction\n"

# Alpha factor for weighted score.
alpha = 0.06

# Storing the output in the required format.
for idx in range(len(usr_bsn_list)):
    final_score = alpha*item_CF_pred_list[idx] + (1 - alpha)*pred_test_Y[idx]
    output_body = output_body + "{},{},{}\n".format(usr_bsn_list[idx][0], usr_bsn_list[idx][1], final_score)

# Saving the output file.
with open(output_filepath, "w") as ofp:
    ofp.write(output_body)

# ========== TIMER END ========== #
et = time.time()

# Print error distribution and RMSE value.
calculate_ed_and_rmse()

# Print the duration on console.
print("\nDuration:", et - st)

# ========== CODE END ========== #
