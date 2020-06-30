from collections import defaultdict
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import math
import random

users = [
    {"id": 0, "name": "Hero", "age": 18, "genre": "Male", "genre interest": "Female" },
    {"id": 1, "name": "Dunn", "age": 19, "genre": "Male", "genre interest": "Female"},
    {"id": 2, "name": "Chi", "age": 20, "genre": "Male", "genre interest": "Female"},
    {"id": 3, "name": "Thor", "age": 21, "genre": "Male", "genre interest": "Female"},
    {"id": 4, "name": "Clive", "age": 22, "genre": "Male", "genre interest": "Female"},
    {"id": 5, "name": "Sue", "age": 18, "genre": "Female", "genre interest": "Male"},
    {"id": 6, "name": "Kate", "age": 19, "genre": "Female", "genre interest": "Male"},
    {"id": 7, "name": "Anna", "age": 20, "genre": "Female", "genre interest": "Male"},
    {"id": 8, "name": "Carolina", "age": 21, "genre": "Female", "genre interest": "Male"},
    {"id": 9, "name": "Daisy", "age": 22, "genre": "Female", "genre interest": "Male"},
]

friendships = [
    (0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6),
    (3, 7), (3, 8), (4, 9), (4, 0), (5, 1), (5, 2),
    (6, 3), (6, 4), (7, 5), (7, 6), (8, 7), (8, 8),
    (9, 9), (9, 0), (9, 1), (9, 2), (9, 3), (9, 5) 
]

interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),(0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),(1, "Postgres"), 
    (2, "Python"), (2, "Scikit-Learn"), (2, "Scipy"),(2, "Numpy"), (2, "Statsmodel"), (2, "Pandas"), 
    (3, "R"), (3, "Python"),(3, "Statistics"), (3, "Regression"), (3, "Probability"),
    (4, "Machine Learning"), (4, "Regression"), (4, "Decision Trees"),(4, "Libsvm"), 
    (5, "Python"), (5, "R"),(5, "Java"), (5, "C++"),(5, "Haskell"), (5, "Programming Languages"), 
    (6, "Theory"),(7, "Machine Learning"), (7, "Scikit-Learn"), 
    (7, "Mahout"),(7, "Neural Networks"), (8, "Neural Networks"), (8, "Deep Learning"),(8, "Big Data"), (8, "Artificial Intelligence"), (8, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data"),  ]

salaries_and_tenures = [
(83000, 8.7), (88000, 8.1),
(48000, 0.7), (76000, 6),
(69000, 6.5), (76000, 7.5),
(60000, 2.5), (83000, 10),
(48000, 1.9), (63000, 4.2)
]

tenure_and_account_type = [
(0.7, 'paid'),
(1.9,'unpaid'),
(2.5,'paid'),
(4.2,'unpaid'),
(6,'unpaid'),
(6.5,'unpaid'),
(7.5,'unpaid'),
(8.1,'unpaid'),
(8.7,'paid'),
(10,'paid')
]

user_ages = []
user_friends = []
user_minutes = []

for user in users:
    user["friends"] = []

for i, j in friendships:
    users[i]["friends"].append(users[j])
    users[j]["friends"].append(users[i])

def name_of_users (user):
    return user['name']

def number_of_friends (user):
    return len(user['friends'])

for user in users:
    user_ages.append(user["age"])
    user_friends.append(number_of_friends(user))

def age_of_users(prim, sec):
    total_friends = 0
    for user in users:
        if user["age"] > prim and user["age"] <= sec:
            total_friends += number_of_friends(user)
    return total_friends

def number_of_friends_by_genre (user,genre):
    cont = 0
    for friend in user['friends']:
        if(friend['genre']==genre):
            cont=cont+1
    return cont 

def total_number_of_friends_by_genre():
    total_of_friends_by_genre = [0, 0]
    for user in users:
        if user["genre"] == "Male": 
            total_of_friends_by_genre[0] += number_of_friends(user)
        else: 
            total_of_friends_by_genre[1] += number_of_friends(user)
    return total_of_friends_by_genre 

def average_ages(ages):
    return sum (ages) / (len(ages) -1)

def remains_of_average(ages):
    average = average_ages(ages)
    return [x - average for x in ages]

def sum_of_squares(remains):
    return sum(x ** 2 for x in remains)

def variance(ages):
    return sum_of_squares(remains_of_average(ages)) / (len(ages) - 1)

#def standard_deviation(ages):
#   return math.sqrt(variance(ages))

def dot (v, w):
    return sum(v_i * w_i for v_i, w_i in zip (v, w))

def sum_of_squares(v):
    return dot(v, v)

def variance(v):
    mean = sum(v) / len(v)
    return [v_i - mean for v_i in v]

def covariance(x, y):
    n = len(x)
    return dot(variance(x), variance(y)) / (n - 1)

def correlation(x, y):
    standard_deviation_x = math.sqrt(sum_of_squares(variance(x)) / (len(x) - 1))
    standard_deviation_y = math.sqrt(sum_of_squares(variance(y)) / (len(y) - 1))
    if standard_deviation_x > 0 and standard_deviation_y > 0:
        return covariance(x, y) / standard_deviation_y / standard_deviation_x
    return 0

#for user in users:
#	print({user['id']: (number_of_friends_by_genre(user,'Male'), number_of_friends_by_genre(user,'Female')) 
#    })  

user_ids_by_interest = defaultdict(list)
for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

interests_by_user_id = defaultdict(list)
for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

def users_with_common_interests(user):
    return set([
        interested_user_id
        for interest in interests_by_user_id[user["id"]]
        for interested_user_id in user_ids_by_interest[interest]
        if interested_user_id != user["id"]
    ])

def users_with_common_genre_interest(identify):
    return set([
        user["id"]
        for user in users
        if user["id"] != identify["id"] and (identify["genre interest"] == user["genre"])
    ])
    
def users_with_common_interests_and_genre_interest(user):
    return set([
        interests_and_genre_interest_user_id
        for interests_and_genre_interest_user_id in users_with_common_genre_interest(user)
        if interests_and_genre_interest_user_id in users_with_common_interests(user)
    ])

def histogram_total_number_of_friends_by_genre():
    genres = ['Male', 'Female']
    aux_bar= [i for i, _ in enumerate(genres)]
    plt.bar(aux_bar, total_number_of_friends_by_genre())
    plt.title ("Histogram of Total Number of Friends by Genre")
    plt.xlabel('Genres')
    plt.ylabel('Total Number of Friends by Genre')
    plt.xticks(aux_bar, genres)
    plt.show()

def histogram_total_number_of_friends_by_age():
    ages = []
    bars = Counter (user["age"] // 10 * 10 for user in users)
    for x in bars.keys():
        ages.append(age_of_users(x, x+10))
    plt.bar([x for x in bars.keys()],ages)
    plt.title ("Histogram of Total of Number of Friends by Age")
    plt.xlabel('Age')
    plt.ylabel('Number of Friends by Age')
    plt.xticks([10 * i for i in range(10)])
    plt.show()

def tuple_two_lists (number_of_elements):
    for i in range(number_of_elements):
        user_friends.append(random.randint(50, 1000))
    for j in user_friends:
        user_minutes.append(j * 2)
#   print(user_friends)           
#   print(user_minutes)
    return (user_friends, user_minutes)
    
def tuple_two_lists_2 (number_of_elements):
    x = 0
    for i in range(number_of_elements):
        user_friends.append(random.randint(50, 1000))
    for j in user_friends:
        x = random.randint(0, 1)
        if x == 1:
            user_minutes.append(j / -2)
        else:
            user_friends.append(j * -2)
#   print(user_friends)           
#   print(user_minutes)
    return (user_friends, user_minutes)

def tuple_two_lists_3 (number_of_elements):
    x = 0
    for i in range(number_of_elements):
        user_friends.append(random.randint(50, 1000))
    for j in user_friends:
        x = random.randint(0, 1)
        if x == 1:
            user_minutes.append(j / 2)
        else:
            user_friends.append(j * 2) 
#   print(user_friends)           
#   print(user_minutes)
    return (user_friends, user_minutes)

#def calculate_standard_deviation(age):
#    ages = []
#    for user in users:
#        if user['age'] >= age:
#            ages.append(user["age"])
#   print(variance(ages))
#   print(standard_deviation(ages))

def covariance_age_and_number_of_friends(x, y):
    print(covariance(x, y))

def correlation_age_and_number_of_friends(x, y):
    print(correlation(x, y))

def test_data_correlation(number_of_elements):
    test_exercise = tuple_two_lists (number_of_elements)
    print(correlation(test_exercise[0], test_exercise[1]))
    test_exercise_2 = tuple_two_lists_2 (number_of_elements)
    print(correlation(test_exercise_2[0], test_exercise_2[1]))
    test_exercise_3 = tuple_two_lists_3 (number_of_elements)
    print(correlation(test_exercise_3[0], test_exercise_3[1]))

def blank (aux_blank):
     if (aux_blank != ' '):
         return aux_blank.split()
     else:
         return aux_blank

#print (users_with_common_interests(users[0]))

#print (users_with_common_genre_interest(users[0]))

#print (users_with_common_interests_and_genre_interest(users[0]))

name_users = [name_of_users(user) for user in users]
number_friends = [number_of_friends(user) for user in users]
account_types = Counter([account_type for _, account_type in tenure_and_account_type])
interests_quotes = [interest for _, interest in interests]

aux_plot = [i for i, _ in enumerate(users)]
plt.plot(aux_plot, number_friends)
plt.title ("Line Graph of Friends Per User")
plt.xlabel("Name of Users")
plt.ylabel("Number of Friends")
plt.xticks(aux_plot, name_users)
#plt.show()

salaries = []
tenures = []
for s, t in salaries_and_tenures:
     salaries.append(s)
     tenures.append(t)

plt.scatter (salaries, tenures) 
plt.title ("Scatter Chart of Salarie and Tenures")
plt.xlabel ("Salary Value")
plt.ylabel ("Time Experience")
#plt.show()

plt.bar(['Paying', 'Non-Paying'], [account_types['paid'], account_types['unpaid']])
plt.title ("Histogram of Paying and Non-Paying")
plt.xticks(['Paying', 'Non-Paying'])
plt.ylabel("Number of Users")
#plt.show()

words = []
quotes = []
word_quotes = []
for interest in interests_quotes:
     separation = blank(interest)
     if separation == str:
          word_quotes.append(separation)
     else:
        for word in separation:
            word_quotes.append(word)

for i, j in Counter(word_quotes).items():
     words.append(i)
     quotes.append(j)

aux_barh = [i for i, _ in enumerate(words)]
plt.barh(aux_barh, quotes)
plt.title ("Histogram of Words in Interests")
plt.xlabel("Number of Mentions")
plt.ylabel("Words")
plt.yticks(aux_barh, words)
#plt.show()

#histogram_total_number_of_friends_by_genre()

#histogram_total_number_of_friends_by_age()

#calculate_standard_deviation(22)

#print(user_ages)

#print(user_friends)

#print(user_minutes)

covariance_age_and_number_of_friends(user_friends, user_ages)

correlation_age_and_number_of_friends(user_friends, user_ages)

test_data_correlation(10)

