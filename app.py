from flask import Flask
from flask import jsonify
from flask import request
import random

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from nltk.tokenize import sent_tokenize
from watson_developer_cloud import PersonalityInsightsV3
from google.cloud import language
import numpy as np
import pickle
import pandas as pd
import os


app = Flask(__name__)



@app.before_first_request
def load_huge_file():
    credentials()
    global tot_words
    tot_words = np.load('Total_list_words.npy')
    global mean
    mean = np.load('Mean_normalization_parameters.npy')
    global std
    std = np.load('Std_normalization_parameters.npy')
    global lr
    lr = pickle.load(open('Logistic_model_weights.sav', 'rb'))
    global knn
    knn = pickle.load(open('KNN_model_n_value.sav', 'rb'))
    # Do some things and assign data from a large file to loaded_data
    print("Model object loaded!")
    # objects = []
    # with (open(model_path, "rb")) as openfile:
    #     while True:
    #         try:
    #             objects.append(pickle.load(openfile))
    #         except EOFError:
    #             break
    # global clf
    # clf = objects[0]


@app.route('/health', methods=['GET'])
def health():
    """
    Does Health Check
    """
    response_object = {"success": "1","project":"ritiksparser"}
    response = jsonify(response_object)
    response.status_code = 200
    print('Health OK')
    return response

@app.route('/', methods=['GET'])
def index():
    """
    Does Health Check
    """
    response_object = {"Hello":"World", "success": "1","project":"ritiksparser"}
    response = jsonify(response_object)
    response.status_code = 200
    print('Default OK')
    return response

@app.route('/getdummuystats', methods=['POST'])
def getdummuystats():
    """
    Does Health Check
    """
    json_object = request.get_json()
    response_object = {
        "response": "getdummuystats",
        "project": "ritiksparser",
        "sentiment" : {
            "positive": "{:.4f}".format(random.uniform(0, 1)),
            "negative": "{:.4f}".format(random.uniform(0, 1)),
            "neutral": "{:.4f}".format(random.uniform(0, 1))
        },
        "predict": [{
            "label": "KNN",
            "value": "{:.4f}".format(random.uniform(0, 1))
        }, {
            "label": "Logistic Regression",
            "value": "{:.4f}".format(random.uniform(0, 1))
        }],
        "word_freq": [{
            "label": "Machine Learning",
            "count": random.randint(10,1000)
        }, {
            "label": "Word2",
            "count": random.randint(10,1000)
        }, {
            "label": "Word3",
            "count": random.randint(10,1000)
        }, {
            "label": "Word4",
            "count": random.randint(10,1000)
        }, {
            "label": "Word5",
            "count": random.randint(10,1000)
        }, {
            "label": "Word6",
            "count": random.randint(10,1000)
        }, {
            "label": "Word7",
            "count": random.randint(10,1000)
        }, {
            "label": "Word8",
            "count": 5
        }],
        "ibm": {
            "word_count": 7020,
            "processed_language": "en",
            "personality": [
                {
                    "trait_id": "big5_openness",
                    "name": "Openness",
                    "category": "personality",
                    "percentile": 0.9436207034687292,
                    "children": [
                        {
                            "trait_id": "facet_adventurousness",
                            "name": "Adventurousness",
                            "category": "personality",
                            "percentile": 0.8728046722494278
                        },
                        {
                            "trait_id": "facet_artistic_interests",
                            "name": "Artistic interests",
                            "category": "personality",
                            "percentile": 0.4184832909070394
                        },
                        {
                            "trait_id": "facet_emotionality",
                            "name": "Emotionality",
                            "category": "personality",
                            "percentile": 0.1583657795716475
                        },
                        {
                            "trait_id": "facet_imagination",
                            "name": "Imagination",
                            "category": "personality",
                            "percentile": 0.04481189401582769
                        },
                        {
                            "trait_id": "facet_intellect",
                            "name": "Intellect",
                            "category": "personality",
                            "percentile": 0.996393471977502
                        },
                        {
                            "trait_id": "facet_liberalism",
                            "name": "Authority-challenging",
                            "category": "personality",
                            "percentile": 0.8835343648439694
                        }
                    ]
                },
                {
                    "trait_id": "big5_conscientiousness",
                    "name": "Conscientiousness",
                    "category": "personality",
                    "percentile": 0.8717383918743111,
                    "children": [
                        {
                            "trait_id": "facet_achievement_striving",
                            "name": "Achievement striving",
                            "category": "personality",
                            "percentile": 0.8987448251947685
                        },
                        {
                            "trait_id": "facet_cautiousness",
                            "name": "Cautiousness",
                            "category": "personality",
                            "percentile": 0.9712886050822256
                        },
                        {
                            "trait_id": "facet_dutifulness",
                            "name": "Dutifulness",
                            "category": "personality",
                            "percentile": 0.8133994248325889
                        },
                        {
                            "trait_id": "facet_orderliness",
                            "name": "Orderliness",
                            "category": "personality",
                            "percentile": 0.2877418132557438
                        },
                        {
                            "trait_id": "facet_self_discipline",
                            "name": "Self-discipline",
                            "category": "personality",
                            "percentile": 0.8532835038541257
                        },
                        {
                            "trait_id": "facet_self_efficacy",
                            "name": "Self-efficacy",
                            "category": "personality",
                            "percentile": 0.907224670713983
                        }
                    ]
                },
                {
                    "trait_id": "big5_extraversion",
                    "name": "Extraversion",
                    "category": "personality",
                    "percentile": 0.633776377853311,
                    "children": [
                        {
                            "trait_id": "facet_activity_level",
                            "name": "Activity level",
                            "category": "personality",
                            "percentile": 0.8385116473609877
                        },
                        {
                            "trait_id": "facet_assertiveness",
                            "name": "Assertiveness",
                            "category": "personality",
                            "percentile": 0.9962249573928508
                        },
                        {
                            "trait_id": "facet_cheerfulness",
                            "name": "Cheerfulness",
                            "category": "personality",
                            "percentile": 0.2705370138030023
                        },
                        {
                            "trait_id": "facet_excitement_seeking",
                            "name": "Excitement-seeking",
                            "category": "personality",
                            "percentile": 0.038378637486580935
                        },
                        {
                            "trait_id": "facet_friendliness",
                            "name": "Outgoing",
                            "category": "personality",
                            "percentile": 0.654047666938657
                        },
                        {
                            "trait_id": "facet_gregariousness",
                            "name": "Gregariousness",
                            "category": "personality",
                            "percentile": 0.31931224372899314
                        }
                    ]
                },
                {
                    "trait_id": "big5_agreeableness",
                    "name": "Agreeableness",
                    "category": "personality",
                    "percentile": 0.2164721952851401,
                    "children": [
                        {
                            "trait_id": "facet_altruism",
                            "name": "Altruism",
                            "category": "personality",
                            "percentile": 0.8838134070911637
                        },
                        {
                            "trait_id": "facet_cooperation",
                            "name": "Cooperation",
                            "category": "personality",
                            "percentile": 0.7023301790139969
                        },
                        {
                            "trait_id": "facet_modesty",
                            "name": "Modesty",
                            "category": "personality",
                            "percentile": 0.217840467109156
                        },
                        {
                            "trait_id": "facet_morality",
                            "name": "Uncompromising",
                            "category": "personality",
                            "percentile": 0.9236372740673224
                        },
                        {
                            "trait_id": "facet_sympathy",
                            "name": "Sympathy",
                            "category": "personality",
                            "percentile": 0.9873886400757081
                        },
                        {
                            "trait_id": "facet_trust",
                            "name": "Trust",
                            "category": "personality",
                            "percentile": 0.49817604319980757
                        }
                    ]
                },
                {
                    "trait_id": "big5_neuroticism",
                    "name": "Emotional range",
                    "category": "personality",
                    "percentile": 0.9529868730168436,
                    "children": [
                        {
                            "trait_id": "facet_anger",
                            "name": "Fiery",
                            "category": "personality",
                            "percentile": 0.009729046238298733
                        },
                        {
                            "trait_id": "facet_anxiety",
                            "name": "Prone to worry",
                            "category": "personality",
                            "percentile": 0.01077626438615753
                        },
                        {
                            "trait_id": "facet_depression",
                            "name": "Melancholy",
                            "category": "personality",
                            "percentile": 0.1715496625869376
                        },
                        {
                            "trait_id": "facet_immoderation",
                            "name": "Immoderation",
                            "category": "personality",
                            "percentile": 0.12963720671866708
                        },
                        {
                            "trait_id": "facet_self_consciousness",
                            "name": "Self-consciousness",
                            "category": "personality",
                            "percentile": 0.06943652192683758
                        },
                        {
                            "trait_id": "facet_vulnerability",
                            "name": "Susceptible to stress",
                            "category": "personality",
                            "percentile": 0.0160441957979911
                        }
                    ]
                }
            ],
            "needs": [
                {
                    "trait_id": "need_challenge",
                    "name": "Challenge",
                    "category": "needs",
                    "percentile": 0.14627232411178653
                },
                {
                    "trait_id": "need_closeness",
                    "name": "Closeness",
                    "category": "needs",
                    "percentile": 0.022737402795758477
                },
                {
                    "trait_id": "need_curiosity",
                    "name": "Curiosity",
                    "category": "needs",
                    "percentile": 0.36746601523535216
                },
                {
                    "trait_id": "need_excitement",
                    "name": "Excitement",
                    "category": "needs",
                    "percentile": 0.04089878567298083
                },
                {
                    "trait_id": "need_harmony",
                    "name": "Harmony",
                    "category": "needs",
                    "percentile": 0.012458800439233197
                },
                {
                    "trait_id": "need_ideal",
                    "name": "Ideal",
                    "category": "needs",
                    "percentile": 0.07155147639020798
                },
                {
                    "trait_id": "need_liberty",
                    "name": "Liberty",
                    "category": "needs",
                    "percentile": 0.029329578184053795
                },
                {
                    "trait_id": "need_love",
                    "name": "Love",
                    "category": "needs",
                    "percentile": 0.010630360533172567
                },
                {
                    "trait_id": "need_practicality",
                    "name": "Practicality",
                    "category": "needs",
                    "percentile": 0.11263264769173947
                },
                {
                    "trait_id": "need_self_expression",
                    "name": "Self-expression",
                    "category": "needs",
                    "percentile": 0.006771991960655144
                },
                {
                    "trait_id": "need_stability",
                    "name": "Stability",
                    "category": "needs",
                    "percentile": 0.1055047665517071
                },
                {
                    "trait_id": "need_structure",
                    "name": "Structure",
                    "category": "needs",
                    "percentile": 0.7397657021209125
                }
            ],
            "values": [
                {
                    "trait_id": "value_conservation",
                    "name": "Conservation",
                    "category": "values",
                    "percentile": 0.0980710921783175
                },
                {
                    "trait_id": "value_openness_to_change",
                    "name": "Openness to change",
                    "category": "values",
                    "percentile": 0.1420357059296135
                },
                {
                    "trait_id": "value_hedonism",
                    "name": "Hedonism",
                    "category": "values",
                    "percentile": 0.011761019463446987
                },
                {
                    "trait_id": "value_self_enhancement",
                    "name": "Self-enhancement",
                    "category": "values",
                    "percentile": 0.004194989604622834
                },
                {
                    "trait_id": "value_self_transcendence",
                    "name": "Self-transcendence",
                    "category": "values",
                    "percentile": 0.05870952071041813
                }
            ],
            "consumption_preferences": [
                {
                    "consumption_preference_category_id": "consumption_preferences_shopping",
                    "name": "Purchasing Preferences",
                    "consumption_preferences": [
                        {
                            "consumption_preference_id": "consumption_preferences_automobile_ownership_cost",
                            "name": "Likely to be sensitive to ownership cost when buying automobiles",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_automobile_safety",
                            "name": "Likely to prefer safety when buying automobiles",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_clothes_quality",
                            "name": "Likely to prefer quality when buying clothes",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_clothes_style",
                            "name": "Likely to prefer style when buying clothes",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_clothes_comfort",
                            "name": "Likely to prefer comfort when buying clothes",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_influence_brand_name",
                            "name": "Likely to be influenced by brand name when making product purchases",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_influence_utility",
                            "name": "Likely to be influenced by product utility when making product purchases",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_influence_online_ads",
                            "name": "Likely to be influenced by online ads when making product purchases",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_influence_social_media",
                            "name": "Likely to be influenced by social media when making product purchases",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_influence_family_members",
                            "name": "Likely to be influenced by family when making product purchases",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_spur_of_moment",
                            "name": "Likely to indulge in spur of the moment purchases",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_credit_card_payment",
                            "name": "Likely to prefer using credit cards for shopping",
                            "score": 1
                        }
                    ]
                },
                {
                    "consumption_preference_category_id": "consumption_preferences_health_and_activity",
                    "name": "Health & Activity Preferences",
                    "consumption_preferences": [
                        {
                            "consumption_preference_id": "consumption_preferences_eat_out",
                            "name": "Likely to eat out frequently",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_gym_membership",
                            "name": "Likely to have a gym membership",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_outdoor",
                            "name": "Likely to like outdoor activities",
                            "score": 1
                        }
                    ]
                },
                {
                    "consumption_preference_category_id": "consumption_preferences_environmental_concern",
                    "name": "Environmental Concern Preferences",
                    "consumption_preferences": [
                        {
                            "consumption_preference_id": "consumption_preferences_concerned_environment",
                            "name": "Likely to be concerned about the environment",
                            "score": 1
                        }
                    ]
                },
                {
                    "consumption_preference_category_id": "consumption_preferences_entrepreneurship",
                    "name": "Entrepreneurship Preferences",
                    "consumption_preferences": [
                        {
                            "consumption_preference_id": "consumption_preferences_start_business",
                            "name": "Likely to consider starting a business in next few years",
                            "score": 0.5
                        }
                    ]
                },
                {
                    "consumption_preference_category_id": "consumption_preferences_movie",
                    "name": "Movie Preferences",
                    "consumption_preferences": [
                        {
                            "consumption_preference_id": "consumption_preferences_movie_romance",
                            "name": "Likely to like romance movies",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_movie_adventure",
                            "name": "Likely to like adventure movies",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_movie_horror",
                            "name": "Likely to like horror movies",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_movie_musical",
                            "name": "Likely to like musical movies",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_movie_historical",
                            "name": "Likely to like historical movies",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_movie_science_fiction",
                            "name": "Likely to like science-fiction movies",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_movie_war",
                            "name": "Likely to like war movies",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_movie_drama",
                            "name": "Likely to like drama movies",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_movie_action",
                            "name": "Likely to like action movies",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_movie_documentary",
                            "name": "Likely to like documentary movies",
                            "score": 1
                        }
                    ]
                },
                {
                    "consumption_preference_category_id": "consumption_preferences_music",
                    "name": "Music Preferences",
                    "consumption_preferences": [
                        {
                            "consumption_preference_id": "consumption_preferences_music_rap",
                            "name": "Likely to like rap music",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_music_country",
                            "name": "Likely to like country music",
                            "score": 0.5
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_music_r_b",
                            "name": "Likely to like R&B music",
                            "score": 0.5
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_music_hip_hop",
                            "name": "Likely to like hip hop music",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_music_live_event",
                            "name": "Likely to attend live musical events",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_music_playing",
                            "name": "Likely to have experience playing music",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_music_latin",
                            "name": "Likely to like Latin music",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_music_rock",
                            "name": "Likely to like rock music",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_music_classical",
                            "name": "Likely to like classical music",
                            "score": 1
                        }
                    ]
                },
                {
                    "consumption_preference_category_id": "consumption_preferences_reading",
                    "name": "Reading Preferences",
                    "consumption_preferences": [
                        {
                            "consumption_preference_id": "consumption_preferences_read_frequency",
                            "name": "Likely to read often",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_books_entertainment_magazines",
                            "name": "Likely to read entertainment magazines",
                            "score": 0
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_books_non_fiction",
                            "name": "Likely to read non-fiction books",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_books_financial_investing",
                            "name": "Likely to read financial investment books",
                            "score": 1
                        },
                        {
                            "consumption_preference_id": "consumption_preferences_books_autobiographies",
                            "name": "Likely to read autobiographical books",
                            "score": 1
                        }
                    ]
                },
                {
                    "consumption_preference_category_id": "consumption_preferences_volunteering",
                    "name": "Volunteering Preferences",
                    "consumption_preferences": [
                        {
                            "consumption_preference_id": "consumption_preferences_volunteer",
                            "name": "Likely to volunteer for social causes",
                            "score": 1
                        }
                    ]
                }
            ],
            "warnings": []
        }

    }
    response = jsonify(response_object)
    response.status_code = 200
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/predict', methods=['POST'])
def predict():
    """
    Does predict
    """
    json_object = request.get_json()

    response_object = predict_api('a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a '
                                  'a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a '
                                  'a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a '
                                    'a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a '
                                  'a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a '
                                    'a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a '
                                  'a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a '
                                  'a a a a a a a a a a a a a a a a a' + json_object['text'])

    response = jsonify(response_object)
    response.status_code = 200
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


def credentials():
    location = "/su.json"
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = location


def logistic_model(x, y=1, train=False):
    # if train:
    #     score_prev = -1
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #     lambda_set = np.linspace(0.5, 10, 15)
    #     for lambo in lambda_set:
    #         lr = LogisticRegression(max_iter=100, tol=0.01, C=(1 / lambo))
    #         lr.fit(x_train, y_train)
    #         score = lr.score(x_test, y_test)
    #         if score > score_prev:
    #             score_prev = score
    #             pickle.dump(lr, open('Logistic_model_weights.sav', 'wb'))
    # else:
    #     global lr
    #     return lr.predict_proba(x)[1]
    return lr.predict_proba(x)[0][1]


def knn_model(x, y=1, train=False):
    # if train:
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #     score_prev = -1
    #     nearest_count = [i for i in range(1, 10)]
    #     for nearest in nearest_count:
    #         knn = KNeighborsClassifier(n_neighbors=nearest)
    #         knn.fit(x_train, y_train)
    #         score = knn.score(x_test, y_test)
    #         if score > score_prev:
    #             score_prev = score
    #             pickle.dump(knn, open('KNN_model_n_value.sav', 'wb'))
    # else:
    #     knn = pickle.load(open('KNN_model_n_value.sav', 'rb'))
    #     return knn.predict_proba(x)[1]
    return knn.predict_proba(x)[0][1]


def data_normalize(x):
    mean = np.sum(x, axis=0)
    std = np.std(x, axis=0)
    return ((x - mean) / (std)), mean, std


def featuring(text):
    # first word frequency research work
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = wordpunct_tokenize(text)
    words = [ps.stem(w.lower()) for w in words if (len(w) > 1 and (w not in stop_words) and (not w.isnumeric()))]
    words_freq = Counter(words)

    words_freq_final = [
        {'label': key, 'count': words_freq[key]} if key in words_freq.keys() else {'label': key, 'count': 0} for key in
        tot_words]
    # now Personality insight work IBM
    personality_insights = PersonalityInsightsV3(version='2017-10-16',
                                                 username='b8a711c2-1a50-4583-83bb-90a829987200',
                                                 password='qhp2EfOBC6Zd',
                                                 url='https://gateway.watsonplatform.net/personality-insights/api'
                                                 )

    profile = personality_insights.profile(content=text,
                                           content_type='text/plain',
                                           raw_scores=True,
                                           consumption_preferences=True
                                           )
    # now sentiment analysis Google
    client = language.LanguageServiceClient()
    document = language.types.Document(content=text, type='PLAIN_TEXT')
    response = client.analyze_sentiment(document=document, encoding_type='UTF32')
    document_sentiment = {'positive': 0, 'neutral': response.document_sentiment.magnitude, 'negative': 0}
    if response.document_sentiment.score > 0:
        document_sentiment['positive'] = response.document_sentiment.score
    else:
        document_sentiment['negative'] = response.document_sentiment.score
    response_object = {"sentiment": document_sentiment,
                       "word_freq": words_freq_final,
                       "ibm": profile
                       }
    df = make_df(response_object)
    return df, response_object


def predict_api(text):
    df, response_object = featuring(text)
    df = (df - mean) / std
    lr_result = logistic_model(df)
    knn_result = knn_model(df)
    response_object["response"] = "predict"
    response_object["predict"] = [{'label': 'KNN', 'value': knn_result},
                                  {'label': 'Logistic Regression', 'value': lr_result}
                                  ]
    return response_object


def make_df(response_object):
    features = {}
    # adding word frequency features
    for in_word in response_object['word_freq']:
        features[in_word['label']] = in_word['count']
    # now adding google sentiment analysis features
    features['sentiment_poitive'] = response_object['sentiment']['positive']
    features['sentiment_negative'] = response_object['sentiment']['negative']
    features['sentiment_neutral'] = response_object['sentiment']['neutral']
    # now adding IBM Watson personality insight analysis features
    for i in ['needs', 'values']:
        for in_needs in response_object['ibm'][i]:
            features[in_needs['name']] = in_needs['percentile']
    for in_personality in response_object['ibm']['personality']:
        for in_children in in_personality['children']:
            features[in_children['name']] = in_children['percentile']
    df = pd.DataFrame([features])
    return df


def train_api(text_list, result_list):
    text_list = [i + i for i in text_list]
    for i in range(0, len(text_list)):
        if i == 0:
            df, temp = featuring(text_list[i])
        else:
            df_temp, temp = featuring(text_list[i])
            df = df.append(df_temp)
    df, mean, std = data_normalize(df)
    np.save('Mean_normalization_parameters.npy', mean)
    np.save('Std_normalization_parameters.npy', std)
    logistic_model(df, result_list, True)
    knn_model(df, result_list, True)

if __name__ == '__main__':
    app.run()