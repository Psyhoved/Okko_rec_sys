import pickle
from pathlib import Path

import pandas as pd

from libs.recommend_user_item import convert_ratings_to_trainset, train_svd_model

params = {
    "n_epochs": 50,
    "lr_all": 0.013,
    "reg_all": 0.89,
    "n_factors": 17}
n_item = 10
n_user = 10
top_n = 5


def load_data():
    df = pd.read_csv('Data/download/ratings.csv').drop(columns='ts')
    user_counts = df['user_uid'].value_counts()
    movie_counts = df['element_uid'].value_counts()

    # Фильтруем пользователей, которые оценили меньше 2 фильмов
    users_to_keep = user_counts[user_counts > n_user].index
    df_filtered_users = df[df['user_uid'].isin(users_to_keep)]
    del users_to_keep, user_counts
    # Фильтруем фильмы, которые оценили меньше 10 раз
    movies_to_keep = movie_counts[movie_counts > n_item].index
    df_filtered = df_filtered_users[df_filtered_users['element_uid'].isin(movies_to_keep)]
    del movie_counts, df_filtered_users, movies_to_keep
    # Список самых высокооценённых фильмов для пользователей, у которых не хватает истории просмотров
    # сумма, а не средняя, чтобы учесть число просмотров. Число оценок косвенно говорит о качестве фильма.
    top_list = \
        df.groupby('element_uid', as_index=False)['rating'].sum().sort_values(by='rating', ascending=False).reset_index(
            drop=True)['element_uid'][:top_n].to_list()
    data = convert_ratings_to_trainset(df_filtered)
    del df_filtered, df
    rec = train_svd_model(data_prep_df=data,
                          best_params=params,
                          n_pred=5,
                          )
    del data
    rec.to_pickle(Path('cache', 'okko_rec.pickle'))
    with open(Path('cache', 'top_list.pickle'), 'wb') as f:
        pickle.dump(top_list, f)
