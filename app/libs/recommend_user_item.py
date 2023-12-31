from collections import defaultdict
import pandas as pd
from typing import Any
from surprise import SVD, Reader, Dataset
from surprise.model_selection import GridSearchCV
from tqdm import tqdm


def __get_train_svd_model(data_prep_df: pd.DataFrame,
                          model_params: dict[str, Any],
                          all_testset: bool = False):
    """

    Args:
        data_prep_df: Данные для обучения SVD модели
        model_params: Параметры для SVD модели
        all_testset: Флаг для получения предсказаний по всем данным

    Returns:

    """
    trainset = data_prep_df.build_full_trainset()

    algo = SVD(**model_params)
    algo.fit(trainset)

    # Подготовка тестового датасета
    testset = trainset.build_anti_testset()

    # если хотим полуить предикт по всем данным
    if all_testset:
        # достаём оценки из трейна и складываем с тестом
        testset_all = trainset.build_testset()
        testset += testset_all
    del trainset
    return testset, algo


def train_svd_model(data_prep_df: pd.DataFrame,
                    best_params: dict[str, Any],
                    n_pred: int = 10,
                    get_all_predicts: bool = False,
                    disable_tqdm: bool = False,
                    all_testset: bool = False,
                    ) -> pd.DataFrame:
    """
        Обучает SVD модель и возвращает DataFrame с рекомендациями

        params:
            data_prep_df: pd.DataFrame - обработанные данные для обучения SVD модели
            best_params: dict[str, Any] - словарь с параметрами для SVD
            n_pred: int - количество рекомендаций, которое нужно возвращать для каждого клиента (default 10)
            get_all_predicts: bool - флаг возвращать ли все рекомендации или top N (default False)
            disable_tqdm: bool - флаг для отключения tqdm (default False)
            all_testset: bool - флаг для получения рекомендаций по ВСЕМ товарам, включая те,
            которые гость уже покупал (default False)
            active_product_list: list[int] - список активных товаров, если не None, то в рекомендациях только эти товары

        return:
            tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[int, int], dict[int, int]] -
            DataFrame с рекомендациями, матрицы рейтингов и словари для расшифровки uid и iid

    """
    testset, algo = __get_train_svd_model(data_prep_df=data_prep_df,
                                          model_params=best_params,
                                          all_testset=all_testset)

    # прогноз для тестового датасета
    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    predictions = algo.test(testset)

    del testset

    # Сохраняем топ-n рекомендаций для каждого юзера в словарь
    top_n_df = get_top_n(predictions, n_pred=n_pred, disable_tqdm=disable_tqdm, all_pred=get_all_predicts)

    del predictions

    # сохраняем топ-n рекомендаций для каждого пользователя
    return rec_to_df(top_n_df)


# Функция для вывода топ-N рекомендаций для каждого пользователя
def get_top_n(predictions: list,
              all_pred: bool = False,
              n_pred: int = 10,
              disable_tqdm: bool = False) -> dict:
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        all_pred(bool): flag to return all predictions
        n_pred(int): The number of recommendation to output for each user. Default
            is 10.
        disable_tqdm(bool): flag to disable tqdm

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, _, est, _ in tqdm(predictions, disable=disable_tqdm):
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in tqdm(top_n.items(), disable=disable_tqdm):
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        if all_pred:
            top_n[uid] = user_ratings
        else:
            top_n[uid] = user_ratings[:n_pred]

    return top_n


def rec_to_df(top_n: dict) -> pd.DataFrame:
    # Сохраняем всё в один датафрейм
    result = []
    for elem in top_n.keys():
        result.extend([i + (elem,) for i in top_n[elem]])

    return pd.DataFrame(result, columns=['item_id', 'rating', 'user_id'])


def gssv(data_prep_df: pd.DataFrame, alg: type, param_grid: dict, measures, opt_by="rmse"):
    """

    :param data_prep_df:
    :param alg:
    :param param_grid:
    :param measures:
    :param opt_by:
    :return:
    """
    gs = GridSearchCV(alg, param_grid, measures=measures, cv=3)
    gs.fit(data_prep_df)

    # best RMSE score
    print(f'best_score {opt_by}: {gs.best_score[opt_by]}')
    # combination of parameters that gave the best RMSE score
    best_params = f'best_params {opt_by}: {gs.best_params[opt_by]}'
    print(best_params)

    return gs, best_params


def convert_ratings_to_trainset(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :param rating:
    :return:
    """

    df.columns = ['userID', 'itemID', 'rating']
    min_rating = df['rating'].min()
    max_rating = df['rating'].max()

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(min_rating, max_rating))
    data_prep_df = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    return data_prep_df
