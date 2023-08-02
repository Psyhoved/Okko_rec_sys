import pickle
from pathlib import Path
import pandas as pd

ROOT_PATH = Path(__file__).parent.parent.parent.absolute()
N_TOP = 5


def __get_frame_from_cache(cache_dir: Path, file_name: str):
    cache_file = cache_dir / file_name
    if not Path(cache_file).is_file():
        return None

    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    if cache is None or len(cache) == 0:
        return None

    return pd.DataFrame(cache)


def __reload_u2i(cache_dir: Path):
    data = __get_frame_from_cache(cache_dir, 'cache\okko_rec.pickle')

    if data is None:
        return False

    return data


def get_top_similar_by_user(user_id: int, top_n: int = N_TOP) -> list[int] | None:
    user2item_df = __reload_u2i(cache_dir=ROOT_PATH)

    if user2item_df is None:
        return None
    top_recs = user2item_df.loc[user2item_df['user_id'] == user_id, 'item_id'].head(top_n).values

    if len(top_recs) == 0:
        return None

    return top_recs
