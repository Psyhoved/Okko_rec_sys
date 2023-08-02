from fastapi import FastAPI, Response
from fastapi.openapi.utils import get_openapi

from libs.http_actions.model_cache import get_top_similar_by_user
from libs.http_actions.responses import RecommendationResponse, RecommendationItem
from create_user_item_cache import reload

top_list = [747, 2714, 5616, 3336, 2245]
app = FastAPI()


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Сервис рекомендаций Okko",
        version="0.0.1",
        description="",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

@app.get('/')
async def hello():
    """
    ping - pong
    :return:
    :rtype:
    """
    return {'ok': True}


@app.get('/get_rec', response_model=RecommendationResponse)
async def get_rec(guest_id: int) -> RecommendationResponse:
    response = RecommendationResponse()

    if guest_id > 0:
        found = get_top_similar_by_user(guest_id)
    else:
        found = None

    if found is None or len(found) == 0:
        response.items = [RecommendationItem(id=item_id, score=len(top_list) - i) for i, item_id in enumerate(top_list)]
    else:
        response.items = [RecommendationItem(id=item_id, score=len(found) - i) for i, item_id in enumerate(found)]

    return response

@app.get('/reload_cache')
async def reload_cache(response: Response):
    response.status_code = 200
    if not reload():
        response.status_code = 500

    return {}
