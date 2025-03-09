import fastapi

router = fastapi.APIRouter(
    prefix="/price",
    tags=["price"],
    responses={404: {"description": "Not found"}, 500: {"description": "Server error"}},
)

@router.get("/{asset_type}/{asset_ticker}/{from_unix_utc}/{to_unix_utc}")
async def get_price(asset_type: str, asset_ticker: str, from_unix_utc: int, to_unix_utc: int):
    return fastapi.Response(status_code=501, content="Not implemented")
