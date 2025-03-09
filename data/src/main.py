from fastapi.middleware.cors import CORSMiddleware
from logger import logger as log
from fastapi import FastAPI
from database import Database
from routers import pricing
import uvicorn

app = FastAPI(
    title="Pricing API",
    description="API for pricing data",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(pricing.router)    



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)