# Overview and Setup

This section of the repository contains all logic and code for the data scraping, cleaning, storage, and retrieval of the data used in the project. The data is stored in a PostgreSQL database modded with timescaleDB for time series data. The data is sourced using the Yahoo Finance API with the yfinance library.


## Run the API

### **Production**

1. Build the Docker Image
```zsh
docker build -t bond-arbitrage-api .
```
1. Run the [Infrastructure](../infrastructure/README.MD) and then run the API
```zsh
docker run -d -p 8000:8000 --network bond-arbitrage bond-arbitrage-api
```

### **Development**

1. Install the dependencies
```zsh
pip install -r requirements.txt
```

2. Run the API
```zsh
uvicorn main:app --host 0.0.0.0 --port 8000
```

> View <a href="http://localhost:8000/docs">/docs</a> for the API documentation

## Launching the data infrastructure for development

```zsh
docker compose -f compose-dev.yml up -d
```


## Ports
| Service | Port |
|---------|------|
| API | 8000 |
| Database | 5432 |

