# Infrastructure For Bond Arbitrage Project

Currently, the infrastructure for the bond arbitrage project is simple and reliant purely on docker-compose. The infrastructure is split into two main parts: the database and the API. The database is a PostgreSQL database modded with timescaleDB for time series data. The API is a FastAPI application that serves the data from the database to the algorithmic trading application.

## Launching the Infrastructure

1. Create a `secrets` directory in the root of the infrastructure directory.
2. Execute `touch ./secrets/db_user.txt` and `touch ./secrets/db_password.txt` and add the desired username and password for the database.
3. Launch the applications using the following commands:
```zsh
docker compose up -d
```

## Stopping the Infrastructure

```zsh
docker compose down
```