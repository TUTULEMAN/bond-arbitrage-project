CREATE TABLE IF NOT EXISTS asset_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    ticker VARCHAR(10) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    asset_type_id INTEGER NOT NULL,
    FOREIGN KEY (asset_type_id) REFERENCES asset_types(id)
);

CREATE TABLE IF NOT EXISTS prices (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER NOT NULL,
    price DECIMAL NOT NULL,
    time TIMESTAMP NOT NULL,
    FOREIGN KEY (asset_id) REFERENCES assets(id)
);

/* Default data */
INSERT INTO asset_types (name, description) VALUES ('Bond ETF', 'Exchange Traded Fund that invests in bonds');
INSERT INTO asset_types (name, description) VALUES ('Future', 'Derivative contract that obligates the buyer to purchase an asset or the seller to sell an asset at a predetermined future date and price.');
SELECT create_hypertable('prices', by_range('time'));