# Bond Arbitrage Project
## **Overview**
This repository contains all the code and documentation for the bond arbitrage project. The project is organized into several directories as described below:


## **Description of Key Directories**  

```bash
.
├── data/
│   ├── raw/                  
│   ├── processed/            
│   └── pipeline.py           
│
├── research/
│   ├── notebooks/
│   │   └── exploratory_analysis.ipynb
│   │   └── backtest_strategy.ipynb
│   └── papers/
│   └── reports/              
│
├── src/
│   ├── strategies/
│   │   └── bond_arb_strategy.py     
│   └── utils/
│       ├── data_loader.py           
│       └── indicators.py            
│
├── risk/
│   ├── risk_metrics.py       
│   └── risk_report.ipynb    
│
├── tests/
│   ├── test_strategy.py      
│   └── test_pipeline.py      
│
├── infrastructure/
│   ├── Dockerfile            
│   ├── docker-compose.yml   
│   └── deploy.sh        
│ 
├── .gitignore               
├── README.md               
├── requirements.txt      
└── .env                    
```
## **Work Flow/Transition in focus of work**
```bash
.
Refining and creating bond spread backtester 
-> Refining strategy
-> Applying ML/RL to improve strategy
-> Final Result/Strategy
```