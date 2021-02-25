
##pip list --format=freeze > req.txt

!git clone https://github.com/Extracheesy/RL_trading_test_2.git

%cd ./RL_trading_test_2/

!pip install -r ./req.txt

!pip install yfinance

!pip install stockstats==0.3.2

!python ./main.py