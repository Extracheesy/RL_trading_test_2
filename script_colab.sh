
##pip list --format=freeze > req.txt

!git clone https://github.com/Extracheesy/RL_trading_test_2.git

%cd ./RL_trading_test_2/

!pip install -r ./req_2.txt

!pip install yfinance

!pip install stockstats==0.3.2

!pip install google-colab

!python ./main.py section_2