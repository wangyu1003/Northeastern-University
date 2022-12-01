1、First, please install all the required packages.
```ruby
pip install -r requirements.txt
```
2、 Now we are ready to train MPDRL agent and DDPG agent. To do this, we must execute the following command.
```ruby
python MPDRL.py
python DDPG.py
```
3、Then, we can evaluate our trained model on different topologies executing the command below. 
```ruby
python Evaluate.py -d ./Logs/Ebone/expsample_MPDRLAgentLogs.txt ./Logs/Ebone/expsample_DDPGAgentLogs.txt
```
4、Finally, we can perform a robustness comparison experiment executing the command below. 
```ruby
python Robustness.py -d ./Logs/Ebone/expsample_MPDRLAgentLogs.txt ./Logs/Ebone/expsample_DDPGAgentLogs.txt
```

If you have any question, please please send email to <wangyu981003@163.com>
