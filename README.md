# Flappy_bird_RL_coins

Run python dqn.py to train the model. You may need to rm dqn_model.pth or dqn_model_simple.pth if you want to train the model from scratch.

Run python dqn.py to train the model without display (much faster). 
- during training, CTRL + C can terminate the training process and save the model
- this is helpful when you want to continue to train the model next time

Run python dqn.py --display to train the model with display.

Run python visualize.py YOUR_DATA_FILE_NAME to get the data plot
- example visualize.py training_metrics_simple

Now dqn.py and dqn_simple.py has the data storing featues as the previous analysis_dqn.py

json for data visualization
