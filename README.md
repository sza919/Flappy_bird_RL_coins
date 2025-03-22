# Flappy_bird_RL_coins

Run python dqn.py to train the model. You may need to rm dqn_model.pth if you want to train the model from scratch. (other dqn_basic, dqn_best_tj, dqn_simple will be similar)

Run python dqn.py to train the model without display (much faster). 
- during training, CTRL + C can terminate the training process and save the model
- this is helpful when you want to continue to train the model next time

Run python dqn.py --display to train the model with display.

Run python dqn.py --episodes NUMBER to train the model for NUMBER episodes

Run python visualize.py YOUR_DATA_FILE_NAME to get the data plot
- example visualize.py training_metrics_simple

Run python play_dqn.py --name YOUR_DATA_FILE_NAME to play the game for 300 episodes

When try to load model, due to the python environment you may need to put or remove weights_only = True from torch.load()
