# Deep Q Learning
## Project Description
Old Python 2 project to recreate the Q Learning Algorithm with Keras and OpenAI Gym. With the program, one can train a deep learning model to play Atari games. Uses deprecated libraries. Needs updating. 

## File Structure
- deepqnetwork.py builds the Q Learning model with the DeepQNetwork class and supports training and inference via its methods
- agent.py hosts the agent class instances of which can play games on OpenAI Gym using an instance of the DeepQNetwork class to choose its action
- stats.py hosts the Stats class instances of which record game facts and write them to a CSV
- utils.py hosts miscellaneous classes and functions needed by other files
- replaymemory.py hosts a ReplayMemory class which is used by the Agent to train the DeepQNetwork instance
