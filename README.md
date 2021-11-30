# Semi-Gradient Episodic SARSA

Here, we use the Semi-Gradient Episodic SARSA reinforcement learning algorithm to train an agent to complete OpenAI Gym's implementation of the classic [mountain car control task](https://en.wikipedia.org/wiki/Mountain_car_problem). It is part of my submission to the Wade Scholarship Program. There I explain why, in ligth of the [reward hypothesis](http://incompleteideas.net/rlai.cs.ualberta.ca/RLAI/rewardhypothesis.html), reinforcement learning could potentially lead us to a generalized artificial intelligence.

## Running the tracker
1. Clone the project repository

```
git clone https://github.com/macvincent/Semi-Gradient-Episodic-SARSA.git
```
2. Install required dependencies, preferably from a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment), by running:

```
pip install -r requirements.txt
```
3. To train agent run this command:
```
python3 train.py
```