# CSCE580 Project 3 - Reinforcement

# Authors: 
## Tristan Klintworth - T49161658
## Melody Fakih - K54301349

# Project 3 (Reinforcement)
In this project, we implemented value iteration and Q-Learning to simulate a crawler robot and a pacman agent. 

# Implementation
### Q1: Value Iteration
Overall, we implemented the value iteration state update equation in which we separately computed the Q-value and the best action in the given state according to the values currently stored in self.values.
This code is the actual update of a state:
```python
values[state] = self.computeQValueFromValues(state,action)
```
And calculating the Q-value:
```python
qVal += probability*(reward+discount*nextValue)
```
### Q2: Bridge Crossing Analysis
In the section, we had to adjust the discount or the noise to allow the agent to successfully cross the bridge.
The default noise was 0.2 but we used 0.0 noise so our agent wouldn't fall off of the bridge if it followed the optimal policy. We also kept the discount at 0.9
```python
answerDiscount = 0.9
answerNoise = 0.0
```
### Q3: Policies
#### 3a. 
To prefer the close exit and risk the cliff, we used discount 0.5 and living reward -1.0. We used a negative living reward to encourage the agent to take more risk and we used 0.5 as the discount to encourage the agent not to travel too far. 

#### 3b. 
For the agent to prefer the close exit but avoid the cliff, we had to set the noise to 0.2 so the agent would sometimes take the north action, which it avoids under almost any other circumstance. Then we set the other values to the same as 3a for the same reasons.

#### 3c.
To prefer the distant exit but risk the cliff, we set the discount at 1.0 so the agent would not be discouraged from going far and we made the living reward very negative so the agent would not want to go to the low value exit and still risk the cliff.

#### 3d.
To prefer the distant exit and avoid the cliff, we set the discount relatively high so that the agent would be encouraged to travel further. To avoid the cliff, we added noise and a negative living reward.

#### 3e.
To avoid both exits and the cliff we set the discount to 1 so the reward never decreases and the reward is high so the agent never desires to leave through a terminal state and will always take the north option to stay inside the gridworld.

### Q4: Asynchronous Value Iteration
For this version of value iteration, we iterated through each update while cycling through the states using the length of the list of states.
Each iteration updates the value of only one state:
```python
self.values[states[count]] = self.computeQValueFromValues(states[count],action)
```
### Q5: Prioritized Sweeping Value Iteration
For prioritzed sweeping value iteration we followed the given procedure.
First, we computed the predecessors of each state as long as there was a nonzero probability of that state reaching the current state:
```python
preds = {}
        for s in self.mdp.getStates():
            preds[s] = set()
...

if probability > 0:
        preds[nextState].add(s)
```
Then using a priority queue, we pushed nonterminal states and -diff, which was the absolute value of the current value of the predecessor and the maximum Q across all possible actions from the predecessors: 
```python
diff = abs(self.values[s] - maxQ)
pq.update(s, -diff)
```
As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
```python
if diff > self.theta:
    pq.update(p, -diff)
```               
### Q6: Q-Learning
The first thing we did was initialize a dictionary to index state and actions with q-values.
```python
self.stateQvals = {}
```
Next, we implemented the functions of getQvalue, computeValuesFromQvalues, and computeActionsFromQvalues.
And then in the update function, we took the sample estimate which is reward + discount * qvalue of the next state
```python
sampleEst = reward + self.discount * self.computeValueFromQValues(nextState)
```
Then, we took old estimate which is the qvalue taking the state and action
```python
oldEstimate = self.getQValue(state, action)
```
Finally, we incorporated the new estimate in the running average
```python
 self.stateQvals[(state,action)] = (1-self.alpha)*oldEstimate + self.alpha*sampleEst
```
### Q7: Epsilon Greedy
Implented the getAction function which takes a random action and then takes the best policy otherwise.
```python
goRand = util.flipCoin(self.epsilon)
if goRand:
   return random.choice(self.getLegalActions(state))
else:
   return self.getPolicy(state)
### Q8: Bridge Crossing Revisited
For this part, we tried different values of epsilon and learning rate and then determined there was no way to adjust these values to meet the requirements of the assignment. One of the reasons was that the number of iterations was definitely too low to learn what it needed to learn as accurate as it needed to be.
### Q9: Q-Learning and Pacman
Due to our Qlearning works without exception and our agent wins 80% of the time, we did not have to change our code. 
### Q10: Approximate Q-Learning
In this section, we had to implement two functions, the getQvalue and the update. In the getQvalue we get the features(which is a vector of state and action pairs) by this function:
```python
features = self.featExtractor.getFeatures(state, action)
```
The way we get the qvalue is to iterate through each feature and multiply each feature by the weight and then sum all the values.
```python
qval += features[i] * self.weights[i]
```
In the update function we get the difference by reward + the discount * the value at the next state and all that minus the qvalue regarding the state and action.
```python
difference = (reward + self.discount*self.getValue(nextState)) - self.getQValue(state,action)
```
Next, we iterate through each feature and multiply alpha by discount and by the features at each iteration and then sum it all up to get the weight.
```python
weights[i] += self.alpha*difference*features[i]
```
