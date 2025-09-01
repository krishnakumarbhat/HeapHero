# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
          """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

            #me defining mdp variables
        self.state = mdp.getStartState()
        self.possActions = mdp.getPossibleActions(self.state)
        #self.reward = mdp.getReward() #Get the reward for the state, action, nextState transition

        #reachable states and their probabilities for the start state
        self.reachableStatesAndProbs = []
        
        for action in self.possActions:
            self.reachableStatesAndProbs += mdp.getTransitionStatesAndProbs(self.state, action)
        

        #run value iteration using what we have
        self.runValueIteration()
        
        
        
    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for k in range(self.iterations):
            values = util.Counter()
            for state in states:
                if not self.mdp.isTerminal(state):
                    action = self.computeActionFromValues(state)
                    values[state] = self.computeQValueFromValues(state,action)
            self.values = values
    
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
              
            getTransitionStatesAndProbs:
            Returns list of (nextState, prob) pairs
            representing the states reachable
            from 'state' by taking 'action' along
            with their transition probabilities.
        
        "*** YOUR CODE HERE ***"

              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        qVal = 0
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state,action):
            reward = self.mdp.getReward(state, action, nextState)
            discount = self.discount
            nextValue = self.values[nextState]
            #print("nextValue: ", nextValue, self.values)
            qVal += probability*(reward+discount*nextValue)
        
        return qVal
    
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        possibleActions = self.mdp.getPossibleActions(state)
        actions = util.Counter()
       
        if self.mdp.isTerminal(state) == True:
            return None
         
        for action in possibleActions:
            qVal = self.computeQValueFromValues(state,action)
            actions[action] = qVal
       
        return actions.argMax()
                
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        

    def runValueIteration(self):
        states = self.mdp.getStates()
        statesLength = len(self.mdp.getStates())
        count = 0
        #values = util.Counter()
        for k in range(self.iterations):
            if count == statesLength:
                count = 0

            if self.mdp.isTerminal(states[count]):
                self.values[count] = self.values[count]
                #print("terminal")
            else:
                action = self.computeActionFromValues(states[count])
                self.values[states[count]] = self.computeQValueFromValues(states[count],action)
                #print("11111111111111111111111111111111111111111111", self.values)
                #count += 1
            count += 1
            #print("k, count", k, count)
            
            #self.values = values
            
    


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #Compute predecessors of all states.
        #the predecessors of a state s are all states that have a nonzero probability of reaching s by taking some action a
        #When you compute predecessors of a state, make sure to store them in a set
        preds = {}
        for s in self.mdp.getStates():
            preds[s] = set()
            

        states = self.mdp.getStates()
        for s in states:
            actions = self.mdp.getPossibleActions(s)
            for action in actions:
                for nextState, probability in self.mdp.getTransitionStatesAndProbs(s, action):
                    if not self.mdp.isTerminal(nextState):
                        if probability > 0:
                            #preds[s].add(nextState)
                            preds[nextState].add(s)
        


        pq = util.PriorityQueue()


        #highest Q-value across all possible actions from s
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                maxQ = float('-inf')
                for action in actions:
                    qVal = self.computeQValueFromValues(s, action)
                    if qVal > maxQ:
                        maxQ = qVal
                #Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across 
                #all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
                diff = abs(self.values[s] - maxQ)
                #Push s into the priority queue with priority -diff (note that this is negative). 
                #We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                #pq.push(s, -diff)
                pq.update(s, -diff)

        for iteration in range(self.iterations):
            #print("iteration ", k)
            if pq.isEmpty():
                break

            s = pq.pop()

            #Update s's value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                action = self.computeActionFromValues(s)
                self.values[s] = self.computeQValueFromValues(s, action)
            #For each predecessor p of s
            for p in preds[s]:
                #Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across 
                #all possible actions from p (this represents what the value should be); call this number diff. 
                #Do NOT update self.values[p] in this step.
                actions = self.mdp.getPossibleActions(p)
                maxQP = float('-inf')
                for action in actions:
                    qValP = self.computeQValueFromValues(p, action)
                    if qValP > maxQP:
                        maxQP = qValP
                diff = abs(self.values[p] - maxQP)
                #If diff > theta, push p into the priority queue with priority -diff (note that this is negative), 
                #as long as it does not already exist in the priority queue with equal or lower priority. 
                #As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                if diff > self.theta:
                    #pq.push(p, -diff)
                    pq.update(p, -diff)
                


        

        


