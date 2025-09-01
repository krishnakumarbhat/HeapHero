# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        epsilon = self.epsilon
        alpha = self.alpha
        discount = self.discount
        #print(util.flipCoin(self.epsilon))
        #print("exploration prob:", epsilon)
        #print("learning rate: ", alpha)
        #print("discount rate: ", discount)
        
        #table of action values indexed by state and action, initially 0
        self.stateQvals = {}


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR HERE ***"
        #print("state, action", state,action)
        #print("legal actions: ", self.getLegalActions(state))
        if (state,action) not in self.stateQvals:
          return 0.0
        else:
          return self.stateQvals[(state,action)]
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if len(self.getLegalActions(state)) == 0:
          return 0.0
        else:
          #compute the best action to take in a state
          #state with the highest q value, take that action
          #print("computeValueFromQvalues")
          #print("legal actions:" , self.getLegalActions(state))
          maxQ = float('-inf')
          for action in self.getLegalActions(state):
            qval = self.getQValue(state, action)
            if qval > maxQ:
              maxQ = qval
          #print("state, maxQ: ", state,maxQ)
          return maxQ
          

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #print("state: ", state)
        #print("legal actions: ", self.getLegalActions(state))
        actions = util.Counter()
        if len(self.getLegalActions(state)) == 0:
          return None
        else:
          for action in self.getLegalActions(state):
            actionVal = self.getQValue(state, action)
            actions[action] = actionVal
          #print(actions)
          #print("best action: ", actions.argMax())
          return actions.argMax()
        util.raiseNotDefined()


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        if not legalActions:
          action = None
        "*** YOUR CODE HERE ***"
        #Randomly take an action epsilon amount of the time
        goRand = util.flipCoin(self.epsilon)
        if goRand:
          return random.choice(self.getLegalActions(state))
        else:
          return self.getPolicy(state)
        util.raiseNotDefined()
        

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        """def observeTransition(self, state,action,nextState,deltaReward):
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same argument
            NOTE: Do *not* override or call this function
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)
        
        Learn Q(s,a) values as you go
        Receive a sample (s,a,s’,r)
        § Consider your old estimate: Q(s,a)
        § Consider your new sample estimate: sample = R(s,a,s') + discount * (max_action(Q(s',a')))
        § Incorporate the new estimate into a running average: Q(s,a) = (1 - alpha)*Q(s,a) + alpha[sample]
        
        """

        #do q value update
        sampleEst = reward + self.discount * self.computeValueFromQValues(nextState)
        oldEstimate = self.getQValue(state, action)
        self.stateQvals[(state,action)] = (1-self.alpha)*oldEstimate + self.alpha*sampleEst
        #print("qvals: ", self.stateQvals)
        #print("update to :", self.stateQvals[(state,action)])
        #print("new qvals: ", self.stateQvals)
        return self.stateQvals[(state,action)]
         

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        
        "*** YOUR CODE HERE ***"
        qval = 0
        features = self.featExtractor.getFeatures(state, action)
        for i in features:
            qval += features[i] * self.weights[i]
            print("QVAL: ", qval)
        return qval
    
        
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state,action)
        difference = (reward + self.discount*self.getValue(nextState)) - self.getQValue(state,action)
        weights = self.weights
        for i in features:
            weights[i] += self.alpha*difference*features[i]
            print("WEIGHTS: ", weights)
        return weights
        
        
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
