"""
TODO: ADDRESS INSTABILITY IN LEARNING

The agents seem to have very unstable learning in this environment. The first thing I did to address this was make the
training data more monte carlo. This was done by keeping all experiences in a deque until end of episode then amending
the experience tuples with final reward. Doing this allows for positive learning on experiences that led to the final
reward rather than treating all of those experiences as negative.

The agent behavior suggests that there is a real difference between good experiences and bad experiences. agent behavior
will plateau then either tank or rise sharply. being able to watch the agent move through those episodes may give better
insight into what is causing this instability.

Other things that may improve the stability of learning:

1. genetic algorithm for choosing the best agent then copying weights to all other agents

2. incorporate prioritized buffer replay ****dougs best choice****

3. rather than learning from subsets of the training data, learn from the entire set then discard

4. incorporate multiple timestamps in to the input state to give the agent temporal context,
***doug: imcorperate timederivate of state to give temporal element

TODO: EXAMINE POSSIBLE ADVERSARIAL BATCHES OF DATA

Occasionally, an agent will find the optimum solution but then its performance will tank. this suggests that there
may be high dimensional pockets that the agent is being trapped in similar to what the alphaGo team described as
"delusional" states, where the agent thought it was doing really well but was actually doing poorly. We could
interrogate this by saving the batches of experiences then filtering for the batches that caused the agent to loose
performance. We could then take a stable agent that has found the optimal solution and train them on the "delusional"
data to see how this affects performance.

1. build sequence based storage for experiences and agent scores

2. build sequential checkpoints of agent weights

3. test both learning and experiencing the adversarial data

TODO: MODEL ENGINEERING

General improvements to the model to enhance interrogation of the system.

1. adjust for the saving of multiple agent weights

2. write tests, obey the testing goat!

3. fix the water/growth function so that the amount of water taken out is not in the asymptotic range of the function

"""