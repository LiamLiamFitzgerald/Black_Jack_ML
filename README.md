# Black_Jack_ML
A model designed to play black jack on a randomised deck. The deck is random in order to prevent simple card counting, forcing the model to learn optimal policies. Download the folder and run the python file within it to view a tk inter ui to mess around
with different values

----------------------------------------------------------
This section serves as series of additional notes if you're interested in some of the finer details of this project, as my original project report went over the word count.

Interesting Things of Note:
1. In the outputs of each nn, I add some degree of additional random variance. One would assume that combining these two random policies would harm the models overall output, but it actually improved it a small percentage. This is called the Parronda Paradox, and is a known effect used in thing such as brownian motion. Discovering this effect has gotten me interested in implementing a purely probabilistic model, that is trained to select different layered random sample windows, on a given deck state to produce a strategy. This paradox is also seen on the betting module, as each bet is multiplied by  factor chosen from an array spanning from 0-30, strangely this also improved profit. This suggests that is an optimal middle ground between random choice alternating with calculated strategy that would produce peak profit

What I could Improve on:

1. My greatest regret was not implementing the gated network architechture in all modules, as the betting module that utilised it proved to be the most efficient. But it's a blessing in disguise as in not using it in the other modueles, i gained an appreciation of its effectiveness. Interestingly the gating on its two submodules was ~0.9 meaning that only in 10% of instances was raw card count used.
2. 
3. I would also make the heuristic module a neural network, rather that purely mathimatical functions. As then it could be equipped to find deeper trends. As an experiment I ran the program using regular, unrandomised decks of cards, and the models lost more games, this indicates that they were in fact developing a novel strategy, unique to the randomised decks, as such a dynamic heuristic nn could be better equipped ot explore these strategies.
4. I would also create a training an enviorment that could recover a model if its loss ever randomly explodes during training, as the majority of each models training consisted of me squinting at the losses and output weights,
5. I think if I were to improve the play module, I would make it a gated network of three subnetworks, each with an independently trained strategy for decks of high, low and normal deviations. I would then train it for hours at a very low learning rate.
6. I would also tidy up my variables more, given the tight timeframe I completed this iteration in I ended up resolving some bugs with excessive and somehwat unelegant solutions, such as all the card_ocrs handling syntax. 

What I learned:

1. I learned that pop() function has o(n) time complexity, looking back on it now its very obvious. In replacing the popping of a temperary list with in place traversal of a deck with the idx method I improved the optimal paths algorithm's time complexity greatly from its previous o(n)**2 to linear time.
2. I learned alot about machine learning, I begun this project by purely reading theory over the course of two months, and have since become deeply interested in this area.
3. I also learned alot about the numpy module, becoming quite a fan of its array handling

Overall I was mildly content with this project, in its current state. From a financial stand point this game is not very impressive, as if you were given the choice between losing your money either 70% of the time of 58%, you would probably just opt to not play the game. But from a facing adversity and fighting the cruel nature of gambling perspective; this model is quite impressive, as it succeeding in its purpose of improving my blackjack luck. I am still quite proud of the BFS algorithm, as I fulled challeged my optimisation skills and algorithmic thinking. I am also quite proud of the gated approach I thought of, as I think it has great potential to enhance this projects success in the future.



