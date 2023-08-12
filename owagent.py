from envs.oware import Oware
from agent_interface import AgentInterface
import random

"""
Note: I got a bit carried away with writing this documentation, and it became quite long :P The most important part concerning this assignment
is section 4, which describes on how the agent was tested against the other agents, and the results of said testing. Also, section 1 contains brief
introduction on the working principles of the agent.


##########

OWAGENT DOCUMENTATION

The following documentation extensively describes the developement process of the Oware-palying agent, Owagent. The text is divided into
five sections:

1. Introduction
    Setting the scope for rest of the documentation
2. Features
    Description of the features that this agent uses
3. Choosing the best combination of features
    Explanation on how the final set of features were chosen among the larger set described in section 2
4. Testing and discussion
    Explanation on the test setting against other agents, results of the tests, and discussion on the implications
5. Conclusion
    Concludes the work



##########

1. INTRODUCTION

The goal was to create as good Oware-playing AI as possible for the CS-E4800 Oware tournament. There were multiple possible 
directions to which one could go (Alpha-Beta, Monte Carlo methods, something completely new?), but I decided to go for Alpha-Beta,
as that is something that has intrigued me also previously with chess engines.

In short, this agent, Owagent, has taken the basic essence of the given Iterative deepening Minimax agent; that is, it starts 
looking the best action from depth 1, yields it, then looks through depth 2, yields it, and so on. The final yielded action 
is thus the action that was deemed best on the deepest depth the agent was able to look through in the given time.

That Iterative deepening Minimax alorithm was enhanced for Owagent by
1) Implementing Alpha-Beta pruning, 
2) Adding a better heuristic function,
3) Letting the agent to record and utilize the acquired heuristic values of previous turns, to help Alpha-Beta pruning, and
4) Letting the agent skip certain depths in iterative deepening, the goal being that by doing so, it could possibly
   have time to go through an even deeper level (essentially, gambling the chance to go through current depth with the reward
   being that it can possibly go through even deeper level).
   
In addition, the idea of expanding and searching the game tree even on opponent's turns with multithreading or multiprocessing
was explored. However, that option was discarded, due to 1) it being quite unfair to use the computing resources on opponents turns,
when they need it themselves, and 2) in my opinion, it being quite cheap trick to not evaluate the performance of the actual algorithm
but to artificially get more resources than the opponent. 
(If one of my opponents used that tactic, shame on them :D)

The rest of this documentation focuses on these features, how they were selected for the final agent, and on how the agent performed
agains other agents.



##########

2. FEATURES

---

2.1. Alpha-Beta pruning

Alpha-Beta pruning was applied, closely following the pseudocode in A+.

---

2.2. Heuristic evaluation function

Multiple different heuristic functions were added and tested:
a) Original heuristic function, which just tried to maximize the difference between player's and opponent's stones;
b) My own heuristic function, which took also the state of the board into account;
c) Function adopted from https://github.com/joansalasoler/oware, evaluates the position by the point difference,
   attacking potential (number of own pits with high number of stones), defensive restrictions (own pits with 1 or 2 stones), and
   mobility value (number of pits on the player's side which have stones at all); and
d) Function adapted from https://digitalcommons.andrews.edu/cgi/viewcontent.cgi?article=1259&context=honors, which uses multiple
   heuristics such as player's and opponent's points and total stones on player's side, and takes a weighted sum over these heuristics,
   though the version implemented here doesn't use all heuristics described in paper, and weights are slightly different.
   
Out of these, the final version uses function d. However, the heuristic function is not straight taken from the paper, as I added
one more heuristic, chose only (in my opinion) the most relevant heuristics from the paper, and played a bit with the weights.

---

2.3. State database

As the agent iterates through the depths with the Alpha-Beta pruned minimax algorithm, it really benefits if the actions on
each state is looked through in a certain order: in particular, first it should look through the best action and its' successor
states, then the second best one, and so on. This helps the algorithm to prune worse actions, and in ideal case it would only
reach leaf nodes when looking through the best action.

To help with ordering the actions suitably, the calculated heuristic values of each action on each state is checked, and the
order of the actions is saved in a dictionary. Thus, when the algorithm next time arrives on this state (iteratively on next
depth), it orders the actions based on the previous depth's values, and goes through them in this order, hopefully helping it
to prune more branches of the search tree. If the order of the actions is changed on this depth, that order is overwritten as
the new best order of the actions in the given state.

In addition, when the agent gets its next turn, it creates a new dictionary to which it records the orders on that depth;
however, as the state it currently starts on is probably on some branch it anaged to look through on previous turn also,
the algorithm orders the actions on lower depths based on the computed values on the previous turn. Only when the algorithm
manages to look deeper than on previous turn, those values are being used. These values are used on the next turn, and so on.

Author's note: Take the written code regarding to this with a grain of salt, as I didn't have the energy to exhaustively
check the correctness of the code I wrote, and it might be that I butchered something while writing it. However, the agents
with this database consistently outperformed those which hadn't it in my tests, meaning that it probably wasn't completely
useless at the least. :P

Tl;dr: The algorithm saves the current and previous turns' actions' values of each state to a dictionary, and on next visits to
said state, orders the actions based on those values to help pruning the tree with Alpha-Beta.

---

2.4. Depth skipping

The last idea was to let the iterative deepening algorithm to skip through some certain depths. This would happen, if
- the last three depths chose the same action as the best action; or
- last depth was not skipped, and during the last five depths, there has been only one action that was deemed the best (meaning that
  among the last five completed depths in ID algorithm, all levels either chose the same action as the best, or were skipped).
  
The point was that, as the last few levels chose the same action as the best, there is not some apparent fluctuation in determining
the best action. Therefore, the algorithm gambles the next depth: it doesn't go through it, potentially missing some crucial new
information; however, by doing so, it can potentially look through an even better level, due to saved time by not going through that
one depth.



##########

3. CHOOSING THE BEST COMBINATION OF FEATURES

The agents with different properties (different heuristics, state database on/off, depth skipping on/off, different weights on
heuristic functions) were tested against each others in multiple round-robin tournaments between similar agents, where each agent
played with each others, depending on the  state of the testing, 20 - 300 times. These weren't documented very well; however, it
was clear that the agents with state database outperformed  those which hadn't it, and those which didn't skip any depths in the
iterative deepening outperformed those which did so. The first (with database) was quite apparent; however, the difference between
the agents that skipped the depths and didn't was quite small, and it is possible that the results were not statistically significant
regarding that feature.

According the testing, the best heuristic function was option d (self.__heuristic4), with c (self.__heuristic3) coming close behind.
Third was option b (self.__heuristic2), and as expected, the worst heuristic function was a (self.__heuristic1). The weights of
self.__heuristic4 were also played with. The final values are based on the paper where the function were adopted from, but they were
empirically tweaked according to test results. However, that tweaking didn't follow any "real" procedure (e.g. machine learning, or
any systematic testing, really), so it is completely possible that there exists (much) better combinations for the weights.

Based on this testing, the final agent
- uses heuristic d (self.__heuristic4),
- uses state database, and
- doesn't skip depths in iterative deepening.



##########

4. TESTING AND DISCUSSION

---

4.1. Setting

The chosen agent was matched against all agents which were given with the assignment. These agents were
- Random        Chooses a random action among all possible actions,
- MCS           Uses basic Monte Carlo search,
- Minimax       Uses minimax algorithm with maximum depth 4, and
- ID-Minimax    Uses iterative deepening minimax algorithm, going through as many depths it can in the given time.

Owagent played 300 games against the first three agents (Random, MCS, Minimax), both with decision times 5 and 10 seconds.
Against the strongest agent, ID-Minimax, it played a total of 1000 games, both with 5s and 10s decision times. Half of the
games on each matchup was played as Owagent as the starting player, and the opponent started the rest of the games.

The games were played on Aalto University's Brute and Force computers via SSH connection, each game having its own processor
core. However, as the conditions were not standardized - during some game it might well have been that another user used much
more resources than on some other time - these results are not waterproof. However, they should still give a reliable picture
on the agent's preformance against each of the other agents, as the possible resource limitations apply to both players.

The results below are shown in format X-Y-Z, where X is the number of wins for player 1, Y is the number of draws,
and Z is the number of wins for player 2. For the win percentage counting, draws are considered as being worth 0.5 points;
thus, the win percentages are calculated as
Win% = (#wins + 0.5*#draws)/#total_games .


---

4.2. Results

5 seconds decision time

Owagent vs Random:          300 -  0 -  0,  Win%: 100.00%
Owagent vs MCS:             300 -  0 -  0,  Win%: 100.00%
Owagent vs Minimax:         300 -  0 -  0,  Win%: 100.00%

Owagent vs ID-Minimax:      918 - 23 - 59,  Win%:  92.95%


10 seconds decision time

Owagent vs Random:          300 -  0 -  0,  Win%: 100.00%
Owagent vs MCS:             300 -  0 -  0,  Win%: 100.00%
Owagent vs Minimax:         296 -  1 -  3,  Win%:  98.83%

Owagent vs ID-Minimax:      906 - 23 - 71,  Win%:  91.75%

---

4.3. Discussion

As expected, Owagent totally crushed the weaker agents on both decision times. The Random agent is the weakest amongst all of them,
and it comes as no surprise that basically any bot with even some logic or heuristic to guide them to the right direction would win
them. As an example, even if the Random bot would have a victory-in-one move available, it doesn't necessarily play it, if it has
other legal moves available. Also, as the decision time is totally irrelevant to this bot, it is clear that if Owagent was able to
beat it wit 5 seconds of decision time, surely it can win also with longer time so that it can look deeper. The performance of Monte
Carlo search agent (MCS) wasn't any stronger, probably as even though it can search much deeper than Owagent, the search is not
exhaustive, meaning that it can miss potentially very good or bad moves in even the depths it gets to.

Minimax is, in essence, the same as Owagent, only with weaker heuristics and lower maximum depth. The key difference is that
the maximum depth for the Minimax agent is 4, while Owagent can dig as deep it has time to. Therefore, it is no surprise either
that Owagent soundly beated that agent on both decision times. It must be noted, however, that Minimax actually beat Owagent
twice on the longer decision time, which should have, in theory, given Owagent even better edge against the other bot. There
are multiple explanations to this. It might be that the more complicated heuristic failed for some certain states. It is also
possible that on some states, even though being able to dig deeper, Owagent couldn't find any better options that could be found
on depth 4 with Minimax, rendering the deeper computation useless. However, it is still not explained, why Minimax won against
Owagent only with longer time control, or if it is just noise in the test environment.

Against ID-Minimax, the two key factors is that Owagent can dig deeper due to Alpha-Beta pruning and state database, and it has 
(empirically) stronger heuristics. Therefore, Owagent performed well against that agent as well. It is noteworthy that the match
was not as one-sided as with the other bots, but the results were still extremely clear, with the win rate being over 90% in both
time controls.

As stated earlier, the test environment was not ideal. The testing was done by multiprocessing with a computer on common use,
meaning that I didn't have control on all the resources. However, these limitation apply to both players of the game. Also, as
the results are so clear, it still gives a good estimate on how Owagent performs against those agents.



##########

5. CONCLUSION

The goal of this project was to create an AI-driven agent to play the game Oware for the tournament held in course CS-E4800.
The created agent, called Owagent, uses iterative deepening with Alpha-Beta pruning, with better heuristic evaluation function
as the given, and saves the previously computed values for actions to help the Alpha-Beta pruning to first go through the most
promising actions on each state.

Owagent was tested by letting it play against the given agents. Against the weaker agents (Random, MCS and Minimax), Owagent
played 300 games on decision times 5 seconds and 10 seconds each. It pretty much crushed the agents on bot time controls, though
lost a few games against Minimax on the higher decision time.

Against the strongest agent, ID-Minimax, Owagent also performed very well, having over 90% performance against it on both decision
times.

As future work, the heuristics could be adjusted. Also, as an idea the state-skipping tactic sounds valid, meaning that it could
be explored further. One idea would also be to follow the time intervals of when the agent's decide method is called to get an
estimate of the decision time of current game, which could be used to use different strategies on shorter and longer games.


"""



class Agent(AgentInterface):
    """
    An agent who plays the Oware game

    Methods
    -------
    `info` returns the agent's information
    `decide` chooses an action from possible actions
    """

    def __init__(self, heuristic_function = 4, is_db = True, skip_depths = False) -> None:
        # The arguments above was mainly for testing the best combination. The default values corresponds to the
        # best values gpt by testing, so this object can be initialize by just Agent() to get the final version
        # of the agent.
        super().__init__()
        self.db             = {'max_depth': 0}
        self.previous_db    = {'max_depth': 0}
        self.max_db_size    = 100_000

        self.heuristic_function = heuristic_function
        self.is_db              = is_db
        self.skip_depths        = skip_depths

    @staticmethod
    def info():
        """
        Return the agent's information

        Returns
        -------
        Dict[str, str]
            `agent name` is the agent's name
            `student name` is the list team members' names
            `student number` is the list of student numbers of the team members
        """
        return {"agent name": "Owagent",  # COMPLETE HERE
                "student name": ["Akseli Konttas"],  # COMPLETE HERE
                "student number": ["587031"]}  # COMPLETE HERE

    def heuristic(self, state: Oware):
        '''
        Evaluate the given state based of some heuristic function.
        Calls to one heuristic function and returns its' return value. The function called depends
        on the variable self.heuristic_function, set in self.__init__.

        @param state: Oware state object, for which the evaluation is done
        @return: float indicating the goodness of the state for the current player
        '''
        if self.heuristic_function == 1:
            return self.__heuristic1(state)
        if self.heuristic_function == 2:
            return self.__heuristic2(state)
        if self.heuristic_function == 3:
            return self.__heuristic3(state)
        return self.__heuristic4(state)


    def __heuristic1(self, state: Oware):
        '''
        Simple heuristic function to evaluate given state. Returns the difference of the players' points.

        @param state: Oware state object, for which the evaluation is done.
        @return: float indicating the goodness of the state for the current player
        '''
        collected_stones = state.get_collected_stone()
        return collected_stones[0] - collected_stones[1]


    def __heuristic2(self, state: Oware):
        '''
        Heuristic function to evaluate given state. Plays also with other parameters than previous.
        @param state: Oware state object, from which the evaluation is done.
        @return: float indicating the goodness of the state for the current player
        '''
        is_winner = state.is_winner()
        if is_winner is not None:
            return is_winner * float('inf')

        # Check if the game is ending due to turn limit
        collected_stones = state.get_collected_stone()
        if state.MAX_TURNS - state.get_turn_number() <= 0:
            if collected_stones[0] > collected_stones[1]:
                return float('inf')
            if collected_stones[0] > collected_stones[1]:
                return float('-inf')
            if collected_stones[0] == collected_stones[1]:
                return 0

        base_value = (collected_stones[0] - collected_stones[1])
        complexity_modification = 0
        close_to_win_modification = 0

        # Try to play with the complexness of the board
        board = state.get_board()
        total_complexity = len([x for x in board if x!= 0])/state.PITS_COUNT # Number of non-zeroes on the board
        player_complexity = len([x for x in board[:state.PITS_PER_PLAYER] if x!= 0])/state.PITS_PER_PLAYER
        player_stones = sum(board[:state.PITS_PER_PLAYER])
        #opponent_stones = sum(board[state.PITS_PER_PLAYER:])
        opponent_complexity = len([x for x in board[:state.PITS_PER_PLAYER] if x!= 0])/state.PITS_PER_PLAYER

        stones_to_win = (state.PITS_PER_PLAYER * state.INITIAL_STONES_PER_PIT)/2 +1

        # Modifications to the basic case:
        # 1) Try to leave the board complex to the opponent, so that they cannot calculate
        # extensively the best approach
        # 2) If player is winning, getting closer to the goal is preferred (even though also opponent would
        # score points). If opponent is winning, try to avoid giving them any points.
        # 3) It is preferred to hold as many stones in ones possession as possible.
        complexity_proportion = 0.2
        close_to_win_proportion = 0.6
        stone_possession_proportion = 0.3
        if collected_stones[0] > collected_stones[1]:
            complexity_modification = complexity_proportion*(-player_complexity - total_complexity)
            close_to_win_modification = close_to_win_proportion * collected_stones[0]/stones_to_win
        if collected_stones[1] > collected_stones[0]:
            complexity_modification = complexity_proportion*(opponent_complexity + total_complexity)
            close_to_win_modification = close_to_win_proportion * collected_stones[1]/stones_to_win
        stone_possession_modification = stone_possession_proportion * player_stones / sum(board)

        return base_value + complexity_modification + close_to_win_modification + stone_possession_modification


    def __heuristic3(self, state: Oware):
        '''
        Heuristic function to evaluate given state. Adopted from the Oware Abapa AI
        (https://github.com/joansalasoler/oware)
        @param state: Oware state object, for which the evaluation is done
        @return: float indicating the goodness of the state for the current player
        '''
        weigths = {'points': 25, 'attack': 28, 'defense': -36, 'mobility': -54, 'draw': -9}

        collected_stones = state.get_collected_stone()
        board = state.get_board()

        # Initial score: point difference of players
        score = weigths['points'] * (collected_stones[0] - collected_stones[1])

        # Attack potential
        score += weigths['attack'] * len([seeds for seeds in board[:state.PITS_PER_PLAYER] if seeds > 12])

        # Defense risk
        score += weigths['defense'] * len([seeds for seeds in board[:state.PITS_PER_PLAYER] if 1 <= seeds <= 2])

        # Mobility potential
        score += weigths['mobility'] * len([seeds for seeds in board[:state.PITS_PER_PLAYER] if seeds == 0])

        return score


    def __heuristic4(self, state: Oware):
        '''
        Heuristic function to evaluate given state. Based on
        https://digitalcommons.andrews.edu/cgi/viewcontent.cgi?article=1259&context=honors
        The H2, H3 etc denote to the different heuristics in said paper. However, some of the
        heuristics from the paper are omitted here. Also, the weights have been adjusted.
        @param state: Oware state object, from which the evaluation is done.
        @return: float indicating the goodness of the state for the current player
        '''
        collected_stones = state.get_collected_stone()
        board = state.get_board()

        # H2: maximize amount of seeds in own pits
        H2 = sum([seeds for seeds in board[:state.PITS_PER_PLAYER]])

        # H3: maximize the number of options the player has
        H3 = len([seeds for seeds in board[:state.PITS_PER_PLAYER]])

        # H4: maximize number of points (collected seeds)
        H4 = collected_stones[0]

        # H6: minimize opponent's points
        H6 = collected_stones[1]

        # H8: maximize the difference between player and opponent
        H8 = collected_stones[0] - collected_stones[1]

        # H11: One extra :D tries to be in lead, once the turn limit starts to be near
        H11 = (collected_stones[0] - collected_stones[1]) * (1 - (state.MAX_TURNS - state.get_turn_number())/state.MAX_TURNS)

        return 0.3*H2 + 0.3*H3 + 1.0*H4 - 0.6*H6 + 0.7*H8 + 0.6*H11
        

    def decide(self, state: Oware, actions: list):
        """
        Generate a sequence of increasingly preferable actions

        Given the current `state` and all possible `actions`, this function
        should choose the action that leads to the agent's victory.
        However, since there is a time limit for the execution of this function,
        it is possible to choose a sequence of increasing preferable actions.
        Therefore, this function is designed as a generator; it means it should
        have no return statement, but it should `yield` a sequence of increasing
        good actions.

        IMPORTANT: If no action is yielded within the time limit, the game will
        choose a random action for the player.

        Parameters
        ----------
        state: Oware
            Current state of the game
        actions: list
            List of all possible actions

        Yields
        ------
        action
            the chosen `action`
        """
        MAX_DEPTH = 1_000

        if self.is_db:
            self.previous_db = dict(self.db)
            self.db = dict()
            self.db['max_depth'] = 0

        chosen_actions = [] # Already yielded 'best actions' this turn
        values = [0]*len(actions)
        
        # Main loop: iterative deepening, going through the game tree depth by depth
        for depth in range(1, MAX_DEPTH):
            
            # If depth skipping is on: check if current depth should be skipped, and do so if necessary
            if self.skip_depths and self.skip_depth(chosen_actions):
                chosen_actions.append(None)
                continue

            best_action = actions[0]
            max_value = float('-inf')
            alpha = float('-inf')
            beta = float('inf')

            # If using state database, set the current depth as the new deepest depth
            if self.is_db:
                self.db['max_depth'] = max(depth, self.db['max_depth'])
            
            # Get the order of the actions (based on their heuristic values on previous turns) and
            # sort the actions based on that value.
            action_order = self.get_indexes_from_db(state, actions, 1)
            actions_sorted = self.select_order_by_indexes(actions, action_order)
            
            # Go through each action ond start going deeper on depths with self.alphabeta
            for idx,action in enumerate(actions_sorted):
                action_value = self.alphabeta(-1, state.successor(action), depth, alpha, beta)      # Main call: go to alphabeta minimax search from current state with current action
                values[idx] = action_value
                alpha = max(max_value,alpha)    # Update the alpha value according to the acquired value
                if action_value > max_value:    # If got better action than the previously ecorded best, save this as best
                    max_value = action_value
                    best_action = action
            #print(f'Depth: {depth}, pruned: {pruned}, states visited: {states_visited}, db size: {len(self.db)}, matched: {db_hit}, values: {values}, best action: {best_action}, value of action: {max_value}')
            chosen_actions.append(best_action)  # For the depth skipping, add the chosen action to list
            
            # Save the values to the state database
            self.db[(hash(state), 1)] = self.sort_list_by_list(action_order, values, descending=True)
            
            # Finally, yield the computed best action
            yield best_action

            # If a certain victory (value of inf) was found on lower depths, just follow it -> no need to search deeper,
            # better to follow the quickest path to certain victory.
            # No need to be fancy :P
            if max_value == float('inf'):
                break

    def skip_depth(self, chosen_actions):
        '''
        Determines if the current depth can be semi-safely skipped.
        Logic: If many depths in succession have chosen the same action for best action, we can quite safely skip the
        next depth to maybe get even more depth in the search.
        @param chosen_actions: list consisting of all yielded 'best actions' this turns, having None on each turn that was skipped
        '''
        # If only below 3 levels, don't skip
        n = len(chosen_actions)
        if n < 3:
            return False

        # If last three levels have same opinion of best action, skip
        if chosen_actions[-1] is not None and chosen_actions[-1] == chosen_actions[-2] and chosen_actions[-2] == chosen_actions[-3]:
            return True
        
        # If among the last five turns there have only been one cerftain action presented as best actions, in addition to possible
        # skippings, and the privious depth had not been skipped -> skip this depth
        if n >= 5 and chosen_actions[-1] != None:
            last_five_turns = chosen_actions[n-5:]
            if len([x for x in last_five_turns if x not in (None, chosen_actions[-1])]) == 0:
                return True
        
        # Otherwise, do not skip this depth
        return False


    def alphabeta(self, player, state, depht_left, alpha, beta):
        '''
        Minimax game tree search with alphabeta pruning.
        @param player: 1 or -1. Denoting if the current player tries to maximize (1) or minimize (-1) the value.
        @param state: Oware object of current state.
        @param depth_left: How much deeper can still go on this particular iteration (int)
        @param alpha: Alpha value of the alpha-beta pruning (float)
        @param beta: Beta value of the alpha-beta pruning (float)
        
        @return: Given state's value (float)
        '''
        # Termination condition
        is_winner = state.is_winner()
        if is_winner is not None:
            return is_winner * player * float('inf')
        if depht_left == 0:
            return player * self.heuristic(state)

        actions = state.actions()

        # Get the order of actions (from best to worst) from the previously acquired list of indexes
        previous_action_order = self.get_indexes_from_db(state, actions, player)
        actions_sorted = self.select_order_by_indexes(actions, previous_action_order)

        # Value of each action
        action_values = [-player * float('inf')]*len(actions)   

        # Main loop: go through each of the possible actions, record their values, and store the currently
        # best value
        best = -player * float('inf')   # Initial values of the actions: for maximizing palyer, all ar -inf, for minimizing player, all ar inf
        for idx, action in enumerate(actions_sorted):
            value = self.alphabeta(-player, state.successor(action), depht_left-1, alpha, beta) # Call to the next depth of alpha-beta minimax
            action_values[idx] = value
            
            # Save the current best action and update alpha/beta if needed
            if player == 1:
                best = max(best, value)
                alpha = max(best, alpha)
            else:
                best = min(best, value)
                beta = min(best, beta)
                
            # If alpha >= beta, there cannot be any better options left -> break the loop early
            if alpha >= beta:
                break

        # Save the new order of actions to the dictionary
        if self.is_db and len(self.db) < self.max_db_size:
            if player == 1:
                new_action_order = self.sort_list_by_list(previous_action_order, action_values, descending=True)
            else:
                new_action_order = self.sort_list_by_list(previous_action_order, action_values, descending=False)
            self.db[(hash(state), player)] = new_action_order

        # Return value of this state
        return best
    
    def get_indexes_from_db(self, state, actions, player):
        '''
        Returns the order of indexes of the actions of the current state.
        @param state: Oware object of current state
        @param player: -1 or 1, denoting the player
        @return: list containing the indexes of actions, from best to worst
        '''
        # Case 1: Using database, and current state had been recorded last turn, and on last turn we got deeper than
        # we currently have gone this turn (the difference of previous turn and this turn being 2 depths) -> check the order
        # from last turn's dictionary
        if self.is_db and (hash(state),player) in self.previous_db and self.previous_db['max_depth']-2 > self.db['max_depth']:
            previous_action_order = self.previous_db[(hash(state),player)]
            
        # Case 2: Using database, and either didn't find the state in last turn's dictionary or we have already gone deeper this turn
        # -> check the order from this turn's dictionary
        elif self.is_db and (hash(state),player) in self.db:
            previous_action_order = self.db[(hash(state),player)]
            
        # Not using database, or didn't find the state in either dictionary
        # -> just take the original ordering and shuffle them randomly to get initial order.
        else:
            previous_action_order = [i for i in range(len(actions))]
            random.shuffle(previous_action_order)
        
        return previous_action_order


    def sort_list_by_list(self, to_be_sorted: list, values: list, descending=True):
        '''
        Sorts the items in list 'to_be_sorted' according to the values in list 'values': the value of first
        index in 'values' points to first index in 'to_be_sorted', and so on.
        For example: to_be_sorted = [a,b,c,d,e], values = [1,3,4,2,0],
                    sort_list_by_list(to_be_sorted, values) -> [c,b,d,a,e]
        @return: A new, sorted list.
        '''
        return [item for _,item in sorted(zip(values, to_be_sorted), key=lambda pair: pair[0], reverse=descending)]

    def select_order_by_indexes(self, to_be_sorted: list, indexes: list):
        '''
        Sorts the items in list 'to_be_sorted' according to the indexes in list 'indexes':
        the first value in 'indexes' tells the index of to_be_sorted which should go first, and so on.
        For example: to_be_sorted = [a,b,c,d,e], indexes = [1,3,4,2,0],
                    select_order_by_indexes(to_be_sorted, indexes) -> [b,d,e,c,a]
        @return: A new, sorted list.
        '''
        try:
            return [to_be_sorted[idx] for idx in indexes]
        # For some reason, sometimes the lists were not of equal length. I'm not
        # completely sure where the error was, but if that happens, just return the original list.
        except IndexError:
            return to_be_sorted
