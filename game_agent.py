"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""

import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Get the number of legal move for the active player
    score_player_1 = float(len(game.get_legal_moves(game._active_player)))
    # Get the number of legal moves for the inactive player
    score_player_2 = float(len(game.get_legal_moves(game._inactive_player)))
    # Get the difference of legal moves between the 2 players
    score = score_player_1 - 1.5*score_player_2
    # If the score is a positive number, than Player 1 has a greater chance of winning
    return score

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    The closer the players are to the edge of the board game, the fewer options they have to move.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Center of the board
    center = game.height/2

    # Get the position of Player 1 and Player 2 on the board
    player_1_position = game.get_player_location(player)
    player_2_position = game.get_player_location(game.get_opponent(player))

    # Get the number of legal moves for Player 1 and Player 2
    player_1_moves = len(game.get_legal_moves(player))
    player_2_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # Get the distance of Player 1 from the center
    player_1_distance_row = abs(center - player_1_position[0])
    player_1_distance_col = abs(center - player_1_position[1])

    # Get the distance of Player 2 from the center
    player_2_distance_row = abs(center - player_2_position[0])
    player_2_distance_col = abs(center - player_2_position[1])

    # The score is greater if Player 1 has more moves than Player 2 and if the distance of Player 1 is close to the center.
    score = 3*(player_1_moves - player_2_moves) + (player_1_distance_row + player_1_distance_col) - (player_2_distance_row + player_2_distance_col)

    # If the score is a positive number, than Player 1 has a greater chance of winning
    return float(score)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    score_player_1 = float(len(game.get_legal_moves(game._active_player)))
    # Get the number of legal moves for the inactive player
    score_player_2 = float(len(game.get_legal_moves(game._inactive_player)))
    # Get the difference of legal moves between the 2 players
    score = 2.5*score_player_1 - score_player_2


    return score


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting


        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout

        # Get legal moves
        legal_moves = game.get_legal_moves()

        # If no legal moves remains return the utility value of the current game state for the specified player and the coordinate of best move
        if not legal_moves:
            return game.utility(self), (-1, -1)
        # Initialize best_move
        best_move = None
        # Initialize best_score to - inf, small score
        best_score = float("-Inf")

        # For all legal moves, get the minimum value of the highest score
        for move in legal_moves:
            score = self.min_value(game.forecast_move(move), depth - 1)
            if score > best_score:
                best_score = score
                best_move = move
        # Return the coordinate of the best move with the highest score
        return best_move

    def max_value(self, game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
               raise SearchTimeout()
            #
            if depth == 0:
                return self.score(game, self)
            # Get all legal moves
            legal_moves = game.get_legal_moves()
            # if no legal moves return the utility value of the current game state for the specified player
            if not legal_moves:
                return game.utility(self)
            # Initialize best_move
            best_move = None
            # Initialize best_score to -inf, small score
            best_score = float("-Inf")
            # For all legal moves, get the maximum value of the lowest score
            for move in legal_moves:
                best_score = max(best_score,self.min_value(game.forecast_move(move),depth - 1))
            return best_score

    def min_value(self, game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0:
                return self.score(game, self)
            # Get all legal moves
            legal_moves = game.get_legal_moves()
            # If no legal moves remains return the utility value of the current game state for the specified player
            if not legal_moves:
                return game.utility(self)
            # Initialize best_move
            #best_move = None
            # Initialize best_score to inf, big score
            best_score = float("Inf")
            # For all legal moves, get the minimum value of the highest score
            for move in legal_moves:
                best_score = min(best_score, self.max_value(game.forecast_move(move), depth - 1))
            return best_score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Get all the legal moves
        legal_moves = game.get_legal_moves()
        # If no legal moves remains return the utility value of the current game state for the specified player and the coordinate of best move\
        best_move = (-1,-1)
        if not legal_moves:
           return best_move
            # The try/except block will automatically catch the exception raised when the timer is about to expire.
            # It will returns a good move before the search time limit expires.
        depth = 1
        try:
            while True:
                best_move = self.alphabeta(game, depth)
                depth = depth +1
        except SearchTimeout:
                pass
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


        # If no legal moves remains return the utility value of the current game state for the specified player and the coordinate of best move

        # Initialize best_move to zero
        #best_move = None
        # Initialize best score to -inf, small score

        # Initialize best move
        best_move = (-1, -1)
    # For all the legal moves get the minimum value of the maximum score
        # Get all the legal moves
        legal_moves = game.get_legal_moves()
        if not legal_moves:
           return game.utility(self)
        best_score = float("-Inf")
        for move in legal_moves:

            score = self.alpha_beta_min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
        return best_move

    def alpha_beta_max_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)
        # Initialize best_move
        #best_move = (-1, -1)

        # Get all the legal moves
        legal_moves = game.get_legal_moves()

        # If no legal moves remains return the utility value of the current game state for the specified player and the coordinate of best move
        if not legal_moves:
           return game.utility(self)
            #return (-1,-1)
        # Initialize best_move to zero
        #best_move = None
        # Initialize best_move to -inf, small score
        best_score = float("-Inf")
        for move in legal_moves:
            # For all the legal moves get the minimum value of the maximum score
            best_score = max(best_score, self.alpha_beta_min_value(game.forecast_move(move), depth - 1, alpha, beta))
            if best_score >= beta:
                return best_score
            alpha = max(alpha, best_score)
        return best_score

    def alpha_beta_min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0:
            return self.score(game, self)
        # Initilize best_move
        #best_move = (-1,-1)
        # Get all legal moves
        legal_moves = game.get_legal_moves()
        # If no legal moves remains return the utility value of the current game state for the specified player and the coordinate of best move
        if not legal_moves:
            return game.utility(self)

        # Initilize best_move to zero
       # best_move = None
        # Initilize best_score to inf, high score
        best_score = float("Inf")
        # For all the legal moves get the maximum value of the minimum score
        for move in legal_moves:
            best_score = min(best_score, self.alpha_beta_max_value(game.forecast_move(move), depth - 1, alpha, beta))
            if best_score <= alpha:
                return best_score
            beta = min(beta, best_score)
        return best_score

