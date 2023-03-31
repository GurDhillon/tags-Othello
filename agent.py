"""
An AI player for Othello.

What my heuristic considers:
1. Ratio of moves available to the player vs the opponent. The heuristic assigns a greater value
when the player has more moves available and the opponent has less.
2. Ratio of potential moves available to the player vs the opponent. These are the moves that could be
available in the future, even if they are not available right now. The heuristic assigns a greater value
when the player has more potential moves available and the opponent has less.
3. Number of corners taken. The heuristic assigns a greater value when a corner is taken by the player, and
it assigns a lower value if it is taken by the opponent.
"""

import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

cache = None


def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)
    
# Method to compute utility value of terminal state
def compute_utility(board, color):
    #IMPLEMENT
    num_dark, num_light = get_score(board)
    if color == 1:
        return num_dark - num_light
    return num_light - num_dark


def check_empty(board, coordinates):
    """Return true if there is an empty block around coordinates.
    """
    n = len(board)
    for i in range(coordinates[0] - 1, coordinates[0] + 2):
        if 0 <= i <= n - 1:
            for j in range(coordinates[1] - 1, coordinates[1] + 2):
                if (0 <= j <= n - 1) and (i, j) != coordinates:
                    if board[i][j] == 0:
                        return True

    return False


def count_empty(board, color):
    """Return the number of pieces of color "color" that have an empty space around them.
    """
    n, total = len(board), 0
    for i in range(n):
        for j in range(n):
            if board[i][j] == color:
                if check_empty(board, (i, j)):
                    total += 1

    return total


# Better heuristic value of board
def compute_heuristic(board, color):  # not implemented, optional
    # IMPLEMENT
    opponent = 1
    if color == 1:
        opponent = 2

    result = 0

    my_mobility = len(get_possible_moves(board, color))
    opp_mobility = len(get_possible_moves(board, opponent))
    if my_mobility + opp_mobility != 0:
        result += 100 * ((my_mobility - opp_mobility) / (my_mobility + opp_mobility))

    my_potmobility = count_empty(board, color)
    opp_potmobility = count_empty(board, opponent)
    if my_potmobility + opp_potmobility != 0:
        result += 100 * ((my_potmobility - opp_potmobility) / (my_potmobility + opp_potmobility))

    n = len(board)
    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    my_corners, opp_corners = 0, 0
    for corner in corners:
        corner_color = board[corner[0]][corner[1]]
        if corner_color == color:
            my_corners += 1
        elif corner_color == opponent:
            opp_corners += 1

    if my_corners + opp_corners != 0:
        result += 100 * ((my_corners - opp_corners) / (my_corners + opp_corners))
    return result


############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching = 0):
    #IMPLEMENT (and replace the line below)
    if caching:
        if board in cache:
            return None, cache[board]

    opponent = 1
    if color == 1:
        opponent = 2

    moves = get_possible_moves(board, opponent)
    if not moves or not limit:
        util = compute_utility(board, color)
        if caching:
            cache[board] = util
        return None, util

    best_move, min_util = None, float("inf")
    for move in moves:
        new_board = play_move(board, opponent, move[0], move[1])
        _, new_util = minimax_max_node(new_board, color, limit - 1, caching)
        if new_util < min_util:
            best_move, min_util = move, new_util

    if caching:
        cache[board] = min_util

    return best_move, min_util


def minimax_max_node(board, color, limit, caching = 0): #returns highest possible utility
    #IMPLEMENT (and replace the line below)
    if caching:
        if board in cache:
            return None, cache[board]

    moves = get_possible_moves(board, color)
    if not moves or not limit:
        util = compute_utility(board, color)
        if caching:
            cache[board] = util
        return None, util

    best_move, max_util = None, float("-inf")
    for move in moves:
        new_board = play_move(board, color, move[0], move[1])
        _, new_util = minimax_min_node(new_board, color, limit - 1, caching)
        if new_util > max_util:
            best_move, max_util = move, new_util

    if caching:
        cache[board] = max_util

    return best_move, max_util


def select_move_minimax(board, color, limit, caching = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    """
    #IMPLEMENT (and replace the line below)
    global cache
    cache = dict()
    return minimax_max_node(board, color, limit, caching)[0]


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT (and replace the line below)
    if caching:
        if board in cache:
            return None, cache[board]

    opponent = 1
    if color == 1:
        opponent = 2

    moves = get_possible_moves(board, opponent)
    if not moves or not limit:
        util = compute_utility(board, color)
        if caching:
            cache[board] = util
        return None, util

    successors = []
    for move in moves:
        successors.append((play_move(board, opponent, move[0], move[1]), move))

    if ordering:
        successors.sort(key=lambda successor: compute_utility(successor[0], color))

    best_move, min_util = None, float("inf")
    for successor in successors:
        _, new_util = alphabeta_max_node(successor[0], color, alpha, beta, limit - 1, caching, ordering)
        if new_util < min_util:
            best_move, min_util = successor[1], new_util

        if min_util <= alpha:
            break
        beta = min(beta, min_util)

    if caching:
        cache[board] = min_util

    return best_move, min_util


def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT (and replace the line below)
    if caching:
        if board in cache:
            return None, cache[board]

    moves = get_possible_moves(board, color)
    if not moves or not limit:
        util = compute_utility(board, color)
        if caching:
            cache[board] = util
        return None, util

    successors = []
    for move in moves:
        successors.append((play_move(board, color, move[0], move[1]), move))

    if ordering:
        successors.sort(key=lambda successor: compute_utility(successor[0], color), reverse=True)

    best_move, max_util = None, float("-inf")
    for successor in successors:
        _, new_util = alphabeta_min_node(successor[0], color, alpha, beta, limit - 1, caching, ordering)
        if new_util > max_util:
            best_move, max_util = successor[1], new_util

        if max_util >= beta:
            break
        alpha = max(alpha, max_util)

    if caching:
        cache[board] = max_util

    return best_move, max_util


def select_move_alphabeta(board, color, limit, caching = 0, ordering = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    #IMPLEMENT (and replace the line below)
    global cache
    cache = dict()
    return alphabeta_max_node(board, color, float("-inf"), float("inf"), limit - 1, caching, ordering)[0]


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")
    
    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light. 
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching 
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)
            
            print("{} {}".format(movei, movej))

if __name__ == "__main__":
    run_ai()
