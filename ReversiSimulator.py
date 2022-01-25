import time
import numpy as np
import numba
import matplotlib.pyplot as plt
import typing

from timeout_decorator import timeout

CHESSBOARD_SIZE = 8
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0


def evaluate_competition(candidate_information: typing.Tuple[int, list]):
    """
    :param candidate_information: Tuple[candidate_index, population]
    """
    candidate_index, population = candidate_information
    candidate = population[candidate_index]
    scores = np.zeros(shape=len(population), dtype=np.intc)
    game = ReversiSimulator()
    for opponent_index in range(len(population)):
        if opponent_index == candidate_index:
            continue
        game.initialize_agents(black=candidate,
                               white=population[opponent_index])
        game.quick_run()
        winner = game.get_winner()
        if winner == COLOR_BLACK:
            scores[candidate_index] += 1
            scores[opponent_index] -= 1
        elif winner == COLOR_WHITE:
            scores[candidate_index] -= 1
            scores[opponent_index] += 1
    return scores


class ReversiSimulator(object):
    def __init__(self, chessboard_size=CHESSBOARD_SIZE):
        self.chessboard_size = chessboard_size
        self.chessboard = None

        self.white_agent = None
        self.black_agent = None
        self.current_player = COLOR_NONE

        self.directions = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
        self.marker_size = 2400
        self.game_running = False
        self.game_step = 0

        self.time_behaviour = 'warn'
        self.time_limit = 5.0
        self.time_wait = 10.0
        self.time_behaviours = {'ignore', 'wait', 'warn', 'print', 'exception', 'interrupt'}

        self.chessboard_records = []
        self.black_run_time = []
        self.white_run_time = []
        self.white_eval_list = []
        self.black_eval_list = []

    def set_time_limit(self, limit=5.0, wait=10.0, behaviour='warn'):
        assert behaviour in self.time_behaviours, f'Invalid time behaviour {self.time_behaviours}'
        self.time_limit = limit
        self.time_wait = wait
        self.time_behaviour = behaviour

    def get_time_limit(self):
        if self.time_behaviour in {'warn', 'wait', 'print'}:
            return self.time_wait
        elif self.time_behaviour in {'exception', 'interrupt'}:
            return self.time_limit
        else:
            return np.Inf

    def get_player(self):
        try:
            if self.current_player == COLOR_BLACK:
                return f"BLACK: {self.black_agent.name}"
            elif self.current_player == COLOR_WHITE:
                return f"WHITE: {self.white_agent.name}"
        except AttributeError:
            if self.current_player == COLOR_BLACK:
                return "BLACK"
            elif self.current_player == COLOR_WHITE:
                return "WHITE"

    def print_winner(self):
        try:
            if self.get_winner() == COLOR_BLACK:
                print(f"BLACK: {self.black_agent.name} WINS")
            elif self.get_winner() == COLOR_WHITE:
                print(f"WHITE: {self.white_agent.name} WINS")
            else:
                print("DRAW")
        except AttributeError:
            if self.get_winner() == COLOR_BLACK:
                print("BLACK WINS")
            elif self.get_winner() == COLOR_WHITE:
                print("WHITE WINS")
            else:
                print("DRAW")

    def initialize_agents(self, black, white):
        self.chessboard = np.zeros([self.chessboard_size, self.chessboard_size], dtype=int)
        centre = self.chessboard_size // 2
        self.chessboard[centre, centre] = 1
        self.chessboard[centre - 1, centre - 1] = 1
        self.chessboard[centre - 1, centre] = -1
        self.chessboard[centre, centre - 1] = -1

        self.current_player = COLOR_BLACK

        self.white_agent = white
        self.black_agent = black
        self.black_agent.color = COLOR_BLACK
        self.white_agent.color = COLOR_WHITE

        self.chessboard_records = []
        self.black_run_time = []
        self.white_run_time = []
        self.white_eval_list = []
        self.black_eval_list = []
        self.game_running = True

    def quick_run(self):
        assert self.game_running
        while self.game_running:
            current_agent = self.black_agent if self.current_player == COLOR_BLACK else self.white_agent
            current_agent.go(self.chessboard)

            if current_agent.candidate_list:
                self.update_chessboard(current_agent.candidate_list[-1], self.current_player)
            else:
                self.game_running = bool(generate_legal_actions(self.chessboard, -self.current_player))

            self.current_player = -self.current_player

    def quick_step(self):
        assert self.game_running
        current_agent = self.black_agent if self.current_player == COLOR_BLACK else self.white_agent
        current_agent.go(self.chessboard)

        if current_agent.candidate_list:
            self.update_chessboard(current_agent.candidate_list[-1], self.current_player)
        else:
            self.game_running = bool(generate_legal_actions(self.chessboard, -self.current_player))

        self.current_player = -self.current_player
        return True

    def step(self):
        """
        Step with time and game rule check
        """
        assert self.game_running
        self.game_step += 1

        assert self.current_player == COLOR_BLACK or self.current_player == COLOR_WHITE
        current_agent = self.black_agent if self.current_player == COLOR_BLACK else self.white_agent

        wait_time = self.get_time_limit()
        start = time.time()
        timeout(wait_time, use_signals=False)(current_agent.go)(self.chessboard)
        current_agent.go(self.chessboard)
        duration = time.time() - start

        if self.time_behaviour == 'print':
            print(f'{self.get_player()} Time: {duration}')
        elif self.time_behaviour == 'warn' and duration > self.time_limit:
            print(f'{self.get_player()} Time Limit Exceed at step {self.game_step} Time: {duration}')

        self.chessboard_records.append(self.chessboard.copy())

        if self.current_player == COLOR_WHITE:
            self.white_run_time.append(duration)
        else:
            self.black_run_time.append(duration)

        if current_agent.candidate_list:
            position = current_agent.candidate_list[-1]
            assert self.chessboard[position] == COLOR_NONE, f'Illegal move: {position} is already captured'
            update_list = self.get_update_list(position, self.current_player)
            assert update_list, f'Illegal move: {position} affects no components'
            self.chessboard[position] = self.current_player
            for point in update_list:
                self.chessboard[point] = self.current_player
        else:
            player_actions = generate_legal_actions(self.chessboard, self.current_player)
            assert player_actions, f'Illegal action: there is at least 1 available points, ' + \
                                   f'but {self.get_player()} gives no point.'
            if not generate_legal_actions(self.chessboard, -self.current_player):
                self.game_running = False
                self.chessboard_records.append(self.chessboard.copy())

        self.current_player = -self.current_player
        return True

    def update_chessboard(self, position, player):
        update_list = get_update_list(self.chessboard, position, player)
        self.chessboard[position] = player
        for point in update_list:
            self.chessboard[point] = player
        return np.array(update_list, dtype=int)

    def get_update_list(self, position, player):
        indexes_list = []
        for direction in self.directions:
            iter_x, iter_y = position[0] + direction[0], position[1] + direction[1]
            direction_indexes_list = [(iter_x, iter_y)]
            if not (0 <= iter_x < self.chessboard_size and 0 <= iter_y < self.chessboard_size):
                continue  # the move is out of random_args, abandon this direction
            while self.chessboard[iter_x, iter_y] == -player:
                iter_x += direction[0]
                iter_y += direction[1]
                if not (0 <= iter_x < self.chessboard_size and 0 <= iter_y < self.chessboard_size):
                    # indicates that the search is out of random_args, abandon this direction
                    break
                if self.chessboard[iter_x, iter_y] == player:
                    indexes_list += direction_indexes_list
                    break
                direction_indexes_list.append((iter_x, iter_y))
        return indexes_list

    def plot_chessboard(self):
        blank_indexes = np.where(self.chessboard == COLOR_NONE)
        black_indexes = np.where(self.chessboard == COLOR_BLACK)
        white_indexes = np.where(self.chessboard == COLOR_WHITE)
        plt.axes(facecolor='#afdfe4')
        plt.scatter(blank_indexes[0], blank_indexes[1], marker=".", c="gray", s=self.marker_size)
        plt.scatter(black_indexes[0], black_indexes[1], marker=".", c="black", s=self.marker_size)
        plt.scatter(white_indexes[0], white_indexes[1], marker=".", c="white", s=self.marker_size)
        plt.show()

    def get_winner(self):
        assert not self.game_running
        white_count = np.sum(self.chessboard == COLOR_WHITE)
        black_count = np.sum(self.chessboard == COLOR_BLACK)
        return COLOR_WHITE if black_count > white_count else COLOR_BLACK if white_count > black_count else COLOR_NONE


@numba.njit(cache=True)
def generate_legal_actions(chessboard: np.ndarray, player: int):
    blank_index = np.where(chessboard == COLOR_NONE)
    blank_positions = zip(blank_index[0], blank_index[1])
    legal_moves = []
    for position in blank_positions:
        if is_legal_position(chessboard, position, player):
            legal_moves.append(position)
    return legal_moves


@numba.njit(cache=True)
def is_legal_position(chessboard: np.ndarray, position: tuple, player: int):
    chessboard_size = chessboard.shape[0]
    for direction in ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)):
        iter_x = position[0] + direction[0]
        iter_y = position[1] + direction[1]
        if not (0 <= iter_x < chessboard_size and 0 <= iter_y < chessboard_size):
            continue  # the indexes are out of random_args, abandon this direction
        while chessboard[iter_x, iter_y] == -player:
            iter_x += direction[0]
            iter_y += direction[1]
            if not (0 <= iter_x < chessboard_size and 0 <= iter_y < chessboard_size):
                break
            if chessboard[iter_x, iter_y] == player:
                return True
    return False


@numba.njit(cache=True)
def get_update_list(chessboard, position, player):
    update_list = []
    chessboard_size = np.shape(chessboard)[0]
    for direction in ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)):
        iter_x = position[0] + direction[0]
        iter_y = position[1] + direction[1]
        direction_indexes_list = [(iter_x, iter_y)]
        if not (0 <= iter_x < chessboard_size and 0 <= iter_y < chessboard_size):
            continue
        while chessboard[iter_x, iter_y] == -player:
            iter_x += direction[0]
            iter_y += direction[1]
            if not (0 <= iter_x < chessboard_size and 0 <= iter_y < chessboard_size):
                break
            if chessboard[iter_x, iter_y] == player:
                [update_list.append(x) for x in direction_indexes_list]
                break
            direction_indexes_list.append((iter_x, iter_y))
    return update_list
