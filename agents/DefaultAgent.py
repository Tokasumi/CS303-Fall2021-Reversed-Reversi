import numpy as np
import random
import numba

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0


class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size=8, color=COLOR_NONE, time_out=5.0):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need add your decision into your candidate_list. System will get the end of your candidate_list as your
        # decision.
        self.candidate_list = []

        self.weight_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.chessboard_weight = assign_weight_array(self.weight_vector, self.chessboard_size)
        self.stability_weight = 1
        self.mobility_weight = (0, 0, 0, 0)
        self.frontier_weight = (0, 0)

        self.directions = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
        self.INF = 1e+8
        self.WIN_GAME = 1e+5
        self.SEARCH_DEPTH = 0
        self.UTILITY_THRESHOLD = 3
        self.chessboard = None

    def to_list(self):
        return list(self.weight_vector) + [self.stability_weight] + \
               list(self.mobility_weight) + list(self.frontier_weight)

    def from_list(self, arg_list):
        self.weight_vector = arg_list[:10]
        self.stability_weight = arg_list[10]
        self.mobility_weight = (arg_list[11], arg_list[12], arg_list[13], arg_list[14])
        self.frontier_weight = (arg_list[15], arg_list[16])
        self.chessboard_weight = assign_weight_array(self.weight_vector, self.chessboard_size)

    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()

        self.chessboard = chessboard
        blank_points = np.sum(chessboard == COLOR_NONE)
        possible_actions = generate_actions_filter(chessboard, self.color)

        if not possible_actions:
            return []
        else:
            random.shuffle(possible_actions)
            self.candidate_list = possible_actions

        if blank_points <= self.UTILITY_THRESHOLD:
            self.candidate_list = self.min_max_decision(self.UTILITY_THRESHOLD, possible_actions)
        else:
            self.candidate_list = self.min_max_decision(self.SEARCH_DEPTH, possible_actions)
        return self.candidate_list

    def _eval(self):
        return evaluate_chessboard(chessboard=self.chessboard, color=self.color,
                                   stability_weight=self.stability_weight,
                                   mobility_weight=self.mobility_weight,
                                   frontier_weight=self.frontier_weight,
                                   chessboard_weight=self.chessboard_weight
                                   )

    def min_max_decision(self, depth, possible_actions):
        alpha = -self.INF

        actions_list = []
        if depth == 0:
            for action in possible_actions:
                update_array = update_chessboard(self.chessboard, action, self.color)
                value = self._eval()
                revert_chessboard(self.chessboard, action, self.color, update_array)

                actions_list.append((value, action))
        else:
            depth -= 1
            for action in possible_actions:
                update_array = update_chessboard(self.chessboard, action, self.color)
                value = self.min_value_generating(-self.color, depth, alpha, self.INF)
                alpha = max(value, alpha)
                revert_chessboard(self.chessboard, action, self.color, update_array)

                actions_list.append((value, action))
        actions_list.sort(key=lambda elem: elem[0])
        return [elem[1] for elem in actions_list]

    def max_value_generating(self, player, depth, alpha, beta):
        if depth == 0:
            return self._eval()

        value = -self.INF

        blank_points_index = np.where(self.chessboard == COLOR_NONE)
        blank_points = zip(blank_points_index[0], blank_points_index[1])
        no_step = True

        if blank_points_index[0].size > 0:
            for point in blank_points:
                if is_legal_position(self.chessboard, point, player):
                    no_step = False

                    update_array = update_chessboard(self.chessboard, point, self.color)
                    value = max(value, self.min_value_generating(-player, depth - 1, alpha, beta))
                    revert_chessboard(self.chessboard, point, player, update_array)

                    if value >= beta:
                        return value
                    alpha = max(value, alpha)
            if no_step:
                return self.min_value_generating(-player, depth - 1, alpha, beta)
        else:
            return self._eval()
        return value

    def min_value_generating(self, player, depth, alpha, beta):
        if depth == 0:
            return self._eval()

        value = self.INF

        blank_points_index = np.where(self.chessboard == COLOR_NONE)
        blank_points = zip(blank_points_index[0], blank_points_index[1])
        no_step = True

        if blank_points_index[0].size > 0:
            for point in blank_points:
                if is_legal_position(self.chessboard, point, player):
                    no_step = False

                    update_array = update_chessboard(self.chessboard, point, self.color)
                    value = min(value, self.max_value_generating(-player, depth - 1, alpha, beta))
                    revert_chessboard(self.chessboard, point, player, update_array)

                    if value <= alpha:
                        return value
                    beta = min(value, beta)
            if no_step:
                return self.max_value_generating(-player, depth - 1, alpha, beta)
        else:
            return self._eval()
        return value


@numba.njit(cache=True)
def generate_actions_filter(chessboard: np.ndarray, player: int):
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
def find_boundaries(chessboard: np.ndarray):
    chessboard_size = chessboard.shape[0]
    max_index = chessboard_size - 1

    black_boundaries = np.zeros((chessboard_size, chessboard_size), dtype=np.intc)
    white_boundaries = np.zeros((chessboard_size, chessboard_size), dtype=np.intc)
    stability = [None, white_boundaries, black_boundaries]

    for x, y in ((0, 0), (0, 1), (1, 0), (1, 1)):
        i_depth, j_depth = chessboard_size, chessboard_size
        capturing_player = chessboard[x * max_index, y * max_index]
        if capturing_player == COLOR_NONE:
            continue
        for i in range(0, chessboard_size, 1):
            for j in range(0, j_depth, 1):
                point = max_index - i if x else i, max_index - j if y else j
                if chessboard[point] == capturing_player:
                    stability[capturing_player][point] = 1
                else:
                    j_depth = j - 1
                    break
            if j_depth < 0:
                break
        for j in range(0, chessboard_size, 1):
            for i in range(0, i_depth, 1):
                point = max_index - i if x else i, max_index - j if y else j
                if chessboard[point] == capturing_player:
                    stability[capturing_player][point] = 1
                else:
                    i_depth = i - 1
                    break
            if i_depth < 0:
                break

    return np.sum(black_boundaries), np.sum(white_boundaries)


def assign_weight_array(v, csize):
    assert csize == 8

    weight_matrix = np.array([
        [v[9], v[8], v[6], v[3], v[3], v[6], v[8], v[9]],
        [v[8], v[7], v[5], v[2], v[2], v[5], v[7], v[8]],
        [v[6], v[5], v[4], v[1], v[1], v[4], v[5], v[6]],
        [v[3], v[2], v[1], v[0], v[0], v[1], v[2], v[3]],
        [v[3], v[2], v[1], v[0], v[0], v[1], v[2], v[3]],
        [v[6], v[5], v[4], v[1], v[1], v[4], v[5], v[6]],
        [v[8], v[7], v[5], v[2], v[2], v[5], v[7], v[8]],
        [v[9], v[8], v[6], v[3], v[3], v[6], v[8], v[9]]
    ])
    return weight_matrix


@numba.njit(cache=True)
def find_sentinels(chessboard: np.ndarray):
    max_index = chessboard.shape[0] - 1
    blanks_indexes = np.where(chessboard == COLOR_NONE)
    sentinel_map = np.zeros_like(chessboard)
    for dx, dy in ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)):
        for point in zip(blanks_indexes[0] + dx, blanks_indexes[1] + dy):
            if 0 <= point[0] <= max_index and 0 <= point[1] <= max_index:
                sentinel_map[point] = 1
    black_sentinels = (chessboard == COLOR_BLACK) & sentinel_map
    white_sentinels = (chessboard == COLOR_WHITE) & sentinel_map
    return np.sum(black_sentinels), np.sum(white_sentinels)


def find_sentinels_nonumba(chessboard: np.ndarray):
    size = chessboard.shape[0]
    row, col = np.zeros(size, dtype=np.bool_), np.zeros((size, 1), dtype=np.bool_)
    blank_map = (chessboard == COLOR_NONE)
    horizontal = blank_map | np.vstack((row, blank_map[:size - 1, :])) | np.vstack((blank_map[1:, :], row))
    sentinel_map = horizontal | np.hstack((col, horizontal[:, :size - 1])) | np.hstack((horizontal[:, 1:], col))
    black_sentinels = (chessboard == COLOR_BLACK) & sentinel_map
    white_sentinels = (chessboard == COLOR_WHITE) & sentinel_map
    return np.sum(black_sentinels), np.sum(white_sentinels)


def revert_chessboard(chessboard, position, player, update_array):
    chessboard[position] = COLOR_NONE
    for point in update_array:
        chessboard[point[0], point[1]] = -player


def update_chessboard(chessboard, position, player):
    update_list = get_update_list(chessboard, position, player)
    chessboard[position] = player
    for point in update_list:
        chessboard[point] = player
    return np.array(update_list, dtype=int)


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


@numba.njit(cache=True)
def evaluate_chessboard(chessboard, color, stability_weight, mobility_weight, frontier_weight, chessboard_weight):
    step = np.sum(chessboard != COLOR_NONE) - 4
    my_actions = generate_actions_filter(chessboard, color)
    opponent_actions = generate_actions_filter(chessboard, -color)
    if (not my_actions) and (not opponent_actions):
        advantage = np.sum(chessboard) * (-color)
        return 1e+5 if advantage > 0 else -1e+5 if advantage < 0 else 5e+4

    if color == COLOR_BLACK:
        my_boundary, opponent_boundary = find_boundaries(chessboard)
        my_sentinels, opponent_sentinels = find_sentinels(chessboard)
    else:
        opponent_boundary, my_boundary = find_boundaries(chessboard)
        my_sentinels, opponent_sentinels = find_sentinels(chessboard)

    stability_score = stability_weight * (opponent_boundary - my_boundary)

    if step < 30:
        mobility_score = len(my_actions) * mobility_weight[0] - len(opponent_actions) * mobility_weight[1]
        frontier_score = frontier_weight[0] * (opponent_sentinels - my_sentinels)
    else:
        mobility_score = len(my_actions) * mobility_weight[2] - len(opponent_actions) * mobility_weight[3]
        frontier_score = frontier_weight[1] * (opponent_sentinels - my_sentinels)

    positional_score = np.sum(np.multiply(chessboard, chessboard_weight)) * (-color)

    return positional_score + stability_score + mobility_score + frontier_score
