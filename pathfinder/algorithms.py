import numpy as np
import math


def evristick(map, start, end, evr):
    width = len(map[0])
    x_s, y_s = start % width, start // width
    x_e, y_e = end % width, end // width

    if evr == 0:  # Manhattan
        return math.fabs(x_e - x_s) + math.fabs(y_e - y_s)
    elif evr == 1:  # Chebyshev
        return max(math.fabs(x_e - x_s), math.fabs(y_e - y_s))
    else:  # Euclidean
        return math.sqrt((x_e - x_s) ** 2 + (y_e - y_s) ** 2)


def evristick_line(map, line, end, evr):
    width = len(map[0])
    new_line = np.zeros(len(line))
    for i in range(len(line)):
        new_line[i] = evristick(map, i, end, evr)
    return new_line


def calculate_weight(speed):
    return 1 / speed


def table_of_map(map, speed, evr):
    height, width = map.shape
    weights = np.ones((height * width, height * width)) * np.inf
    for i in range(height):
        for j in range(width):
            if map[i][j] != 1:  # Not an obstacle
                current_index = width * i + j
                weights[current_index][current_index] = 0
                resistance_coef = calculate_weight(speed) if map[i][j] not in [0, 1] else 1
                # Straight moves
                if j != width - 1 and map[i][j + 1] != 1:
                    weights[current_index][current_index + 1] = resistance_coef if map[i][j + 1] not in [0, 1] else 1
                if j != 0 and map[i][j - 1] != 1:
                    weights[current_index][current_index - 1] = resistance_coef if map[i][j - 1] not in [0, 1] else 1
                if i != 0 and map[i - 1][j] != 1:
                    weights[current_index][current_index - width] = resistance_coef if map[i - 1][j] not in [0,
                                                                                                             1] else 1
                if i != height - 1 and map[i + 1][j] != 1:
                    weights[current_index][current_index + width] = resistance_coef if map[i + 1][j] not in [0,
                                                                                                             1] else 1
                # Diagonal moves
                if evr in [1, 2]:
                    diag_weight = 1.41 if evr == 2 else 1
                    if i != height - 1 and j != width - 1 and map[i + 1][j + 1] != 1:
                        weights[current_index][current_index + width + 1] = (
                            diag_weight * resistance_coef if map[i + 1][j + 1] not in [0, 1] else diag_weight)
                    if i != 0 and j != width - 1 and map[i - 1][j + 1] != 1:
                        weights[current_index][current_index - width + 1] = (
                            diag_weight * resistance_coef if map[i - 1][j + 1] not in [0, 1] else diag_weight)
                    if i != height - 1 and j != 0 and map[i + 1][j - 1] != 1:
                        weights[current_index][current_index + width - 1] = (
                            diag_weight * resistance_coef if map[i + 1][j - 1] not in [0, 1] else diag_weight)
                    if i != 0 and j != 0 and map[i - 1][j - 1] != 1:
                        weights[current_index][current_index - width - 1] = (
                            diag_weight * resistance_coef if map[i - 1][j - 1] not in [0, 1] else diag_weight)
    return weights


def find_new_index(map, last_line, end, evr, heuristic_evaluation):
    line = np.copy(last_line)
    line[line == 0] = np.inf
    line_with_evr = line + heuristic_evaluation
    line_0 = np.sort(line_with_evr)
    for x in line_0:
        index = np.where(line_with_evr == x)[0][0]
        return index


def iteration_of_dijkstra(index, last_line, table):
    new_line = np.copy(last_line)
    new_line[index] = 0
    last_step = last_line[index]

    for i in range(len(new_line)):
        if table[index, i] + last_step < new_line[i]:
            new_line[i] = table[index, i] + last_step

    return new_line


def A_star(map, table, start, end, evr):
    width = len(map[0])
    end_index = end[0] + end[1] * width
    heuristic_evaluation = evristick_line(map, table[0], end_index, evr)
    line = np.ones(len(table)) * np.inf
    index = start
    line[start] = 0

    table_dijkstra = np.zeros([len(table) - 1, len(table)])

    for i in range(len(line) - 1):
        new_line = iteration_of_dijkstra(index, line, table)
        table_dijkstra[i] = new_line
        if index == end_index:
            break
        index = find_new_index(map, new_line, end_index, evr, heuristic_evaluation)
        line = new_line
    return table_dijkstra


def calculate_total_weight(map, way, speed):
    height, width = map.shape
    total_weight = 0
    for i in range(len(way) - 1):
        current_index = way[i]
        next_index = way[i + 1]
        current_x, current_y = current_index % width, current_index // width
        next_x, next_y = next_index % width, next_index // width
        if abs(current_x - next_x) + abs(current_y - next_y) == 1:  # Straight
            total_weight += calculate_weight(speed) if map[next_y][next_x] not in [0, 1] else 1
        else:  # Diagonal
            total_weight += 1.41 * (calculate_weight(speed) if map[next_y][next_x] not in [0, 1] else 1)
    return total_weight


def find_new_ceng_from_zero(line, last_line):
    index_0 = np.where(line == 0)
    for x in index_0[0]:
        if last_line[x] != line[x]:
            return x
    return None


def find_way(table, start, end):
    index = end
    for i in range(len(table)):
        j = len(table) - i - 1
        if table[j][index] != 0:
            break

    way = [end]
    length = table[j][index]

    for i in range(j + 1):
        k = j - i
        if table[k][index] != length:
            index = find_new_ceng_from_zero(table[k + 1], table[k])
            length = table[k][index]
            way.append(index)
    way.append(start)
    return way


def A_star_final(map, start_coord, end_coord, speed, evr):
    table_1 = table_of_map(map, speed, evr)
    start_x, start_y = start_coord
    end_x, end_y = end_coord
    start = start_x + start_y * len(map[0])
    end = end_x + end_y * len(map[0])
    table_2 = A_star(map, table_1, start, end_coord, evr)
    way = find_way(table_2, start, end)
    total_weight = calculate_total_weight(map, way, speed)

    # Mark path on map
    new_map = np.copy(map)
    for z in way:
        x = z % len(map[0])
        y = z // len(map[0])
        new_map[y, x] = 2  # path marker

    return new_map, total_weight