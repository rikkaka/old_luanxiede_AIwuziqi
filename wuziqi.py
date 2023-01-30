# -*- coding:utf-8 -*-

import turtle
import numpy as np
from enum import Enum
import random
import time
from copy import deepcopy
from numba import jit

ROW_NUM = 9
COL_NUM = 9
GOS = 5
C_UCT = 1
DRAW_SCORE = 0.5
WIN_SCORE = 1
LOST_SCORE = 0

MCTS_NUMS = None


class PotColor(Enum):
    Black = 1
    White = 2


def getOpponentColor(color):
    if color == PotColor.Black:
        return PotColor.White
    else:
        return PotColor.Black


class WuziBoard(object):
    def __init__(self, RowNum=ROW_NUM, ColNum=COL_NUM):
        turtle.speed(9)
        turtle.hideturtle()
        self.RowNum = RowNum
        self.ColNum = ColNum
        self.halfDim = 500 / (RowNum - 1) / 2.0
        pass

    def drawBoard(self, ActionHis=None):
        turtle.screensize(400, 400, "Brown")
        turtle.title('五子棋')
        turtle.home()
        turtle.speed(0)
        time.sleep(0.1)

        for i in range(self.ColNum):
            x = 0 - 250 + i * self.halfDim * 2
            y = 0 - 250
            turtle.penup()
            turtle.setpos(x, y)
            turtle.pendown()
            turtle.goto(x, y + 500)
            turtle.penup()
            turtle.setpos(x, y + 520)
            turtle.pendown()
            turtle.write(i + 1, align='center')

        for i in range(self.RowNum):
            x = 0 - 250
            y = 0 - 250 + i * self.halfDim * 2
            turtle.penup()
            turtle.setpos(x, y)
            turtle.pendown()
            turtle.setpos(x + 500, y)
            turtle.penup()
            turtle.setpos(x + 520, y)
            turtle.pendown()
            turtle.write(i + 1, align='center')

        if ActionHis is not None:
            self.drawNow(ActionHis)

        # turtle.done()

        pass

    def action2potxy(self, action):
        x = 0 - 250 + action[0] * self.halfDim * 2
        y = 0 - 250 + action[1] * self.halfDim * 2
        return x, y

    def drawNow(self, RunAction):

        for potsite in RunAction:

            x, y = self.action2potxy((potsite[0], potsite[1]))
            turtle.penup()
            turtle.setpos(x, y)
            turtle.pendown()
            if potsite[2] != PotColor.White:
                turtle.dot(20, "Black")
            else:
                turtle.dot(20, "White")

            if potsite == RunAction[len(RunAction) - 1]:
                if potsite[2] != PotColor.White:
                    turtle.dot(35, "Black")
                else:
                    turtle.dot(35, "White")
            input()

    def drawAction(self, potsite):
        x, y = self.action2potxy((potsite[0], potsite[1]))
        turtle.penup()
        turtle.setpos(x, y)
        turtle.pendown()
        if potsite[2] == PotColor.Black:
            turtle.dot(20, "Black")
        else:
            turtle.dot(20, "White")
        turtle.penup()

    pass


class GameFivePot(object):

    def __init__(self, AllAction=None, RunAction=None):
        self.potCount = 0
        self.AllAction = []
        self.ActionHis = []
        self.winpot = []

        if AllAction:
            self.AllAction = AllAction
        else:
            for x in range(ROW_NUM):
                for y in range(COL_NUM):
                    self.AllAction += [(x, y)]

        self.AvailAction = self.AllAction
        if RunAction:
            self.RunAction = RunAction
        else:
            self.RunAction = [[0 for col in range(ROW_NUM)] for row in range(COL_NUM)]

    def getActions(self, reasonable=True):
        if not reasonable:
            return self.AvailAction

        if self.potCount == 0:
            return [(ROW_NUM // 2, COL_NUM // 2)]

        AvailAction=self.AvailAction
        RunAction=self.RunAction

        # @jit(nopython=False)
        def t():
            taken_actions = []
            for x, y in AvailAction:
                if x - 1 > -1:
                    if y - 1 > -1:
                        if RunAction[x - 1][y - 1]:
                            taken_actions.append((x, y))
                            continue
                    if y + 1 < COL_NUM:
                        if RunAction[x - 1][y + 1]:
                            taken_actions.append((x, y))
                            continue
                    if RunAction[x - 1][y]:
                        taken_actions.append((x, y))
                        continue
                if x + 1 < ROW_NUM:
                    if y - 1 > -1:
                        if RunAction[x + 1][y - 1]:
                            taken_actions.append((x, y))
                            continue
                    if y + 1 < COL_NUM:
                        if RunAction[x + 1][y + 1]:
                            taken_actions.append((x, y))
                            continue
                    if RunAction[x + 1][y]:
                        taken_actions.append((x, y))
                        continue
                if y - 1 > -1:
                    if RunAction[x][y - 1]:
                        taken_actions.append((x, y))
                        continue
                if y + 1 < COL_NUM:
                    if RunAction[x][y + 1]:
                        taken_actions.append((x, y))
                        continue
            return taken_actions

        return t()

    def getRunAction(self):
        return self.RunAction

    def getActionHis(self):
        return self.ActionHis

    def is_over(self, action, potColor):
        x = action[0]
        y = action[1]
        dimCount = [1, 1, 1, 1]

        # ���� xiang qian
        last = []
        for x1 in range(x + 1, x + 5):

            if (x1 >= ROW_NUM):
                break
            if (self.RunAction[x1][y] == potColor):
                dimCount[0] += 1
            else:
                if not self.RunAction[x1][y]:
                    last.append((x1, y))
                break

        # - xiang hou
        for x1 in range(x - 1, x - 5, -1):
            if (x1 < 0):
                break

            if (self.RunAction[x1][y] == potColor):
                dimCount[0] += 1
            else:
                if not self.RunAction[x1][y]:
                    last.append((x1, y))
                break

        if (dimCount[0] >= GOS):
            return True, True
        if dimCount[0] == GOS - 1:
            self.winpot += last

        # ���� ����
        last = []
        for y1 in range(y + 1, y + 5):
            if (y1 >= ROW_NUM):
                break

            if (self.RunAction[x][y1] == potColor):
                dimCount[1] += 1
            else:
                if not self.RunAction[x][y1]:
                    last.append((x, y1))
                break

        # - ����
        for y1 in range(y - 1, y - 5, -1):
            if (y1 < 0):
                break

            if (self.RunAction[x][y1] == potColor):
                dimCount[1] += 1
            else:
                if not self.RunAction[x][y1]:
                    last.append((x, y1))
                break

        if (dimCount[1] >= GOS):
            return True, True
        if dimCount[1] == GOS - 1:
            self.winpot += last

        # -��б ����
        last = []
        for offset in range(1, 5):
            x1 = x + offset
            y1 = y + offset

            if (y1 >= ROW_NUM or x1 >= ROW_NUM):
                break

            if (self.RunAction[x1][y1] == potColor):
                dimCount[2] += 1
            else:
                if not self.RunAction[x1][y1]:
                    last.append((x1, y1))
                break

        # - ����
        for offset in range(-1, -5, -1):
            x1 = x + offset
            y1 = y + offset
            if (y1 < 0 or x1 < 0):
                break

            if (self.RunAction[x1][y1] == potColor):
                dimCount[2] += 1
            else:
                if not self.RunAction[x1][y1]:
                    last.append((x1, y1))
                break

        if (dimCount[2] >= GOS):
            return True, True
        if dimCount[2] == GOS - 1:
            self.winpot += last

        # -��б ����
        last = []
        for offset in range(1, 5):
            x1 = x + offset
            y1 = y - offset

            if y1 < 0 or x1 >= ROW_NUM:
                break

            if self.RunAction[x1][y1] == potColor:
                dimCount[3] += 1
            else:
                if not self.RunAction[x1][y1]:
                    last.append((x1, y1))
                break

        # - ����
        for offset in range(-1, -5, -1):
            x1 = x + offset
            y1 = y - offset
            if (y1 >= ROW_NUM or x1 < 0):
                break

            if (self.RunAction[x1][y1] == potColor):
                dimCount[3] += 1
            else:
                if not self.RunAction[x1][y1]:
                    last.append((x1, y1))
                break

        if (dimCount[3] >= GOS):
            return True, True
        if dimCount[3] == GOS - 1:
            self.winpot += last

        if (len(self.AvailAction) == 0):
            return True, False

        return False, False

    def action(self, action, potColor):
        while action in self.winpot:
            self.winpot.remove(action)
        self.potCount += 1
        self.ActionHis += [(action[0], action[1], potColor)]
        self.AllAction.remove(action)
        self.RunAction[action[0]][action[1]] = potColor

        isOver, isWin = self.is_over(action, potColor)
        return self.RunAction, isOver, isWin

    def __repr__(self):
        return "Game step count: {}, AvailAction len: {},  ".format(self.potCount, len(self.AvailAction))


class GamePlayer(object):

    def __init__(self, potColor):
        self.actionHis = []
        self.nextcolor = potColor
        self.thiscolor = getOpponentColor(potColor)
        self.opponent_color = PotColor.White if potColor == PotColor.Black else PotColor.Black

    def getActionHis(self):
        return self.actionHis

    def play(self, game):
        for pot in game.winpot:
            if pot not in game.AllAction:
                print(1)
        actions = game.getActions()
        action = self.choiceActions(actions, game)
        self.actionHis = self.actionHis + [action]

        gameInfo, isOver, isWin = game.action(action, self.nextcolor)

        return gameInfo, isOver, isWin

        pass

    def over_pot(self, game=GameFivePot()):
        if game.winpot:
            return game.winpot[0]

    def choiceActions(self, actions, game):
        # if game.winpot:
        #     return game.winpot[0]
        action = random.choice(actions)
        return action

    def __repr__(self):
        return "color: {}, actionHis: {},  ".format(self.nextcolor, self.actionHis)


class Node(object):

    def __init__(self, nextcolor=PotColor(1), game=GameFivePot(), parent=None, layer=0):
        self.nextcolor = nextcolor
        self.thiscolor = getOpponentColor(nextcolor)
        self.game = deepcopy(game)
        self.parent = parent
        self.action = deepcopy(game.getActions())
        self.children = []
        self.son_number = 0
        self.Q = 0
        self.N = 0
        self.site = None
        self.score = None
        self.layer = layer
        self.condition = 0  # 0:unexpanded; 1:expanded; 2:over;

    def add_child(self):
        if self.condition != 0:
            return 1, 0

        nowgame = deepcopy(self.game)

        p = random.randint(0, len(self.action) - 1)
        action = self.action.pop(p)
        if not self.action:
            self.condition = 1

        result = nowgame.action(action, self.nextcolor)
        child = Node(self.thiscolor, nowgame, parent=self, layer=self.layer + 1)
        child.site = action

        self.children.append(child)

        if result[1]:
            child.condition = 2
            if result[2]:
                child.score = {child.thiscolor: WIN_SCORE, child.nextcolor: LOST_SCORE}
                self.score = {self.thiscolor: WIN_SCORE, self.nextcolor: LOST_SCORE}
                return 2, child
            else:
                child.score = {child.thiscolor: DRAW_SCORE, child.nextcolor: DRAW_SCORE}
                self.score = {self.thiscolor: DRAW_SCORE, self.nextcolor: DRAW_SCORE}
                return 2, child

        return 0, child


class GamePlayer_MCTS(GamePlayer):

    def __init__(self, maxn=500, potColor=PotColor.Black):
        super().__init__(potColor)
        self.maxn = maxn

    def choiceActions_MCTS(self, game=GameFivePot(), first_node=None):

        if game.potCount == 0:
            return (ROW_NUM // 2, COL_NUM // 2)

        overpot = self.over_pot(game)
        if overpot:
            return overpot
        game = deepcopy(game)

        if not first_node:
            first_node = Node(self.nextcolor, game)

        i = 0
        while True:

            c, selected = self.Select(first_node)
            if c == 2:
                self.Backpropagate(selected.score, selected)
            elif c == 0:
                c1, r = self.expand(selected)
                if c1 == 2:
                    endnode, score = r, r.score
                else:
                    endnode = r
                    score = self.Simulate(endnode)
            else:
                endnode = selected
                score = self.Simulate(endnode)

            if c != 2:
                self.Backpropagate(score, endnode)

            i += 1
            if i % self.maxn == 0:
                max_N = 0
                for node in first_node.children:
                    if node.N > max_N:
                        max_N = node.N
                if max_N >= self.maxn:
                    break

        max_N = 0
        max_N_node = 0
        for node in first_node.children:
            if node.N > max_N:
                max_N, max_N_node = node.N, node

        return max_N_node.site

    def Select(self, node=Node()):
        if node.condition == 2:
            return 2, node

        if node.condition == 0:
            return 0, node

        UTCs = list(map(lambda x: self.UCT(node.N, x), node.children))
        selected = node.children[np.argmax(UTCs)]
        return self.Select(selected)

    def expand(self, node=Node()):
        return node.add_child()

    def Simulate(self, node=Node()):
        game = deepcopy(node.game)
        player1 = GamePlayer(node.nextcolor)
        player2 = GamePlayer(getOpponentColor(node.nextcolor))

        while True:

            result = player1.play(game)
            if result[1]:
                if result[2]:
                    return {player1.thiscolor: WIN_SCORE, player2.thiscolor: LOST_SCORE}
                else:
                    return {player1.thiscolor: DRAW_SCORE, player2.thiscolor: DRAW_SCORE}

            result = player2.play(game)
            if result[1]:
                if result[2]:
                    return {player2.thiscolor: WIN_SCORE, player1.thiscolor: LOST_SCORE}
                else:
                    return {player1.thiscolor: DRAW_SCORE, player2.thiscolor: DRAW_SCORE}

    def Backpropagate(self, score, node=Node()):
        node.N += 1
        node.Q += score[node.nextcolor]

        if node.parent:
            self.Backpropagate(score, node.parent)

    def UCT(self, Nv=1, node=Node(), c=C_UCT):
        return node.Q / node.N + c * np.sqrt(np.log(Nv) / node.N)

    def play_MCTS(self, game):
        action = self.choiceActions_MCTS(game)
        self.actionHis = self.actionHis + [action]

        gameInfo, isOver, isWin = game.action(action, self.nextcolor)

        return action, isOver, isWin

    def myplay(self, game, sbn=False):
        return self.play_MCTS(game)


class GamePlayer_Real(GamePlayer):

    def play_real(self, game=GameFivePot()):
        actions = game.getActions(reasonable=False)

        while True:
            action = eval(input('请输入%s落点:' % ('黑棋' if self.nextcolor == PotColor.Black else '白棋')))
            action = (action[0] - 1, action[1] - 1)
            if action not in actions:
                print('请输入正确的落点!')
                continue
            else:
                break

        RunAction, isOver, isWin = game.action(action, self.nextcolor)

        return action, isOver, isWin

    def myplay(self, game, sbn=False):
        return self.play_real(game)
