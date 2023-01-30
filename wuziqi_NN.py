from wuziqi import *

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model
import operator

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

RESNET_NUM = 2
C_UCT_NN = 5
MAXN_NN = 10
TEMP = 1  # (0,1]
SBN = True
SBN_COUNT = 5
KSH = False
BATCH_SIZE = 1024
EPOCHS = 5
LR = 5e-4
WP = 0.55
DRAW_SCORE = 0
WIN_SCORE = 1
LOST_SCORE = -1
SAVE_PATH = 'newmodel_resnet=2'

DATASET = []

if KSH:
    wuziBoard = WuziBoard()


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def select_by_N(node=Node(), temp=TEMP):
    nodes = node.children
    nodes_N = list(map(lambda x: x.N, nodes))
    probs = softmax(1.0 / temp * np.log(np.array(nodes_N) + 1e-10))
    child = np.random.choice(
        nodes,
        p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
    )
    return child


class model(object):

    def __init__(self, model_path=None):
        self.train_times = 0
        self.l2_const = 1e-4
        self.learning_rate = LR
        if model_path:
            self.model = load_model(model_path)
        else:
            self.init_model()

    def init_model(self, input_shape=(ROW_NUM, COL_NUM, 9)):
    # def init_model(self, input_shape=(ROW_NUM, COL_NUM, 3)):

        X_input = Input(input_shape)

        X = Conv2D(filters=64, kernel_size=[3, 3], padding='same', kernel_regularizer=l2(self.l2_const))(X_input)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        for _ in range(RESNET_NUM):
            X = self.identify_block(X)

        policy = Conv2D(filters=2, kernel_size=[1, 1], padding='same', kernel_regularizer=l2(self.l2_const))(X)
        policy = BatchNormalization()(policy)
        policy = Activation('relu')(policy)
        policy = Flatten()(policy)
        policy = Dense(ROW_NUM * ROW_NUM, activation='softmax', name='policy', kernel_regularizer=l2(self.l2_const))(
            policy)

        value = Conv2D(filters=1, kernel_size=[1, 1], padding='same', kernel_regularizer=l2(self.l2_const))(X)
        value = BatchNormalization()(value)
        value = Activation('relu')(value)
        value = Flatten()(value)
        value = Dense(64, activation='relu', kernel_regularizer=l2(self.l2_const))(value)
        value = Dense(1, activation='tanh', name='value', kernel_regularizer=l2(self.l2_const))(value)

        model = Model(inputs=X_input, outputs=[policy, value])

        Adam = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=Adam,
                      loss={"policy": "categorical_crossentropy",
                            "value": "mse"},
                      loss_weights={"policy": 1,
                                    "value": 1}
                      )

        self.model = model

    def identify_block(self, X, f=3, filter_num=64):

        X_shortcut = X

        X = Conv2D(filters=filter_num, kernel_size=[f, f], padding='same', kernel_regularizer=l2(self.l2_const))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        X = Conv2D(filters=filter_num, kernel_size=[f, f], padding='same', kernel_regularizer=l2(self.l2_const))(X)
        X = BatchNormalization()(X)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def train(self, dataset):

        X = []
        Y1 = []
        Y2 = []
        for runaction, nextcolor, site, value in dataset:
            x = self.getX(runaction, nextcolor)
            y1 = self.getPolicy(site)
            y1 = y1.flatten()
            y2 = [value]
            X.append(x[0])
            Y1.append(y1)
            Y2.append(y2)
        X = np.array(X)
        Y1 = np.array(Y1)
        Y2 = np.array(Y2)

        self.train_times += 1
        Adam = optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=Adam,
                           loss={"policy": "categorical_crossentropy",
                                 "value": "mse"},
                           loss_weights={"policy": 1,
                                         "value": 1}
                           )
        self.model.fit(x=X, y=[Y1, Y2], verbose=2, epochs=EPOCHS, batch_size=len(dataset))

    def save(self, path):
        self.model.save(path)

    def get_situation(self, runaction):

        black_situ = np.zeros(shape=(ROW_NUM, COL_NUM))
        white_situ = np.zeros(shape=(ROW_NUM, COL_NUM))

        for x in range(ROW_NUM):
            for y in range(COL_NUM):
                if runaction[x][y] == PotColor.Black:
                    black_situ[x, y] = 1
                elif runaction[x][y] == PotColor.White:
                    white_situ[x, y] = 1

        return black_situ, white_situ

    def getX(self, runaction, nextcolor):

        black_situ, white_situ = self.get_situation(runaction)

        if nextcolor == PotColor.Black:
            colorchannel = np.zeros(shape=(ROW_NUM, COL_NUM))
        else:
            colorchannel = np.ones(shape=(ROW_NUM, COL_NUM))

        return np.array([np.stack([black_situ, black_situ, black_situ, black_situ,
                                   white_situ, white_situ, white_situ, white_situ, colorchannel], axis=2)])
        # return np.array([np.stack([black_situ, white_situ, colorchannel], axis=2)])

    def getPolicy(self, site):

        policy = np.zeros(shape=(ROW_NUM, COL_NUM))
        policy[site] = 1

        return policy

    def getOutput(self, node=Node()):

        runaction = node.game.RunAction
        nextcolor = node.nextcolor
        x = self.getX(runaction, nextcolor)
        policy, value = self.model.predict(x)
        policy.resize([ROW_NUM, COL_NUM])

        return policy, value


class GamePlayer_MCTS_NN(GamePlayer_MCTS):

    def __init__(self, maxn=MAXN_NN, potColor=PotColor.Black, model=model()):
        super().__init__(maxn, potColor)
        self.model = model

    def choiceActions_MCTS(self, game=GameFivePot(), first_node=None):

        if game.potCount == 0:
            return ROW_NUM // 2, COL_NUM // 2

        overpot = self.over_pot(game)
        if overpot:
            return overpot
        game = deepcopy(game)

        if not first_node:
            first_node = Node(self.nextcolor, game)

        i = 0
        while True:
            # t=time.time()

            c, selected = self.Select_Expand(first_node)
            if c == 2:
                self.Backpropagate(selected.score, selected)
            else:
                endnode = selected
                score = self.Getscore(endnode)
                self.Backpropagate(score, endnode)

            i += 1
            # print(time.time()-t)
            if i % self.maxn == 0:
                max_N = 0
                max_N_node = None
                for node in first_node.children:
                    if node.N > max_N:
                        max_N, max_N_node = node.N, node
                if max_N >= self.maxn:
                    return max_N_node.site

    def Select_Expand(self, node=Node()):
        if node.condition == 2:
            return 2, node

        end = False
        if not node.children:
            end = True

        while node.condition == 0:
            c, child = node.add_child()
            if c == 2:
                return 2, child

        UTCs = list(map(lambda x: self.UCT(node.N, x), node.children))
        selected = node.children[np.argmax(UTCs)]

        if not selected.value:
            policy, value = self.model.getOutput(selected)
            selected.value = value[0, 0]
            selected.policy = policy

        if end:
            return 0, selected

        return self.Select_Expand(selected)

    def Getscore(self, node=Node()):
        return {node.nextcolor: node.value, node.thiscolor: WIN_SCORE + LOST_SCORE - node.value}

    def Backpropagate(self, score, node=Node()):
        node.N += 1
        node.Q += score[node.thiscolor]

        if node.parent:
            self.Backpropagate(score, node.parent)

    def UCT(self, Nsum=1, node=Node(), c=C_UCT_NN):
        if node.N != 0:
            return node.Q / node.N + c * node.P * np.sqrt(Nsum) / (node.N + 1)
            # return node.Q / node.N + c * 1 * np.sqrt(Nsum) / (node.N + 1)
        else:
            return c * node.P * np.sqrt(Nsum) / (node.N + 1)
            # return c * 1 * np.sqrt(Nsum) / (node.N + 1)

    def myplay(self, game, sbn=False):

        first_node = Node(self.nextcolor, game)
        first_node.policy = self.model.getOutput(first_node)[0]
        action = self.choiceActions_MCTS(game, first_node)

        if sbn and first_node.children and game.potCount <= SBN_COUNT - 1:
            c = select_by_N(first_node)
            action = c.site

        runaction, isOver, isWin = game.action(action, self.nextcolor)

        return action, isOver, isWin


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
        self.value = None
        self.site = None
        self.score = None
        self.layer = layer
        self.policy = None
        self.P = None
        self.action_P = None
        self.condition = 0  # 0:unexpanded; 1:expanded; 2:over;

    def add_child(self):
        if self.condition != 0:
            return 1, 0

        nowgame = deepcopy(self.game)
        if self.action_P is None:
            a = [[action, self.policy[action]] for action in self.action]
            self.action_P = sorted(a, key=operator.itemgetter(1), reverse=True)
        action = self.action_P.pop(0)[0]
        if not self.action_P:
            self.condition = 1

        result = nowgame.action(action, self.nextcolor)
        child = Node(self.thiscolor, nowgame, parent=self, layer=self.layer + 1)
        child.site = action
        child.P = self.policy[child.site]

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


def get_value_in(dataset, winner=None):
    for data in dataset:
        if not winner:
            data[3] += DRAW_SCORE
        elif winner == data[1]:
            data[3] += WIN_SCORE
        else:
            data[3] += LOST_SCORE


def self_play(themodel=model(), sbn=False, ksh=False, always_save=True, p1=None, p2=None, train=True):
    global DATASET
    c = 0

    if ksh:
        wuziBoard.drawBoard()
    game = GameFivePot()
    if p1 is None:
        p1 = GamePlayer_MCTS_NN(potColor=PotColor(1), model=themodel)
    if p2 is None:
        p2 = GamePlayer_MCTS_NN(potColor=PotColor(2), model=themodel)

    dataset = []
    while True:

        runaction = deepcopy(game.RunAction)
        # t=time.time()
        action, isOver, isWin = p1.myplay(game, sbn)
        # print(time.time() - t)
        dataset.append([runaction, p1.nextcolor, action, 0])
        if ksh:
            wuziBoard.drawAction(game.getActionHis()[-1])
        if isOver:
            if isWin:
                get_value_in(dataset, p1.nextcolor)
                if train:
                    DATASET += dataset
                winner = p1
                break
            else:
                get_value_in(dataset)
                if train:
                    DATASET += dataset
                winner = None
                break

        runaction = deepcopy(game.RunAction)
        action, isOver, isWin = p2.myplay(game, sbn)
        dataset.append([runaction, p2.nextcolor, action, 0])
        if ksh:
            wuziBoard.drawAction(game.getActionHis()[-1])
        if isOver:
            if isWin:
                get_value_in(dataset, p2.nextcolor)
                if train:
                    DATASET += dataset
                winner = p2
                break
            else:
                get_value_in(dataset)
                if train:
                    DATASET += dataset
                winner = None
                break

    if len(DATASET) >= BATCH_SIZE and train:
        themodel.train(DATASET)
        DATASET = []
        c = 1
        if always_save:
            themodel.save(SAVE_PATH + '_new')
            # themodel.save(SAVE_PATH)

    if ksh:
        turtle.clear()

    return game, c, winner, len(DATASET)


def model_pk(model_old, model_new, wp=WP, half_times=20,renew=True):
    print('开始新旧模型对战')
    q = 0

    for i in range(half_times):
        print('第%d局游戏'%(i+1))
        p1 = GamePlayer_MCTS_NN(potColor=PotColor(1), model=model_old)
        p2 = GamePlayer_MCTS_NN(potColor=PotColor(2), model=model_new)
        _, _, winner,_ = self_play(sbn=SBN, always_save=False, p1=p1, p2=p2, train=False)
        if winner is p2:
            print('新模型胜')
            q += 1
        elif winner is p1:
            print('旧模型胜')
        else:
            print('和棋')
            q += 0.5

    for i in range(half_times):
        print('第%d局游戏'%(i+half_times+1))
        p1 = GamePlayer_MCTS_NN(potColor=PotColor(1), model=model_new)
        p2 = GamePlayer_MCTS_NN(potColor=PotColor(2), model=model_old)
        _, _, winner,_ = self_play(sbn=True, always_save=False, p1=p1, p2=p2, train=False)
        if winner is p1:
            print('新模型胜')
            q += 1
        elif winner is p2:
            print('旧模型胜')
        else:
            print('和棋')
            q += 0.5

    print('新模型胜率:%.4f' % (q / (half_times * 2)))
    if renew:
        if q / (half_times * 2) >= wp:
            model_new.save(SAVE_PATH)
            print('模型更新')
        else:
            print('模型未更新')

