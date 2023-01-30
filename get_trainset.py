from wuziqi import *
import pickle

save_path='train_test_maxn=1000'


def get_value_in(dataset, winner=None):
    for data in dataset:
        if not winner:
            data[3] += DRAW_SCORE
        elif winner == data[1]:
            data[3] += WIN_SCORE
        else:
            data[3] += LOST_SCORE

def self_play(p1=None, p2=None):
    global DATASET

    # f = open(save_path, 'r')
    # DATASET = pickle.load(f)
    # f.close()

    game = GameFivePot()
    if p1 is None:
        p1 = GamePlayer_MCTS(potColor=PotColor(1), maxn=1000)
    if p2 is None:
        p2 = GamePlayer_MCTS(potColor=PotColor(2), maxn=1000)

    dataset = []
    while True:

        runaction = deepcopy(game.RunAction)
        action, isOver, isWin = p1.myplay(game, False)
        dataset.append([runaction, p1.nextcolor, action, 0])
        if isOver:
            if isWin:
                get_value_in(dataset, p1.nextcolor)
                return dataset
            else:
                get_value_in(dataset)
                return dataset

        runaction = deepcopy(game.RunAction)
        action, isOver, isWin = p2.myplay(game, False)
        dataset.append([runaction, p2.nextcolor, action, 0])
        if isOver:
            if isWin:
                get_value_in(dataset, p2.nextcolor)
                return dataset
            else:
                get_value_in(dataset)
                return dataset


if __name__ == '__main__':
    while True:
        dataset=self_play()
        f = open(save_path, 'wb+')
        DATASET = pickle.load(f)
        DATASET+=dataset
        pickle.dump(DATASET, f)
        f.close()
