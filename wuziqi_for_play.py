from wuziqi import *
# from wuziqi_NN import GamePlayer_MCTS_NN, model,SAVE_PATH
import pickle

SAVEPATH='game1'
wuziBoard = WuziBoard()


def setplayer():
    p1 = eval(input('请选择黑棋（0：真人，1：电脑）：'))
    if p1 == 0:
        player1 = GamePlayer_Real(PotColor(1))
    else:
        d1 = eval(input('请选择电脑难度 （0：白痴，1：小白，2：菜鸟，3：萌新，4：入门，5：神经网络）：'))
        if d1 == 0:
            player1 = GamePlayer_MCTS(maxn=10, potColor=PotColor(1))
        elif d1 == 1:
            player1 = GamePlayer_MCTS(maxn=100, potColor=PotColor(1))
        elif d1 == 2:
            player1 = GamePlayer_MCTS(maxn=500, potColor=PotColor(1))
        elif d1 == 3:
            player1 = GamePlayer_MCTS(maxn=1000, potColor=PotColor(1))
        elif d1 == 4:
            player1 = GamePlayer_MCTS(maxn=2000, potColor=PotColor(1))
        elif d1 == 5:
            player1 = GamePlayer_MCTS_NN(potColor=PotColor(1),model=model(SAVE_PATH))

    p2 = eval(input('请选择白棋（0：真人，1：电脑）：'))
    if p2 == 0:
        player2 = GamePlayer_Real(PotColor(2))
    else:
        d2 = eval(input('请选择电脑难度，（0：白痴，1：小白，2：菜鸟，3：萌新，4：入门，5：神经网络）:'))
        if d2 == 0:
            player2 = GamePlayer_MCTS(maxn=10, potColor=PotColor(2))
        elif d2 == 1:
            player2 = GamePlayer_MCTS(maxn=100, potColor=PotColor(2))
        elif d2 == 2:
            player2 = GamePlayer_MCTS(maxn=500, potColor=PotColor(2))
        elif d2 == 3:
            player2 = GamePlayer_MCTS(maxn=1000, potColor=PotColor(2))
        elif d2 == 4:
            player2 = GamePlayer_MCTS(maxn=2000, potColor=PotColor(2))
        elif d2 == 5:
            player2 = GamePlayer_MCTS_NN(potColor=PotColor(2), model=model(SAVE_PATH))

    return player1, player2


def maingame(player1, player2):
    wuziBoard.drawBoard()
    game = GameFivePot()

    print('落点输入格式：横坐标，纵坐标')
    while True:

        r = player1.myplay(game)
        wuziBoard.drawAction(game.getActionHis()[-1])
        if r[1]:
            if r[2]:
                print('黑棋获胜')
                save_game(game,SAVEPATH)
                break
            else:
                print('和棋')
                save_game(game,SAVEPATH)
                break

        r = player2.myplay(game)
        wuziBoard.drawAction(game.getActionHis()[-1])
        if r[1]:
            if r[2]:
                print('白棋获胜')
                save_game(game,SAVEPATH)
                break
            else:
                print('和棋')
                save_game(game,SAVEPATH)
                break


def playgame(p1=None, p2=None):
    if not p1 or not p2:
        p1, p2 = setplayer()

    while True:

        maingame(p1, p2)
        a = eval(input('0：退出；1：重新开始：'))
        if a == 0:
            quit()
        else:
            turtle.clear()
            b = eval(input('0:维持玩家设置；1：重置玩家：'))
            if b == 1:
                p1, p2 = setplayer()

def save_game(game,path):
    with open(path,'wb') as f:
        pickle.dump(game,f)

if __name__ == "__main__":
    playgame()
    # mymodel = model('mymodel')
    # p1 = GamePlayer_MCTS_NN(potColor=PotColor(1), maxn=10, model=mymodel)
    # p2 = GamePlayer_MCTS_NN(potColor=PotColor(2), maxn=10, model=mymodel)
    # playgame(p1, p2)
