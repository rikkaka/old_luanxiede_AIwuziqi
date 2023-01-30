from wuziqi import *
import pickle

path='game2'

with open(path,'rb') as f:
    game=pickle.load(f)

wuziBoard = WuziBoard()
wuziBoard.drawBoard()
wuziBoard.drawNow(game.ActionHis)