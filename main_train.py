from wuziqi_NN import *

if __name__ == "__main__":
    mymodel = model(model_path=SAVE_PATH)
    # mymodel = model()
    # print(mymodel.model.summary())
    episode = 0
    while True:
        episode += 1
        # game = self_play(mymodel, sbn=SBN, ksh=KSH, p1=GamePlayer_Real(potColor=PotColor(1)))
        # game, c, winner = self_play(mymodel, sbn=SBN, ksh=KSH, p1=GamePlayer_MCTS(potColor=PotColor(1), maxn=200),
        #                             p2=GamePlayer_MCTS(potColor=PotColor(2), maxn=260))
        game, c, winner,jindu = self_play(mymodel, sbn=SBN, ksh=KSH, train=True)
        # print(game.ActionHis)
        # episode += 1
        # print(episode)
        # game, c, winner = self_play(mymodel, sbn=SBN, ksh=KSH, p1=GamePlayer_MCTS(potColor=PotColor(1), maxn=1000))
        print('第%d次self play,游戏长度%d,模型更新进度%.3f' % (episode, len(game.ActionHis), jindu / BATCH_SIZE))
        if c:
            model_old = model(model_path=SAVE_PATH)
            model_new = model(model_path=SAVE_PATH + '_new')
            model_pk(model_old, model_new)
        # print(episode)
        # game, c, winner = self_play(mymodel, sbn=SBN, ksh=KSH, p2=GamePlayer_MCTS(potColor=PotColor(2), maxn=1000))
        # print(winner)
        # episode += 1
        # if c:
        #     model_old = model(model_path=SAVE_PATH)
        #     model_new = model(model_path=SAVE_PATH + '_new')
        #     model_pk(model_old, model_new)
