import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# 如果不在 PYTHONPATH 中，则加入
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import constants
import chess_constants as cc
#import time
from pieces import listPiecestoArr
import my_game as mg
import my_chess as mc
def getPlayInfo(listpieces, from_x, from_y, to_x, to_y, mgInit):
    pieces = movedeep(listpieces ,1 ,constants.player2Color, from_x, from_y, to_x, to_y, mgInit)
    return [pieces[0].x, pieces[0].y, pieces[1], pieces[2]]

def movedeep(listpieces, deepstep, player, x1, y1, x2, y2, mgInit,is_piecesbest = True):

    # temp.board.print_board()
    # 专职
    s = mc.step(8 - x1, y1, 8 - x2, y2)
    mgInit.move_to(s)
    # temp.evaluate(True)
    mgInit.alpha_beta(cc.max_depth, cc.min_val, cc.max_val)
    t = mgInit.best_move
    print('输入的结果：', s)
    print('计算的结果：',t)
    if not is_piecesbest:
        return t

    mgInit.move_to(t)
    arr = listPiecestoArr(listpieces)
    listMoveEnabel = []
    for i in range(0, 9):
        for j in range(0, 10):
            for item in listpieces:
                if item.x == 8 - t.from_x and item.y == t.from_y:
                    listMoveEnabel.append([item, 8 - t.to_x, t.to_y])
    piecesbest = listMoveEnabel[0]
    return piecesbest


def execute_move(mgInit, move):
    """
    执行移动到游戏状态中

    Args:
        mgInit: 游戏初始化对象
        move: 要执行的移动
    """
    if move:
        mgInit.move_to(move)
        print(f"已执行移动: {move}")


