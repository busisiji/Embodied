import os
import sys

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（cchess）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))

# 如果不在 PYTHONPATH 中，则加入
if project_root not in sys.path:
    print(f"[INFO] 正在加入 {project_root}...")
    sys.path.insert(0, project_root)

import argparse
import time
from collections import defaultdict

import pandas as pd
import os
from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen, Request
import csv
import re
import requests
import webbrowser
import socket
from jiema import jiema
from src.cchessAI.parameters import DATA_PATH

CHESS_RESULR = ['认负',"绝杀",'局时',"议和","判和","读秒","断线",'步时',"判负","困毙",'放弃','超时',"取消",'封局','未知']
CRAWL_NUM = 0
def login(session, username, password):
    """
    执行登录操作

    参数:
        session (requests.Session): 会话对象
        username (str): 用户名
        password (str): 密码

    返回:
        bool: 登录是否成功
    """
    login_url = "http://www.dpxq.com/hldcg/search/login.asp"

    # 准备登录数据
    login_data = {
        "username": username,
        "password": password,
        "action": "chk",  # 注意这里的值是 'chk'
        "owner": "",
        "id": "",
        "f": "",
        "isSave": "",
        "CookieOK": "1"  # 可选
    }

    # 发送POST请求进行登录
    try:
        response = session.post(login_url, data=login_data)

        # 检查登录是否成功（根据实际响应内容调整）
        if "登录成功" in response.text:
            print("[INFO] 登录成功")
            return True
        else:
            print("[ERROR] 登录失败")
            return False
    except Exception as e:
        print(f"[ERROR] 登录时发生错误：{e}")
        return False
def extract_movelist(html):
    """
    使用纯字符串操作提取走子序列

    参数:
        html: 包含棋谱的HTML字符串

    返回:
        str: 提取到的走子序列字符串
    """
    # 定义开始和结束标记的模式
    start_marker = "var DhtmlXQ_movelist = ''+'[DhtmlXQ_move'+'list]"
    end_marker = "[/DhtmlXQ_move'+'list]'"

    # 查找开始标记位置
    start_index = html.find(start_marker)
    if start_index == -1:
        # 尝试简化的开始标记
        simplified_start = "[DhtmlXQ_move'+'list]"
        start_index = html.find(simplified_start)
        if start_index == -1:
            return ""  # 未找到开始标记
        start_index += len(simplified_start)
    else:
        start_index += len(start_marker)

    # 查找结束标记位置
    end_index = html.find(end_marker, start_index)
    if end_index == -1:
        # 尝试简化的结束标记
        simplified_end = "[/DhtmlXQ_move'+'list]"
        end_index = html.find(simplified_end, start_index)
        if end_index == -1:
            return ""  # 未找到结束标记

    # 提取两个标记之间的内容
    move_sequence = html[start_index:end_index]

    # 清理可能的空白字符
    return move_sequence.strip()
def getOwnerPage( owner, page):
    page = str(page)
    url = "http://www.dpxq.com/hldcg/search/list.asp?owner=" + owner + "&page=" + page
    user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0"
    req = Request(url, headers={'User-Agent': user_agent})
    try:
        html = urlopen(req).read().decode('gb2312')
    except Exception:
        try:
            html = urlopen(req).read().decode('gbk')
        except Exception:
            html = urlopen(req).read().decode('GB18030')
        finally:
            return []
    soup = BeautifulSoup(html, features='lxml')
    boardData = soup.find_all('a', {"href": re.compile(r"javascript:view\(\'owner=(.*).*id=\d*\'\)")})
    url_list = []
    for bo in boardData:
        url_action = bo['href']
        url_result = re.findall(r'javascript:view\(\'owner=(.*)&id=(.*)\'', url_action)[0]
        url_list.append([url_result[0], url_result[1]])
    return url_list

def getAllIdsFromFirstPage(url_list,add=0):
    if not url_list:
        print("[ERROR] 第一页未找到有效的 ID 数据")
        return []

    first_id = url_list[0][1]
    last_id = url_list[-1][1]
    len_url = len(url_list) * add

    try:
        first_id_int = int(first_id)
        last_id_int = int(last_id)
    except ValueError:
        print("[ERROR] ID 不是数字")
        return []

    # 判断是递增还是递减
    step = 1 if last_id_int > first_id_int else -1

    # 生成 ID 列表（包含首尾）
    if step == 1:
        all_ids = [str(i) for i in range(first_id_int + len_url, last_id_int + len_url + 1)]
    else:
        all_ids = [str(i) for i in range(first_id_int - len_url, last_id_int - 1 - len_url, -1)]

    return [[url_list[0][0], ids] for ids in all_ids]


def getChessManual(session, url):  # 获取游戏棋谱，返回谁获胜，棋谱
    user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0"
    headers = {'User-Agent': user_agent}

    try:
        response = session.get(url, headers=headers)
        html = response.content.decode(response.apparent_encoding or 'gb2312')
    except Exception as e:
        print(f"[ERROR] 请求棋谱页面失败: {e}")
        return ("error", '')

    soup = BeautifulSoup(html, features='lxml')
    divs = soup.find_all('div', id='dhtmlxq_view')
    if not divs:
        print("[ERROR] 未找到棋盘数据")
        time.sleep(30)
        return ("error", '')

    boardData = str(divs[0])

    try:
        if 'owner=t' in url:
            a = extract_movelist(html)
        else:
            a_match = re.search(r"var DhtmlXQ_movelist.*?\[DhtmlXQ_movelist\](.*?)\[/DhtmlXQ_movelist\]", html)
            a = a_match.group(1) if a_match else None
        b_match = re.search(r"\[DhtmlXQ_binit\](.*?)\[/DhtmlXQ_binit\]", boardData)
        b = b_match.group(1) if b_match.group(1) else '8979695949392919097717866646260600102030405060708012720323436383'


        if not a:
            print("[ERROR] 本局不在vip会员搜索结果中，无法查看着法<br>已登录vip并状态正常？请刷新左侧页面再看")
            # time.sleep(60)
            return ("error", "")

        type_match = re.search(r"\[DhtmlXQ_type\](.*?)\[/DhtmlXQ_type\]", boardData)
        chess_type = type_match.group(1) if type_match else ''

        if chess_type != '全局':
            print("[ERROR] 棋局类型不是全局")
            return ("error", '')

        win_match = re.search(r"\[DhtmlXQ_result\](.*?)\[/DhtmlXQ_result\]", boardData)
        winType = f'"{win_match.group(1)}"' if win_match else '"未知结果"'

        endtype_match = re.search(r"\[DhtmlXQ_endtype\](.*?)\[/DhtmlXQ_endtype\]", boardData) # 结束方式
        chess_endtype = endtype_match.group(1) if endtype_match else ''
        # if chess_endtype  in ['认负','局时',"议和","判和","读秒","断线",'步时','放弃','超时',"取消",'封局','未知']:
        #     print("[ERROR] 棋局结果不是正常结果")
        #     return
        if chess_endtype not in ["绝杀","判负","困毙"]:
            return ("error", '')

        result_list = jiema().getMoveListString(a, b)
        nextType = 'r' if result_list[0][1] == 0 else 'b'
        fen = binitToFen(b) + ' ' + nextType

        return fen, chess_type, eval(winType), result_list

    except Exception as e:
        print(f"[ERROR] 解析棋谱失败: {e}")
        return ("error", '')

def binitToFen(binit):
    chess_board = [[" " for i in range(9)] for i in range(10)]
    chess_type = ['R', 'N', 'B', 'A', 'K', 'A', 'B', 'N', 'R', 'C', 'C', 'P', 'P', 'P', 'P', 'P']
    for i in range(32):
        x = int(binit[i * 2])
        y = int(binit[i * 2 + 1])
        if x == 9:
            continue
        if i <= 15:
            ct = chess_type[i]
        else:
            ct = str.lower(chess_type[i - 16])
        chess_board[y][x] = ct
    fen = ""
    for line in chess_board:
        black = 0
        for index, char in enumerate(line):
            if char == ' ':
                black += 1
            else:
                if black != 0:
                    fen += str(black)
                black = 0
                fen += char
            if index == 8:
                if char == ' ':
                    fen += str(black)
                fen += '/'
    return fen[:-1]

def is_game_id_exists(file_path, game_id):
    """
    检查指定的 game_id 是否已经存在于 CSV 文件中。

    参数:
        file_path (str): CSV 文件路径
        game_id (str): 要检查的 gameID

    返回:
        bool: 如果存在返回 True，否则返回 False
    """
    if not os.path.exists(file_path):
        return False  # 文件不存在，自然也不存在该 game_id

    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # 跳过表头
        for row in reader:
            if row[0] == game_id:  # 第一列是 gameID
                return True
    return False

def getOwnerList(owner, page_down=100):
    re_set = set()
    for i in range(page_down):
        t = getOwnerPage(owner, i)
        for uname, id_name in t:
            if id_name == '':
                continue
            re_set.add(id_name)
    return re_set

def save_file(re_list, bh, file_path, mode='append'):
    """
    保存棋谱到 CSV 文件。

    参数:
        re_list (list): 棋谱数据列表。
        bh (str): gameID（即棋局编号）。
        mode (str): 写入模式，'append' 追加，'overwrite' 覆盖。
    """
    global CRAWL_NUM
    if re_list[0] == 'error':
        return

    win_type = re_list[1]
    win_who = re_list[2]

    # 不要和棋或无效结果
    if win_who in ['和棋', '']:
        # print(f"[INFO] gameID {bh} 无效结果，跳过写入。")
        return

    if win_who == '红胜':
        win_who = 'red'
    elif win_who == '黑胜':
        win_who = 'black'

    # 如果是追加模式且 gameID 已存在，则跳过
    if mode == 'append' and is_game_id_exists(file_path, bh):
        print(f"[INFO] gameID {bh} 已存在，跳过写入。")
        return

    # 创建目录（如果不存在）
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    # 写入文件
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['gameID', 'turn', 'side', 'move', 'winner'])

        turn = 0
        for index, (move_str, side_flag) in enumerate(re_list[3], start=1):
            side = 'red' if side_flag == 0 else 'black'
            converted_move = (
                move_str.replace('j', '+')
                        .replace('Z', '.')
                        .replace('t', '-')
                        .replace('一', '1')
                        .replace('二', '2')
                        .replace('三', '3')
                        .replace('四', '4')
                        .replace('五', '5')
                        .replace('六', '6')
                        .replace('七', '7')
                        .replace('八', '8')
                        .replace('九', '9')
            )
            converted_move = converted_move[0].upper() + converted_move[1:] if side == 'red' else converted_move[0].lower() + converted_move[1:]
            if side == 'red':
                turn += 1
            writer.writerow([bh, str(turn), side, converted_move, win_who])
        print(f"{bh} 保存成功")
        CRAWL_NUM += 1
    return file_path,gameinfo_path

def merge_csv_files_recursively(directory, output_file='merged_output.csv', append_mode=False):
    """
    递归合并指定目录及其所有子目录下的CSV文件，并根据模式决定是否保留已有数据。

    参数:
        directory (str): 要搜索的根目录路径。
        output_file (str): 输出文件的路径，默认为 'merged_output.csv'。
        append_mode (bool): 是否启用追加写入模式。True 表示保留旧数据，False 表示覆盖写入。

    返回:
        None: 结果保存到指定的输出文件中。
    """
    existing_game_ids = set()
    all_data = []

    # 如果是追加模式，先读取输出文件中已有的 gameID
    if append_mode and os.path.exists(output_file):
        try:
            output_df = pd.read_csv(output_file)
            if 'gameID' in output_df.columns:
                existing_game_ids.update(set(output_df['gameID']))
                print(f"[INFO] 已加载 {len(existing_game_ids)} 个已有 gameID。")
            else:
                print(f"[WARNING] 输出文件 {output_file} 缺少 gameID 字段。")
        except Exception as e:
            print(f"[ERROR] 无法读取输出文件 {output_file}: {e}")

    # 遍历目录并过滤重复 gameID 的数据
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.csv'):
                file_path = os.path.join(root, filename)
                try:
                    df = pd.read_csv(file_path)

                    if 'gameID' not in df.columns:
                        print(f"[WARNING] {file_path} 缺少 gameID 字段，跳过。")
                        continue

                    # 过滤掉已存在的 gameID 数据
                    filtered_df = df[~df['gameID'].isin(existing_game_ids)]

                    # 更新已处理的 gameID 列表
                    new_ids = set(df['gameID'])
                    existing_game_ids.update(new_ids)

                    if not filtered_df.empty:
                        all_data.append(filtered_df)
                        print(f"[INFO] 已读取并过滤 {file_path}")
                    else:
                        print(f"[INFO] {file_path} 中的数据已全部存在，跳过。")

                except Exception as e:
                    print(f"[ERROR] 无法读取文件 {file_path}: {e}")

    # 合并数据并写入输出文件
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)

        # 根据模式选择写入方式
        if append_mode and os.path.exists(output_file):
            merged_df.to_csv(output_file, mode='a', header=False, index=False)
            print(f"[INFO] 所有文件已追加写入至 {output_file}")
        else:
            merged_df.to_csv(output_file, index=False)
            print(f"[INFO] 所有文件已覆盖写入至 {output_file}")
    else:
        print("[WARNING] 没有找到可以合并的CSV文件。")

def merge_multiple_csv_paths(path_list, output_file='merged_output.csv', append_mode=False):
    """
    合并多个指定路径下的CSV文件到一个输出文件中。

    参数:
        path_list (list): 要合并的文件路径列表。可以是目录或单独的CSV文件。
        output_file (str): 输出文件的路径，默认为 'merged_output.csv'。
        append_mode (bool): 是否启用追加写入模式。True 表示保留旧数据，False 表示覆盖写入。

    返回:
        None: 结果保存到指定的输出文件中。
    """
    existing_game_ids = set()
    all_data = []

    # 如果是追加模式，先读取输出文件中已有的 gameID
    if append_mode and os.path.exists(output_file):
        try:
            output_df = pd.read_csv(output_file)
            if 'gameID' in output_df.columns:
                existing_game_ids.update(set(output_df['gameID']))
                print(f"[INFO] 已加载 {len(existing_game_ids)} 个已有 gameID。")
            else:
                print(f"[WARNING] 输出文件 {output_file} 缺少 gameID 字段。")
        except Exception as e:
            print(f"[ERROR] 无法读取输出文件 {output_file}: {e}")

    # 遍历路径列表
    for path in path_list:
        if os.path.isdir(path):
            # 如果路径是目录，则递归遍历目录下的文件
            for root, dirs, files in os.walk(path):
                for filename in files:
                    if filename.endswith('.csv'):
                        file_path = os.path.join(root, filename)
                        process_csv_file(file_path, existing_game_ids, all_data)
        elif os.path.isfile(path) and path.endswith('.csv'):
            # 如果路径是单个 CSV 文件，直接处理
            process_csv_file(path, existing_game_ids, all_data)
        else:
            print(f"[WARNING] 路径无效或不是CSV文件：{path}")

    # 合并数据并写入输出文件
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)

        # 根据模式选择写入方式
        if append_mode and os.path.exists(output_file):
            merged_df.to_csv(output_file, mode='a', header=False, index=False)
            print(f"[INFO] 所有文件已追加写入至 {output_file}")
        else:
            merged_df.to_csv(output_file, index=False)
            print(f"[INFO] 所有文件已覆盖写入至 {output_file}")
    else:
        print("[WARNING] 没有找到可以合并的CSV文件。")


def process_csv_file(file_path, existing_game_ids, all_data):
    """
    处理单个 CSV 文件，过滤重复 gameID 并添加到数据列表。

    参数:
        file_path (str): CSV 文件路径。
        existing_game_ids (set): 已存在的 gameID 集合。
        all_data (list): 存储所有有效数据的列表。
    """
    try:
        df = pd.read_csv(file_path)

        if 'gameID' not in df.columns:
            print(f"[WARNING] {file_path} 缺少 gameID 字段，跳过。")
            return

        # 过滤掉已存在的 gameID 数据
        filtered_df = df[~df['gameID'].isin(existing_game_ids)]

        # 更新已处理的 gameID 列表
        new_ids = set(df['gameID'])
        existing_game_ids.update(new_ids)

        if not filtered_df.empty:
            all_data.append(filtered_df)
            print(f"[INFO] 已读取并过滤 {file_path}")
        else:
            print(f"[INFO] {file_path} 中的数据已全部存在，跳过。")

    except Exception as e:
        print(f"[ERROR] 无法读取文件 {file_path}: {e}")

# 读取 moves.csv 文件
def read_moves(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        moves = list(reader)
    return moves

# 提取每个 gameID 的基本信息
def extract_game_info(moves_data):
    games = defaultdict(list)

    # 按 gameID 分组
    for move in moves_data:
        game_id = move['gameID']
        games[game_id].append(move)

    gameinfos = []
    for game_id, moves in games.items():
        first_side = moves[0]['side']  # 第一步是谁下的
        blackID = ''
        redID = ''
        winner = moves[0]['winner']

        # 构造 gameinfo 数据
        gameinfo = {
            'gameID': game_id,
            'winner': winner
        }
        gameinfos.append(gameinfo)
    return gameinfos

# 写入 gameinfo.csv 文件
def write_game_info(gameinfos, output_file):
    fieldnames = ['gameID',  'winner']
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gameinfos)

# 主函数
def moves_add_gameinfo(moves_path='moves.csv',gameinfo_path='gameinfo.csv'):
    moves_data = read_moves(moves_path)
    gameinfos = extract_game_info(moves_data)
    write_game_info(gameinfos, gameinfo_path)
    print(f"✅ {gameinfo_path} 已成功生成！")

def merge_multiple_csv():
    folder_path = os.path.join(DATA_PATH, args.owner)
    file_path = os.path.join(folder_path, "moves.csv")
    gameinfo_path = os.path.join(folder_path, "gameinfo.csv")
    path_list = [
        "../data/moves.csv",
        "../data/moves_1.csv",
        "../data/moves_2.csv",
    ]
    merge_multiple_csv_paths([path_list], output_file=file_path)
    moves_add_gameinfo(file_path, gameinfo_path)

if __name__ == '__main__':
    # 设置参数解析器 110
    # owner t:顶级 m:大师 _m_:比赛 o:其他 u:棋友 n:网络比赛
    # 37535, 1489888 棋友上传 u
    # 121608,131607 大师棋谱 m
    # 1313710 顶级棋谱 t
    parser = argparse.ArgumentParser(description='运行棋谱抓取脚本')
    parser.add_argument('--mode', choices=['append', 'overwrite'], default='append',
                        help='运行模式: append 追加, overwrite 覆盖 (默认: append)')
    parser.add_argument('--owner', type=str, default='n')
    parser.add_argument("--start-id", type=int, default=830)
    parser.add_argument("--end-id", type=int, default=3000)
    parser.add_argument("--sort", type=int, default=1)
    parser.add_argument("--id-mode", action='store_true', default=False,
                        help="如果启用，则 start-id 和 end-id 表示具体的 gameID 范围，而非页码")
    args = parser.parse_args()

    # 创建会话对象
    session = requests.Session()
    if args.owner in ['t','o']:
        username = "busisiji"
        password = "zwc6731061"
        if not login(session, username, password):
            print("[ERROR] 登录失败，程序退出。")
            time.sleep(60)

    folder_path = os.path.join(DATA_PATH, args.owner)
    file_path = os.path.join(folder_path, "moves2.csv")
    gameinfo_path = os.path.join(folder_path, "gameinfo2.csv")

    try:
        for i in range(args.start_id, args.end_id, args.sort):
            print(f"[INFO] 正在处理第 {i} 页数据...")
            if CRAWL_NUM >= 10000:
                print(f"[INFO] 已采集 {CRAWL_NUM} 局，达到上限，停止采集。")
                break

            # 根据 id_mode 决定如何获取 url_list
            if args.id_mode:
                # 直接使用指定的 ID 范围
                url_list = [[args.owner, str(i)]]
            else:
                # 使用页码获取该页所有 ID
                url_list = getOwnerPage(args.owner, 1)
                if not url_list:
                    print(f"[ERROR] 获取第 {i} 页数据失败，请检查网络或稍后再试。")
                    time.sleep(60)
                    continue
                url_list = getAllIdsFromFirstPage(url_list, i - 1)

            if args.owner in ['t', 'o']:
                # 必须激活页面
                test_url = f"http://www.dpxq.com/hldcg/search/list.asp?owner={args.owner}&page={i}"
                response = session.get(test_url)
                if response.status_code == 200:
                    print("[INFO] 页面激活成功")
                else:
                    print(f"[ERROR] 页面激活失败，状态码：{response.status_code}")
                    continue

            for _, id_name in url_list:
                index = str(id_name)
                # 如果是追加模式，检查 gameID 是否已存在
                if args.mode == 'append' and is_game_id_exists(file_path, index):
                    print(f"[INFO] 编号 {index} 已存在，跳过。")
                    continue

                # 构造 URL
                if args.owner in ['t', 'o']:

                    url = f"http://www.dpxq.com/hldcg/search/view.asp?owner={args.owner}&id={id_name}"
                    # url = f"http://www.dpxq.com/hldcg/search/view.asp?owner={args.owner}&list.asp?owner=haidingding&page={i}&id={id_name}"
                elif args.owner == '_m_':
                    url = f"http://www.dpxq.com/hldcg/search/view{args.owner}{id_name}.html"
                else:
                    url = f"http://www.dpxq.com/hldcg/search/view.asp?owner={args.owner}&id={id_name}"

                try:
                    result = getChessManual(session, url)
                    if result[0] == 'error':
                        continue
                    save_file(result, index, file_path)
                except Exception as e:
                    print(f"[ERROR] 处理编号 {index} 时出错：{e}")
                    time.sleep(30)
                    continue
            if not args.id_mode:
                time.sleep(60)  # 每页抓取后延时防止请求频繁

        moves_add_gameinfo(file_path, gameinfo_path)

    except KeyboardInterrupt:
        print("\n[INFO] 检测到中断，程序即将退出。")
        exit()
