if __name__ == '__main__':
    lines = []
    try:
        with open('../dataset/raw/AIGC/essay1.txt', 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
    except FileNotFoundError:
        print("文件未找到")