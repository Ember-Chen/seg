from utils import predict, get_descriptions
from pathlib import Path

def main():
    img_save_path, res_dict = predict(Path(r'dataset/images/train/0.png'))
    print(res_dict)
    print('\n')
    res_description = get_descriptions(img_save_path)
    print(res_description)


if __name__ == '__main__':
    main()
