#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import random
import shutil
import uuid
import itertools
import threading
from queue import Queue
# 需要pillow库，安装方法： pip3 install ${name}  （terminal下）
from PIL import Image, ImageDraw, ImageFont, ImageFilter

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'ttf')     #字体文件夹放在与脚本同目录下，可以自行添加和修改字体
DEFAULT_FONTS = [os.path.join(DATA_DIR, 'DroidSansMono.ttf')]


class Captcha():

    def __init__(self, width=128, height=32, fonts=DEFAULT_FONTS, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts
        self._font_sizes = font_sizes or (26, 28, 30)
        self._true_fonts = tuple((ImageFont.truetype(f, s)
                                  for f in self._fonts
                                  for s in self._font_sizes))

    @staticmethod
    def create_noise_dots(image, number=50):
        color = Captcha.random_color(100, 238, random.randint(220, 255))
        w, h = image.size
        draw = ImageDraw.Draw(image)
        for i in range(number):
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.point((x1, y1), fill=color)
        return image

    @staticmethod
    def create_noise_curve(image):
        color = Captcha.random_color(100, 255, random.randint(220, 255))
        w, h = image.size
        x1 = random.randint(0, int(w / 5))
        x2 = random.randint(w - int(w / 5), w)
        y1 = random.randint(int(h / 5), h - int(h / 5))
        y2 = random.randint(y1, h - int(h / 5))
        points = [x1, y1, x2, y2]
        end = random.randint(160, 200)
        start = random.randint(0, 20)
        ImageDraw.Draw(image).arc(points, start, end, fill=color)
        return image

    @staticmethod
    def random_color(start, end, opacity=None):
        red = random.randint(start, end)
        green = random.randint(start, end)
        blue = random.randint(start, end)
        return (red, green, blue) if opacity is None \
            else (red, green, blue, opacity)

    def draw_image(self, chars, background):
        """Create the CAPTCHA image itself.
        :param chars: text to be generated.
        :param background:  background color.
        """
        image = Image.new('RGB', (self._width, self._height), background)

        def _draw_character(c):
            font = random.choice(self._true_fonts)
            color = self.random_color(10, 175, random.randint(220, 255))
            w, h = ImageDraw.Draw(image).textsize(c, font=font)

            dx = random.randint(0, 3)
            dy = random.randint(0, 4)
            img = Image.new('RGBA', (w + dx, h + dy))
            ImageDraw.Draw(img).text((dx, dy), c, font=font, fill=color)

            # rotate
            img = img.crop(img.getbbox())
            img = img.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)

            # warp
            dx = w * random.uniform(0.1, 0.3)
            dy = h * random.uniform(0.2, 0.3)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (x1, y1,
                    -x1, h2 - y2,
                    w2 + x2, h2 + y2,
                    w2 - x2, -y1)
            img = img.resize((w2, h2))
            img = img.transform((w, h), Image.QUAD, data)
            return img

        char_images = []
        for c in chars:
            if random.random() > 0.5:
                char_images.append(_draw_character(" "))
            char_images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in char_images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        for img in char_images:
            w, h = img.size
            mask = img.split()[3]
            image.paste(img, (offset, int((self._height - h) / 2)), mask)
            offset = offset + w + random.randint(-rand, 0)

        if width != self._width:
            image = image.resize((self._width, self._height))

        return image

    def generate_captcha_image(self, chars):
        """Generate the image of the given characters.
        :param chars: text to be generated.
        """
        background = self.random_color(238, 255)
        img = self.draw_image(chars, background)
        #self.create_noise_dots(img)
        #self.create_noise_curve(img)
        img = img.filter(ImageFilter.SMOOTH)
        return img

    def write(self, chars, output, format='png'):
        """Generate and write an image CAPTCHA data to the output.
        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        img = self.generate_captcha_image(chars)
        return img.save(output, format=format)


def get_choices(digit=True, lowercase=True, uppercase=True):
    choices = ""
    # 删除 0 O o 1 I l 变形后非常容易认错的
    digits = "23456789"
    lowercases = 'abcdefghijkmnpqrstuvwxyz'
    uppercases = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
    # 移除变形后不易辨认的字符
    if digit:
        choices += digits
    if lowercase:
        choices += lowercases
    if uppercase:
        choices += uppercases
    return choices


def _gen_captcha(img_dir, n, choices, img_queue, is_remove=True):
    if is_remove:
        # delete dir
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
    # create dir
    if not os.path.exists(img_dir):
        try:
            os.makedirs(img_dir)
        except OSError as exc:
            print("路径创建失败，请检查输入参数")
            exit(1)

    captcha = Captcha(width=80, height=32)
    print('generating %s captchas in %s' % (n, img_dir))
    candidate = tuple(itertools.permutations(choices, 4))
    c=0
    for _ in range(n):
        c=c+1
        chars = "".join(candidate[random.randint(0, len(candidate))])
        #fn = os.path.join(img_dir, '%s_%s.png' % (chars, uuid.uuid4()))
        fn = os.path.join(img_dir, '%s_%s.png' % (str(c),chars))
        img_queue.put_nowait((fn, captcha.generate_captcha_image(chars)))
        # captcha.write(chars, fn)


def _write(img_queue, format, t1):
    count = 0
    while t1.is_alive():
        while not img_queue.empty():
            fn, img = img_queue.get_nowait()
            img.save(fn, format=format)
            count += 1
            if count % 1000 is 0:
                print(f"gen {count}")


# 生成文件列表
def _gen_list(path, n):
    img_list = os.listdir(path)          # 读取目录下文件列表
    random.shuffle(img_list)             # 随机打乱
    img_path = path + "/img_list.txt"
    label_path = path + "/label.txt"
    choices = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    print('generating %s captchas list in %s' % (n, path))
    with open(img_path, "w") as f_img, open(label_path, "w") as f_label:
        for i in img_list[:n]:
            if i.split('.')[-1] == "png":
                chars = i.split('_', 2)[0]
                label = ' '.join([str(choices.index(x)) for x in chars])
                f_img.write(i + '\n')
                f_label.write(i + ' ' + label + '\n')


if __name__ == "__main__":

    info = """
            use：python3 gen_verification_code.py argv[1] argv[2] argv[3]
            argv[1]: img / list             |-> 
                    img 生成图片+列表文件；list 根据目录生成图片列表
            argv[2]: {img_path}(文件夹路径)  |-> 
                    如果argv[1]是img，则该目录为生成图片的路径；
                    如果argv[1]是list,则生成该目录下图片的列表文件，且该列表文件保存在该目录下
            argv[3]: {num}(数量，正整数)     |->
                    生成图片的数量，或者生成图片列表中的图片数量
                    如果在选择生成list时，该数量大于目录下总图片数，则生成的列表长度为总图片数，即num = min{argv[3], num of 目录下图片数}
                    
            示例:   生成图片： python3 gen_verification_code.py img ~/data/img 10000  
                        |->   在~/data/img文件夹下生成10000张验证码图片
                    生成列表： python3 gen_verification_code.py list ~/data/img 10000 
                        |->   在~/data/img文件夹下该目录下图片的一个列表文件，长度最多10000
           """
    if len(sys.argv) == 4:
        task = sys.argv[1]
        path = sys.argv[2]
        num = int(sys.argv[3])

        if task == "img":
            img_queue = Queue()
            t1 = None
            # 生成图片
            t1 = threading.Thread(target=_gen_captcha, args=(path, num, get_choices(), img_queue, True))
            # 存储图片
            t2 = threading.Thread(target=_write, args=(img_queue, "png", t1))
            # 开始线程
            t1.start()
            t2.start()
            # 等待图片生成完成
            while t2.is_alive():
                pass
            # 生成文件列表
            _gen_list(path, num)

        elif task == "list":
            # 生成文件列表
            _gen_list(path, num)

        else:
            print(info)

    else:
        print(info)