from PIL import Image, ImageDraw, ImageFont
import os
import math
import json
from copy import deepcopy
font_type = ImageFont.truetype('SIMYOU.TTF', size=20)
with open('../action.json', 'r') as f:
    action = json.load(f)

start_index = -1

def softmax(l):
    sum = 0
    for item in l:
        sum += math.exp(item)
    return [math.exp(item) / sum for item in l]


if __name__ == '__main__':
    with open('../demo.json', 'r') as f:
        data = json.load(f)

    data = sorted(data, key=lambda x: int(x['info']['path'].split('/')[-1]) * 1000000 + int(x['info']['start_time']))

    a, b = 115, 255
    for video_index in range(a, b):

        # if the mp4 file all ready existed, remove it
        # if os.path.exists('../demo/{}.mp4'.format(video_index * 2)):
        #     os.remove('../demo/{}.mp4'.format(video_index * 2))
        # if os.path.exists('../demo/{}.mp4'.format(video_index * 2 - 1)):
        #     os.remove('../demo/{}.mp4'.format(video_index * 2 - 1))

        if os.path.exists('../demo/{}.mp4'.format(video_index)):
            os.remove('../demo/{}.mp4'.format(video_index))

        # write the context to the imgs
        item = data[video_index]
        for frame in range(item['info']['start_time'], item['info']['end_time'] + 1):
            img_path = os.path.join(item['info']['path'], '{}.png'.format(str(frame)))
            assert os.path.exists(img_path)

            im = Image.open(img_path)
            im = im.resize((480, 480))
            context = ImageDraw.Draw(im)

            # get the highest index
            highest = 0
            for index in range(len(item['predict'])):
                if item['predict'][index] > item['predict'][highest]:
                    highest = index

            # write the top 2 lines
            context.text(xy=(10, 75),
                         text='{0}: {1}'.format('实际动作', action[str(item['ground_truth'])]),
                         fill='white', font=font_type)
            context.text(xy=(10, 100),
                         text='{0}: {1}'.format('预测动作', action[str(highest)]),
                         fill='white', font=font_type)

            # write the context
            labels = softmax(item['predict'])
            for index, value in enumerate(labels):
                context.text(xy=(10, (index) * 25 + 250),
                             text='{:6} {:7.1f}%'.format(action[str(index)] + ':', value * 100),
                             fill='white',font=font_type)

            output_path = '/home/zhangrui/demo/' + '{}.png'.format(frame)
            im.save(output_path)

        # bash command to generate video
        # bash_command = 'ffmpeg -framerate 5 -start_number {} -i "/home/zhangrui/demo/%d.png" /home/zhangrui/demo/{}.mp4'.format(item['info']['start_time'],video_index * 2)
        bash_command = 'ffmpeg -framerate 8 -start_number {} -i "/home/zhangrui/demo/%d.png" /home/zhangrui/demo/{}.mp4'.format(item['info']['start_time'],video_index)
        os.system(bash_command)

        # remove all existed png
        for path in os.listdir('../demo'):
            if path.split('.')[-1] == 'png':
                os.remove('../demo/' + path)

        # write the video without context(video contains no action)
        # duration: [start_index, end_index)
        # if start_index == -1:
        #     start_index = item['info']['end_time'] + 1
        # else:
        #     end_index = item['info']['start_time']
        #
        #     # in case is the frame between 2 video file, so end_index is less that start index
        #     if end_index < start_index:
        #         continue
        #
        #     # just copy videos which contains no action
        #     for frame in range(start_index, end_index):
        #         img_path = os.path.join(item['info']['path'], '{}.png'.format(str(frame)))
        #         assert os.path.exists(img_path)
        #
        #         im = Image.open(img_path)
        #         im = im.resize((480, 480))
        #
        #         output_path = '/home/zhangrui/demo/' + '{}.png'.format(frame)
        #         im.save(output_path)
        #
        #     # generate video from imgs
        #     bash_command = 'ffmpeg -framerate 60 -start_number {} -i "/home/zhangrui/demo/%d.png" /home/zhangrui/demo/{}.mp4'.format(start_index, video_index * 2 - 1)
        #     os.system(bash_command)
        #
        #     # redirect the start_index
        #     start_index = item['info']['end_time'] + 1
        #
        #     # remove all existed png
        #     for path in os.listdir('../demo'):
        #         if path.split('.')[-1] == 'png':
        #             os.remove('../demo/' + path)

        print('finish {}'.format(video_index))

