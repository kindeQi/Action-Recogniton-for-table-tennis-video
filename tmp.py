with open('../demo/video_list.txt', 'w') as f:
    for index in range(115, 255):
        f.write('file {}.mp4\n'.format(index))
        # print('{}.mp4'.format(index))