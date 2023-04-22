import cv2,time
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    colors = scipy.io.loadmat('./data/color150.mat')['colors']
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
    #         print(f'{names[index+1]}:')

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)
    # im_vis = numpy.concatenate((img, pred_color), axis=1)
    return pred_color

def get_segment():
    names = {}
    with open('./data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights='./ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='./ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.cuda()  # 如支持cuda,则使用cuda()
    return segmentation_module

def run(segmentation_module,clip_file):
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    pil_image = PIL.Image.open(clip_file) \
        .convert('RGB')
    img_original = numpy.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    singleton_batch = {'img_data': img_data[None].cuda()}
    output_size = img_data.shape[1:]

    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(singleton_batch, segSize=output_size)

    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    visualize_result(img_original, pred)

    # Top classes in answer
    predicted_classes = numpy.bincount(pred.flatten()).argsort()[::-1]
    # print(predicted_classes)
    id_use = list(predicted_classes).index(2)

    c = predicted_classes[id_use]
    vs = visualize_result(img_original, pred, c)

    bw = cv2.cvtColor(vs, cv2.COLOR_BGR2RGB)
    cv2.imwrite(split_file, bw)

def mk_folders():
    """创建4个新的文件夹"""
    folders = ['2.streetview_clip', '3.streetview_split']
    for folder in folders:
        if os.path.exists('../data/{}'.format(folder)):
            print('文件已存在')
        else:
            os.mkdir('../data/{}'.format(folder))
            print('{}创建'.format(folder))

def clip_img():
    """对街景图片进行切割，去掉拍摄车"""
    # print(src_file)
    img_panorama = cv2.imread(src_file, 1)
    img_panorama = img_panorama[0:256, :]  # 去掉

    cv2.imwrite(clip_file, img_panorama)
    print('{} 处理完成'.format(clip_file))
    return clip_file

if __name__ == '__main__':
    t1 = time.time()
    original_path = '../data/1.streetview_original/2013'  # 街景图片所在的文件夹

    clip_path = '../data/2.streetview_clip/2013'          # 街景图片裁剪后所在的文件夹

    split_path = '../data/3.streetview_split/2013'        # 街景图片语义分割之后所在的文件夹

    # 判断文件夹是否存在并创建
    mk_folders()
    # 初始化模型
    sg = get_segment()
    # 获取已经处理过的图像编号
    exist_list = []
    for _,_,exist_images in os.walk(split_path):
        for exist_image in exist_images:
            exist_list.append(exist_image)

    for i,v,m in os.walk(original_path):
        for image in m:
            # 判断数据是否已经处理过, 处理过则跳过
            if image in exist_list:
                print('该图片 %s 已经处理过' % (image))
                continue

            src_file = "{}/{}".format(original_path, image)
            clip_file = "{}/{}".format(clip_path, image)
            split_file = "{}/{}".format(split_path, image)
            clip_img()
            run(sg,clip_file)
    t2 = time.time()
    print('耗时{:.2f}秒'.format(t2 - t1))