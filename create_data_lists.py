from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(voc07_path='merge2007/VOC2007',
                      voc12_path='VOCtrainval_11-May-2012/VOCdevkit/VOC2012',
                      output_folder='./')
