class Config:
    # 实验参数
    name = "MobileSPEEDNetv4"
    seed = 13
    deterministic = False
    benchmark = True
    debug = True
    comet_api = "agcu7oeqU395peWf6NCNqnTa7"
    
    # 训练参数
    device = "cuda"
    optimizer = "AdamW"
    lr0 = 0.001
    lr_min = 0.00001
    warmup_epoch = 5
    momentume = 0.9
    weight_decay = 0.0001
    epoch = 300
    batch_size = 32
    num_workers = 8
    ALPHA = (1, 8)
    
    # 损失函数
    pos_loss_type = 'MSE'
    ori_loss_type = 'CE'
    BETA = (1, 6)
    
    # 模型参数
    pool_size = (4, 2)
    head_mlp_type = 'Mlp'
    
    # 离散欧拉角概率分布参数
    pos_dim = 3
    stride = 5
    neighbor = 1
    alpha = 0.1
    
    # 数据参数
    data_dir = "/home/zh/pythonhub/yaolu/datasets/speed"          # 数据集路径
    imgsz = (480, 768)                      # 图片尺寸
    split = (0.85, 0.15)                    # 训练集和验证集比例
    ram = True                              # 是否使用内存加载图片
    resize_first = False                    # 是否先缩放图片
    
    # 数据增强参数
    CropAndPaste_p = 0.5                    # CropAndPaste概率
    
    CropAndPadSafe_p = 0.5                  # CropAndPadSafe概率
    
    DropBlockSafe_p = 0.5                   # DropBlockSafe概率
    drop_n = 5                              # Drop数量
    
    RotateImage_p = 0.8                     # RotateImage概率
    max_rotate_angle = 180                  # 最大旋转角度
    limit_num = 5                           # 最大尝试旋转次数
    
    AlbumentationAug_p = 0.5                # AlbumentationAug概率
    
    