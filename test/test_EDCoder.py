from utils.EDCoder import Encoder, Decoder
from scipy.spatial.transform import Rotation as R

import numpy as np

encoder = Encoder(stride=5, alpha=0.2, neighbor=2)
decoder = Decoder(stride=5, alpha=0.2, neighbor=2)

ori = R.from_euler("YXZ", [-121, 25.57, 150.055], degrees=True).as_quat()
ori = np.array([ori[3], ori[0], ori[1], ori[2]])
yaw_encode, pitch_encode, roll_encode = encoder.encode_ori(ori)
yaw, pitch, roll = decoder.decode_ori(yaw_encode.unsqueeze(0), pitch_encode.unsqueeze(0), roll_encode.unsqueeze(0))
print(yaw, pitch, roll)
print(hex(id(encoder.yaw_index_dict)))
print(hex(id(decoder.yaw_index_dict)))