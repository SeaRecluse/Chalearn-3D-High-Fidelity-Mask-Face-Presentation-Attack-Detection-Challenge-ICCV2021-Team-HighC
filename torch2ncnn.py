
import torch
import model

save_path = "./trans2onnx/"
# onnx_name = "onnx.onnx"
onnx_name = "shuffle.onnx"
c_tag = 2
INPUT_SIZE = 224
model_path = './model_best.pth.tar'

if __name__=='__main__':
    stages_out_channels = [24, 48, 96, 192, 1024]
    if c_tag == 1:
        stages_out_channels = [24, 116, 232, 464, 1024]
    elif c_tag == 1.5:
        stages_out_channels = [24, 176, 352, 704, 1024]
    elif c_tag == 2:
          stages_out_channels = [24, 244, 488, 976, 2048]

    stages_repeats = [4, 8, 4]
 
    net = model.ShuffleNetV2(stages_repeats = stages_repeats, stages_out_channels = stages_out_channels)
    net.to(device='cpu', dtype=torch.float32)

    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = net.state_dict()

    for (k,v) in checkpoint['state_dict'].items():
        print(k)
        key = k[7:]
        print(key)
        if key in state_dict:
        	state_dict[key] = v
        else:
        	state_dict.update({key : v})

    net.load_state_dict(state_dict)
    net.eval()
    dummy_input = torch.randn([1,3,INPUT_SIZE,INPUT_SIZE])
    torch.onnx.export(net, dummy_input, 
        save_path + onnx_name,
        verbose=True,
        export_params=True,
        opset_version=10)

    print("over")