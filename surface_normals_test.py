from PIL import Image
import torchvision.transforms.functional as TF
import visualpriors
import subprocess

feature_type = 'normal'

# Download a test image
subprocess.call("curl -O https://raw.githubusercontent.com/StanfordVL/taskonomy/master/taskbank/assets/test.png", shell=True)

# Load image and rescale/resize to [-1,1] and 3x256x256
image = Image.open('test.png')
o_t = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
o_t = o_t.unsqueeze_(0)

# Transform to normals feature
representation = visualpriors.representation_transform(o_t, feature_type, device='cpu') # phi(o_t) in the diagram below

# Transform to normals feature and then visualize the readout
pred = visualpriors.feature_readout(o_t, feature_type, device='cpu')

# Save it
TF.to_pil_image(pred[0] / 2. + 0.5).save('test_{}_readout.png'.format(feature_type))