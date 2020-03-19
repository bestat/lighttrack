model = None

if cuda:
    model.cuda()

model.eval()  # Set in evaluation mode


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index


def inference_frcnn(img_path):
    pass

def inference_frcnn_from_img(img):
    pass

if __name__ == "__main__":
    img_path = "/export/guanghan/PyTorch-YOLOv3/data/samples/messi.jpg"
    human_candidates = inference_frcnn(img_path)
    print("human_candidates:", human_candidates)
