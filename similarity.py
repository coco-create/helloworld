from model.visil import ViSiL
from datasets import load_video

# Load the two videos from the video files
query_video = load_video('video/passageway1-c0.mp4')
target_video = load_video('video/passageway1-c1.mp4')

# Initialize ViSiL model and load pre-trained weights
model = ViSiL('ckpt/resnet/')

# Extract features of the two videos
query_features = model.extract_features(query_video, batch_sz=16)
target_features = model.extract_features(target_video, batch_sz=16)

# Calculate similarity between the two videos
similarity = model.calculate_video_similarity(query_features, target_features)