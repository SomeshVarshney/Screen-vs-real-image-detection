from features import FeatureExtractor

extractor = FeatureExtractor()

vec = extractor.extract("data/real/your_image.jpg")

print(vec.shape)

print(vec[:10])