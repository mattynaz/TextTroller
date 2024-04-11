from attack import SimilarityPredictor

if __name__ == "__main__":
    similarity_predictor = SimilarityPredictor()
    sentences1 = ["Kurdish or Armenians being aggressive.", "Kurdish or Armenians being aggressive."]
    sentences2 = ["He saw a Kurd beat his mother.", "The Kurdish were riled up so had to find another way."]

    scores = similarity_predictor.run(sentences1, sentences2)
    print(scores)
