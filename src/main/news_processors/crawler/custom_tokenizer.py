class CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger

    def __call__(self, sent) -> list:
        word_tokens = self.tagger.nouns(sent)
        result = [word for word in word_tokens if len(word) > 1]
        return result
