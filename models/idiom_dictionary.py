class IdiomTransDict:
    def __init__(self, idioms_file, translations_file):
        self.idioms = self.read_file(idioms_file)
        self.translations = self.read_file(translations_file)
        self.dictionary = self.build_dictionary()

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file]

    def build_dictionary(self):
        return dict(zip(self.idioms, self.translations))

    def get_translation(self, idiom):
        return self.dictionary.get(idiom, "Translation not found")

