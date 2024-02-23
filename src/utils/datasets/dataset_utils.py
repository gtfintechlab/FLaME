def encode(self, label_name):
    reversed_mapping = {v: k for k, v in self.mappings[self.task].items()}
    return reversed_mapping.get(label_name, -1)

def decode(self, label_number):
    return self.mappings[self.task].get(label_number, "undefined").upper()
